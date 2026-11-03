#!/usr/bin/env python3
"""CPU-only, GT-free per-view consistency gate.

This file intentionally does not run on import.  It consumes a verified bank,
the frozen R0 query record, a public image/prompt manifest, a raw-score pool,
and an explicit R0 node->Gaussian mapping.  Missing contracts fail closed.

The sole preregistered rule is:
    adjusted_score = frozen_text_score * mean_view_cosine(node_evidence)
where mean_view_cosine is computed over nonzero bank observations for the
candidate's (level, candidate_id).  No threshold or learned parameter exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_SHAPES = {1: (299, 2102, 512), 2: (299, 467, 512), 3: (299, 179, 512)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def node_key(level: int, candidate_id: int) -> str:
    return f"{int(level)}:{int(candidate_id)}"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise RuntimeError("FAIL_CLOSED: " + message)


def load_bank(bank_dir: Path):
    banks = {}
    for level, expected in EXPECTED_SHAPES.items():
        path = bank_dir / f"bank_l{level}.npy"
        if not path.is_file():
            fail(f"missing {path}")
        arr = np.load(path, mmap_mode="r")
        if tuple(arr.shape) != expected or arr.dtype != np.float32:
            fail(f"bad bank contract {path}: shape={arr.shape}, dtype={arr.dtype}")
        if not np.isfinite(arr).all():
            fail(f"non-finite bank values in {path}")
        banks[level] = arr
    done = bank_dir / "capture_done.json"
    if not done.is_file():
        fail("missing capture_done.json")
    meta = load_json(done)
    for item in meta.get("banks", []):
        path = Path(item["path"])
        local_path = path if path.exists() else bank_dir / path.name
        if not local_path.exists():
            fail(f"bank listed in capture_done is unavailable: {path.name}")
        if path.name.startswith("bank_l") and sha256(local_path) != item.get("sha256"):
            fail(f"bank hash mismatch: {local_path}")
    return banks, meta


def load_queries(path: Path):
    rows = load_json(path)
    if not isinstance(rows, list) or not rows:
        fail("queries must be a non-empty list")
    for row in rows:
        if set(row) != {"image", "prompt"}:
            fail("queries must contain exactly image,prompt (no GT/score fields)")
    return rows


def normalize_pool(raw_pool, query_count):
    """Normalize only records with frozen text scores; reject selected-only pools."""
    if not isinstance(raw_pool, list) or len(raw_pool) != query_count:
        fail("raw-score pool must have one entry per frozen query")
    out = []
    for qi, item in enumerate(raw_pool):
        candidates = item.get("pool") if isinstance(item, dict) else None
        if not isinstance(candidates, list) or not candidates:
            fail(f"query {qi} has no complete raw-score pool")
        normalized = []
        for c in candidates:
            if not all(k in c for k in ("level", "candidate_id", "raw_score")):
                fail(f"query {qi} candidate lacks level/candidate_id/raw_score")
            normalized.append({
                "level": int(c["level"]),
                "candidate_id": int(c["candidate_id"]),
                "raw_score": float(c["raw_score"]),
            })
        out.append(normalized)
    return out


def consistency(bank, level: int, candidate_id: int) -> float:
    x = np.asarray(bank[level][:, candidate_id, :], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    present = norms > 0
    if not np.any(present):
        return 0.0
    x = x[present] / norms[present, None]
    mean = x.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm == 0:
        return 0.0
    return float(np.mean(x @ (mean / mean_norm)))


def load_node_gaussians(path: Path):
    raw = load_json(path)
    mapping = raw.get("nodes", raw) if isinstance(raw, dict) else raw
    if not isinstance(mapping, dict) or not mapping:
        fail("node_gaussians must be a non-empty {level:candidate_id: [gaussian_ids]} mapping")
    out = {}
    for key, values in mapping.items():
        if not isinstance(values, list) or any(int(v) < 0 for v in values):
            fail(f"invalid gaussian list for {key}")
        out[str(key)] = tuple(sorted(set(int(v) for v in values)))
    return out


def union_hash(ids):
    arr = np.asarray(sorted(set(ids)), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest(), int(arr.size)


def selected_from_r0(per_query, key):
    rows = per_query[key].get("A", per_query[key]) if isinstance(per_query[key], dict) else per_query[key]
    selected = rows.get("selected", []) if isinstance(rows, dict) else []
    # Official records are [level, root_id, candidate_id].
    return [(int(x[0]), int(x[2])) for x in selected]


def official_gap_select(candidates, scores):
    """Exact gap_select semantics from full_nag_r0_validation.py."""
    if len(candidates) != len(scores):
        fail("candidate/score mismatch")
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), tuple(x[0])))
    gaps = ([0.0] + [ranked[i][1] - ranked[i + 1][1]
             for i in range(len(ranked) - 1)]) if ranked else []
    cut = gaps.index(max(gaps)) if gaps else 0
    return [x[0] for x in ranked[:cut]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--per-query", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--raw-score-pool", type=Path, required=True)
    ap.add_argument("--node-gaussians", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    banks, bank_meta = load_bank(args.bank_dir)
    queries = load_queries(args.queries)
    per_query = load_json(args.per_query)
    if isinstance(per_query, dict):
        per_query = per_query.get("queries", per_query.get("value", per_query))
    if not isinstance(per_query, list) or len(per_query) != len(queries):
        fail("per-query record count does not match frozen queries")
    raw_pool = normalize_pool(load_json(args.raw_score_pool), len(queries))
    node_gaussians = load_node_gaussians(args.node_gaussians)

    records = []
    changed = 0
    top1_retained = 0
    for qi, (q, pool, old) in enumerate(zip(queries, raw_pool, per_query)):
        scored = []
        for c in pool:
            key = node_key(c["level"], c["candidate_id"])
            if c["level"] not in banks or c["candidate_id"] >= banks[c["level"]].shape[1]:
                fail(f"query {qi} has invalid node {key}")
            if key not in node_gaussians:
                fail(f"missing R0 node->Gaussian mapping for {key}")
            cv = consistency(banks, c["level"], c["candidate_id"])
            scored.append({**c, "cross_view_consistency": cv,
                           "adjusted_score": c["raw_score"] * cv})
        ranked = sorted(scored, key=lambda x: (-x["adjusted_score"], x["level"], x["candidate_id"]))
        tuples = [(x["level"], x["candidate_id"], x["candidate_id"],
                   x["raw_score"], len(node_gaussians[node_key(x["level"], x["candidate_id"])]))
                  for x in scored]
        adjusted = [x["adjusted_score"] for x in scored]
        selected_tuples = official_gap_select(tuples, adjusted)
        new_selected = [(int(x[0]), int(x[2])) for x in selected_tuples]
        old_selected = selected_from_r0([old], 0)
        old_ids = [g for k in old_selected for g in node_gaussians[node_key(*k)]]
        new_ids = [g for k in new_selected for g in node_gaussians[node_key(*k)]]
        old_hash, old_count = union_hash(old_ids)
        new_hash, new_count = union_hash(new_ids)
        old_top = old_selected[0] if old_selected else None
        if old_top and new_selected and old_top == new_selected[0]:
            top1_retained += 1
        if old_hash != new_hash:
            changed += 1
        records.append({"query_index": qi, "image": q["image"], "prompt": q["prompt"],
                        "old_selected": old_selected, "new_selected": new_selected,
                        "old_union_sha256": old_hash, "new_union_sha256": new_hash,
                        "old_union_count": old_count, "new_union_count": new_count,
                        "top1_retained": bool(old_top and new_selected and old_top == new_selected[0]),
                        "candidate_scores": ranked})

    result = {"rule": "adjusted_score=raw_text_score*mean_view_cosine",
              "gt_opened": False, "query_count": len(queries),
              "union_changed_queries": changed, "top1_retained": top1_retained,
              "records": records, "bank_meta": bank_meta}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
