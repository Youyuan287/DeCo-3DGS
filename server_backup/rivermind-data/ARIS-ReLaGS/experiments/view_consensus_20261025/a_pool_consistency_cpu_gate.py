#!/usr/bin/env python3
"""GT-free A-pool consistency gate; no execution on import.

The candidate pool is rebuilt by the exact SemanticNAG root/leaf methods from
the R0 evaluator.  Both arms use the official gap_select; only the scores in
the same A pre-gap pool differ.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_json(p):
    with Path(p).open(encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(value):
    """Normalize evaluator/NumPy scalars before writing the audit artifact."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def load_eval(path, repo):
    import sys
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("exact_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("FAIL_CLOSED: exact evaluator unavailable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gap_select(candidates, scores):
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), tuple(x[0])))
    gaps = ([0.0] + [ranked[i][1] - ranked[i + 1][1]
             for i in range(len(ranked) - 1)]) if ranked else []
    cut = gaps.index(max(gaps)) if gaps else 0
    return [x[0] for x in ranked[:cut]]


def consistency(bank, level, cid):
    x = np.asarray(bank[level][:, cid, :], dtype=np.float32)
    n = np.linalg.norm(x, axis=1)
    x = x[n > 0]
    if len(x) == 0:
        return 0.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    m = x.mean(axis=0)
    mn = np.linalg.norm(m)
    return float(np.mean(x @ (m / mn))) if mn else 0.0


def union_hash(labels, selected):
    mask = np.zeros(labels[0].shape[0], dtype=bool)
    for row in selected:
        mask |= labels[int(row[0])] == int(row[2])
    ids = np.flatnonzero(mask).astype(np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest(), int(ids.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--raw-score-pool", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--repro-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: CUDA visible in CPU gate")
    if args.out.exists():
        raise RuntimeError("FAIL_CLOSED: refusing existing output")

    queries = load_json(args.queries)
    if len(queries) != 56 or any(set(q) != {"image", "prompt"} for q in queries):
        raise RuntimeError("FAIL_CLOSED: invalid frozen query manifest")
    raw = load_json(args.raw_score_pool)
    if len(raw) != 56:
        raise RuntimeError("FAIL_CLOSED: raw pool/query count mismatch")
    bank = {level: np.load(args.bank_dir / f"bank_l{level}.npy", mmap_mode="r")
            for level in (1, 2, 3)}
    expected = {1: (299, 2102, 512), 2: (299, 467, 512), 3: (299, 179, 512)}
    for level, shape in expected.items():
        if tuple(bank[level].shape) != shape:
            raise RuntimeError(f"FAIL_CLOSED: bank shape mismatch at level {level}")

    evaluator = load_eval(args.evaluator, args.repro_root)
    obj = torch.load(args.r0, map_location="cpu")
    labels = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in obj["nag"]]
    feats = [x.detach().cpu() for x in obj["nag_feat"]]
    # Storage-only construction: invoke exact methods, but refuse any CUDA move.
    try:
        snag = evaluator.SemanticNAG.__new__(evaluator.SemanticNAG)
        snag.labels, snag.feat = obj["nag"], feats
        snag.gaussian_num = labels[0].shape[0]
    except Exception as e:
        raise RuntimeError(f"FAIL_CLOSED: exact SemanticNAG shim unavailable: {e}")

    records = []
    for qi, q in enumerate(queries):
        rows = raw[qi]["pool"]
        sims = [torch.zeros(len(f), dtype=torch.float32) for f in feats]
        for row in rows:
            level, cid = int(row["level"]), int(row["candidate_id"])
            if level not in (2, 3):
                raise RuntimeError("FAIL_CLOSED: raw pool contains non-A level")
            sims[level - 1][cid] = float(row["raw_score"])
        ids, vals = snag._get_root_candidates(sims, 50)
        roots = snag._build_root_candidates_with_leaf_info(ids, vals, sims, len(labels) - 2)[:5]
        pool = snag._analyze_leaf_candidates(roots, len(labels) - 2)
        if not pool:
            old, new = [], []
        else:
            old_scores = [float(x[3]) for x in pool]
            old = gap_select(pool, old_scores)
            new_scores = []
            for x in pool:
                level, cid = int(x[0]), int(x[2])
                new_scores.append(float(x[3]) * consistency(bank, level, cid))
            new = gap_select(pool, new_scores)
        old_h, old_n = union_hash(labels, old)
        new_h, new_n = union_hash(labels, new)
        records.append({"query_index": qi, "image": q["image"], "prompt": q["prompt"],
                        "pool_size": len(pool), "old_selected": old, "new_selected": new,
                        "old_union_sha256": old_h, "new_union_sha256": new_h,
                        "old_union_count": old_n, "new_union_count": new_n,
                        "top1_retained": bool(old and new and old[0][:3] == new[0][:3])})
    result = {"status": "CPU_A_POOL_CONSISTENCY_GATE", "gt_opened": False,
              "rule": "raw_score*mean_view_cosine", "selector": "exact official gap_select",
              "records": records, "inputs_sha256": {
                  str(args.r0): sha256(args.r0), str(args.raw_score_pool): sha256(args.raw_score_pool),
                  str(args.queries): sha256(args.queries), str(args.evaluator): sha256(args.evaluator)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)


if __name__ == "__main__":
    main()
