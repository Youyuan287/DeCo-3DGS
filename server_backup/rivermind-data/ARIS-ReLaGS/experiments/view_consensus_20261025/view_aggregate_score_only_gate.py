#!/usr/bin/env python3
"""Score-only view-aggregate gate; no rendering or GT access.

The historical A pool is rebuilt from frozen R0 raw scores using the exact
SemanticNAG root/leaf methods (via a CPU storage shim) and official gap_select.
Only the score in the same pool is replaced by exact ClipSimMeasure scoring of
the normalized mean of nonzero per-view bank features.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import numbers
import sys
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


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return jsonable(x.detach().cpu().tolist())
    return x


def load_eval(path, repo):
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


def union_record(labels, selected):
    mask = np.zeros(labels[0].shape[0], dtype=bool)
    for row in selected:
        mask |= labels[int(row[0])] == int(row[3])
    ids = np.flatnonzero(mask).astype(np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest(), int(ids.size)


def validate_pool(pool, labels):
    """Assert the exact evaluator tuple schema: [level, score, score, cid, ...]."""
    for i, row in enumerate(pool):
        if len(row) < 4:
            raise RuntimeError(f"FAIL_CLOSED: short A-pool tuple at {i}")
        level = row[0]
        if not isinstance(level, numbers.Integral) or int(level) not in (2, 3):
            raise RuntimeError(f"FAIL_CLOSED: invalid level field at {i}: {level!r}")
        for pos in (1, 2):
            if not isinstance(row[pos], numbers.Real):
                raise RuntimeError(f"FAIL_CLOSED: score field {pos} is not numeric at {i}")
        cid = row[3]
        limit = int(labels[int(level)].max()) + 1
        if not isinstance(cid, numbers.Integral) or not (0 <= int(cid) < limit):
            raise RuntimeError(f"FAIL_CLOSED: invalid candidate_id at {i}: {cid!r}")


def aggregate(bank, level, cid):
    x = np.asarray(bank[level][:, cid, :], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    present = norms > 0
    if not np.any(present):
        raise RuntimeError(f"FAIL_CLOSED: selected node has zero view support {level}:{cid}")
    x = x[present] / norms[present, None]
    mean = x.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm == 0:
        raise RuntimeError(f"FAIL_CLOSED: zero aggregate feature {level}:{cid}")
    return mean / mean_norm, int(present.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--raw-score-pool", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--repro-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: exact text-score GPU unavailable")
    if args.out.exists():
        raise RuntimeError("FAIL_CLOSED: refusing existing output")

    queries = load_json(args.queries)
    raw = load_json(args.raw_score_pool)
    if len(queries) != 56 or len(raw) != 56:
        raise RuntimeError("FAIL_CLOSED: expected 56 frozen queries/raw pools")
    if any(set(q) != {"image", "prompt"} for q in queries):
        raise RuntimeError("FAIL_CLOSED: query manifest is not exactly image,prompt")
    bank = {level: np.load(args.bank_dir / f"bank_l{level}.npy", mmap_mode="r")
            for level in (1, 2, 3)}
    expected = {1: (299, 2102, 512), 2: (299, 467, 512), 3: (299, 179, 512)}
    for level, shape in expected.items():
        if tuple(bank[level].shape) != shape or bank[level].dtype != np.float32:
            raise RuntimeError(f"FAIL_CLOSED: bank contract mismatch at level {level}")
        if not np.isfinite(bank[level]).all():
            raise RuntimeError(f"FAIL_CLOSED: non-finite bank at level {level}")

    evaluator = load_eval(args.evaluator, args.repro_root)
    obj = torch.load(args.r0, map_location="cpu")
    labels_t, feats = obj["nag"], [x.detach().cpu() for x in obj["nag_feat"]]
    labels = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in labels_t]
    if [int(labels[2].max()) + 1, int(labels[3].max()) + 1] != [467, 179]:
        raise RuntimeError("FAIL_CLOSED: R0 level counts mismatch")
    snag = evaluator.SemanticNAG.__new__(evaluator.SemanticNAG)
    snag.labels, snag.feat = labels_t, feats
    snag.gaussian_num = labels[0].shape[0]
    vlm = evaluator.ClipSimMeasure()
    vlm.load_model()
    device = torch.device(args.device)
    records = []
    with torch.no_grad():
        for qi, query in enumerate(queries):
            rows = raw[qi]["pool"]
            sims = [torch.zeros(len(f), dtype=torch.float32) for f in feats]
            for row in rows:
                level, cid = int(row["level"]), int(row["candidate_id"])
                sims[level - 1][cid] = float(row["raw_score"])
            ids, vals = snag._get_root_candidates(sims, 50)
            roots = snag._build_root_candidates_with_leaf_info(ids, vals, sims, len(labels) - 2)[:5]
            pool = snag._analyze_leaf_candidates(roots, len(labels) - 2)
            validate_pool(pool, labels)
            # In this exact A-pool schema x[1] is the frozen raw score; x[2]
            # is the second score field and is validated above, while x[3] is
            # the candidate ID.
            old_scores = [float(x[1]) for x in pool]
            old = gap_select(pool, old_scores)
            agg = [aggregate(bank, int(x[0]), int(x[3])) for x in pool]
            features = torch.from_numpy(np.stack([x[0] for x in agg])).to(device=device)
            vlm.encode_text(query["prompt"])
            view_scores = vlm.compute_similarity(features).detach().cpu().numpy().reshape(-1)
            new = gap_select(pool, [float(x) for x in view_scores])
            old_h, old_n = union_record(labels, old)
            new_h, new_n = union_record(labels, new)
            records.append({"query_index": qi, "image": query["image"],
                            "prompt": query["prompt"], "pool_size": len(pool),
                            "old_selected": old, "new_selected": new,
                            "old_union_sha256": old_h, "new_union_sha256": new_h,
                            "old_union_count": old_n, "new_union_count": new_n,
                            "top1_retained": bool(old and new and
                                                   (int(old[0][0]), int(old[0][3])) ==
                                                   (int(new[0][0]), int(new[0][3]))),
                            "support_counts": [int(x[1]) for x in agg],
                            "old_scores": old_scores,
                            "view_scores": view_scores.tolist()})
    source_path = Path(__file__).resolve()
    result = {"status": "VIEW_AGGREGATE_SCORE_ONLY_GATE", "gt_opened": False,
              "rendering": False, "rule": "exact text score on normalized mean nonzero bank views",
              "selector": "official gap_select", "records": records,
              "inputs_sha256": {str(args.r0): sha256(args.r0), str(args.raw_score_pool): sha256(args.raw_score_pool),
                                str(args.queries): sha256(args.queries), str(args.evaluator): sha256(args.evaluator)},
              "bank_sha256": {str(args.bank_dir / f"bank_l{l}.npy"): sha256(args.bank_dir / f"bank_l{l}.npy")
                              for l in (1, 2, 3)},
              "source_sha256": {str(source_path): sha256(source_path)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(jsonable(result), f, indent=2)


if __name__ == "__main__":
    main()
