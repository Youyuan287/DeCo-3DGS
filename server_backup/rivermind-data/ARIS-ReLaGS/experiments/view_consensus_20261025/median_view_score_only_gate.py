#!/usr/bin/env python3
"""GT-free median per-view text-score gate; no rendering."""

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
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return jsonable(x.detach().cpu().tolist())
    return x


def load_eval(path, repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("exact_eval", path)
    if spec is None or spec.loader is None: raise RuntimeError("FAIL_CLOSED: evaluator")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def gap_select(candidates, scores):
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), tuple(x[0])))
    gaps = ([0.0] + [ranked[i][1] - ranked[i + 1][1]
             for i in range(len(ranked) - 1)]) if ranked else []
    return [x[0] for x in ranked[:gaps.index(max(gaps))]] if gaps else []


def validate_pool(pool, labels):
    for i, row in enumerate(pool):
        if len(row) < 4 or not isinstance(row[0], numbers.Integral) or int(row[0]) not in (2, 3):
            raise RuntimeError(f"FAIL_CLOSED: invalid tuple level at {i}")
        if not isinstance(row[1], numbers.Real) or not isinstance(row[2], numbers.Real):
            raise RuntimeError(f"FAIL_CLOSED: invalid tuple score at {i}")
        limit = int(labels[int(row[0])].max()) + 1
        if not isinstance(row[3], numbers.Integral) or not 0 <= int(row[3]) < limit:
            raise RuntimeError(f"FAIL_CLOSED: invalid tuple candidate_id at {i}")


def union_record(labels, selected):
    mask = np.zeros(labels[0].shape[0], dtype=bool)
    for row in selected: mask |= labels[int(row[0])] == int(row[3])
    ids = np.flatnonzero(mask).astype(np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest(), int(ids.size)


def main():
    ap = argparse.ArgumentParser()
    for name in ("r0", "bank_dir", "raw_score_pool", "queries", "evaluator", "repro_root", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("FAIL_CLOSED: GPU unavailable")
    if args.out.exists(): raise RuntimeError("FAIL_CLOSED: refusing existing output")
    queries, raw = load_json(args.queries), load_json(args.raw_score_pool)
    if len(queries) != 56 or len(raw) != 56: raise RuntimeError("FAIL_CLOSED: expected 56 queries")
    if any(set(q) != {"image", "prompt"} for q in queries): raise RuntimeError("FAIL_CLOSED: query schema")
    bank = {l: np.load(args.bank_dir / f"bank_l{l}.npy", mmap_mode="r") for l in (1, 2, 3)}
    expected = {1: (299, 2102, 512), 2: (299, 467, 512), 3: (299, 179, 512)}
    for l, shape in expected.items():
        if tuple(bank[l].shape) != shape or bank[l].dtype != np.float32 or not np.isfinite(bank[l]).all():
            raise RuntimeError(f"FAIL_CLOSED: bank contract level {l}")
    evaluator = load_eval(args.evaluator, args.repro_root)
    obj = torch.load(args.r0, map_location="cpu")
    labels_t, feats = obj["nag"], [x.detach().cpu() for x in obj["nag_feat"]]
    labels = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in labels_t]
    snag = evaluator.SemanticNAG.__new__(evaluator.SemanticNAG)
    snag.labels, snag.feat, snag.gaussian_num = labels_t, feats, labels[0].shape[0]
    vlm = evaluator.ClipSimMeasure(); vlm.load_model(); device = torch.device(args.device)
    records = []
    with torch.no_grad():
        for qi, query in enumerate(queries):
            sims = [torch.zeros(len(f), dtype=torch.float32) for f in feats]
            for row in raw[qi]["pool"]: sims[int(row["level"]) - 1][int(row["candidate_id"])] = float(row["raw_score"])
            ids, vals = snag._get_root_candidates(sims, 50)
            roots = snag._build_root_candidates_with_leaf_info(ids, vals, sims, len(labels) - 2)[:5]
            pool = snag._analyze_leaf_candidates(roots, len(labels) - 2)
            validate_pool(pool, labels)
            old = gap_select(pool, [float(x[1]) for x in pool])
            vlm.encode_text(query["prompt"])
            view_scores, supports = [], []
            for row in pool:
                level, cid = int(row[0]), int(row[3]); x = np.asarray(bank[level][:, cid, :])
                norm = np.linalg.norm(x, axis=1); keep = norm > 0
                if not np.any(keep): raise RuntimeError(f"FAIL_CLOSED: zero support {level}:{cid}")
                x = x[keep] / norm[keep, None]
                scores = vlm.compute_similarity(torch.from_numpy(x.astype(np.float32)).to(device))
                if not scores.is_cuda: raise RuntimeError("FAIL_CLOSED: non-CUDA score output")
                view_scores.append(float(torch.median(scores.detach().float()).cpu()))
                supports.append(int(keep.sum()))
            new = gap_select(pool, view_scores)
            old_h, old_n = union_record(labels, old); new_h, new_n = union_record(labels, new)
            records.append({"query_index": qi, "image": query["image"], "prompt": query["prompt"],
                            "pool_size": len(pool), "old_selected": old, "new_selected": new,
                            "old_union_sha256": old_h, "new_union_sha256": new_h,
                            "old_union_count": old_n, "new_union_count": new_n,
                            "top1_retained": bool(old and new and (old[0][0], old[0][3]) == (new[0][0], new[0][3])),
                            "support_counts": supports, "old_scores": [float(x[1]) for x in pool],
                            "median_view_scores": view_scores})
    src = Path(__file__).resolve()
    result = {"status": "MEDIAN_VIEW_SCORE_ONLY_GATE", "gt_opened": False, "rendering": False,
              "rule": "median exact per-view text score over nonzero bank views", "selector": "official gap_select",
              "records": records, "inputs_sha256": {str(p): sha256(p) for p in (args.r0, args.raw_score_pool, args.queries, args.evaluator)},
              "bank_sha256": {str(args.bank_dir / f"bank_l{l}.npy"): sha256(args.bank_dir / f"bank_l{l}.npy") for l in (1, 2, 3)},
              "source_sha256": {str(src): sha256(src)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f: json.dump(jsonable(result), f, indent=2)


if __name__ == "__main__": main()
