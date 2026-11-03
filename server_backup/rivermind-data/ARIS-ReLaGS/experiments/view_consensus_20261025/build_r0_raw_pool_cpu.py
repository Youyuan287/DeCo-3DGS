#!/usr/bin/env python3
"""Build the complete frozen-R0 level-2/3 raw-score pool on CPU.

The script deliberately uses the project's exact ClipSimMeasure interface;
it does not implement a replacement scorer.  CUDA is hidden before torch is
imported.  If the exact evaluator/text model cannot run on CPU, this exits
instead of falling back to a proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_exact(path: Path, repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("exact_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("FAIL_CLOSED: cannot load exact evaluator")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_queries(path: Path):
    rows = load_json(path)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("FAIL_CLOSED: query manifest is not a non-empty list")
    for row in rows:
        if set(row) != {"image", "prompt"}:
            raise RuntimeError("FAIL_CLOSED: query manifest must contain exactly image,prompt")
    return rows


def node_gaussians(labels, level: int):
    arr = labels[level].detach().cpu().numpy().astype(np.int64, copy=False)
    k = int(arr.max()) + 1
    out = {}
    for cid in range(k):
        out[f"{level}:{cid}"] = np.flatnonzero(arr == cid).astype(np.int64).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--repro-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: CUDA is visible; refusing CPU-only builder")
    if args.out.exists():
        raise RuntimeError(f"FAIL_CLOSED: refusing existing output {args.out}")

    queries = load_queries(args.queries)
    evaluator = load_exact(args.evaluator, args.repro_root)
    obj = torch.load(args.r0, map_location="cpu")
    labels = obj["nag"]
    feats = obj["nag_feat"]
    if len(labels) < 4 or len(feats) < 3:
        raise RuntimeError("FAIL_CLOSED: incomplete R0 hierarchy")
    if [int(labels[2].max()) + 1, int(labels[3].max()) + 1] != [467, 179]:
        raise RuntimeError("FAIL_CLOSED: expected 467/179 level-2/3 nodes")
    if any(getattr(f, "is_cuda", False) for f in feats):
        raise RuntimeError("FAIL_CLOSED: R0 feature tensor is on CUDA")

    snag = evaluator.SemanticNAG(labels, feats)
    vlm = evaluator.ClipSimMeasure()
    vlm.load_model()
    if torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: exact text model made CUDA visible")

    pool = []
    for qi, query in enumerate(queries):
        vlm.encode_text(query["prompt"])
        sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
        if any(getattr(s, "is_cuda", False) for s in sims):
            raise RuntimeError("FAIL_CLOSED: similarity output is on CUDA")
        candidates = []
        for level in (2, 3):
            scores = sims[level - 1].detach().cpu().numpy().reshape(-1)
            if scores.shape[0] != [467, 179][level - 2]:
                raise RuntimeError("FAIL_CLOSED: unexpected exact score width")
            for cid, score in enumerate(scores):
                candidates.append({"level": level, "candidate_id": cid,
                                   "raw_score": float(score)})
        pool.append({"query_index": qi, "image": query["image"],
                     "prompt": query["prompt"], "pool": candidates})

    mapping = {"nodes": {**node_gaussians(labels, 2), **node_gaussians(labels, 3)}}
    args.out.mkdir(parents=True)
    pool_path = args.out / "r0_raw_score_pool.json"
    map_path = args.out / "r0_node_gaussians.json"
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    manifest = {
        "status": "CPU_RAW_POOL_AND_MAPPING",
        "gt_opened": False,
        "cuda_visible_devices": "",
        "query_count": len(queries),
        "candidate_count_per_query": 646,
        "levels": {"2": 467, "3": 179},
        "text_interface": "exact ClipSimMeasure.encode_text(prompt); compute_similarity(nag_feat[level-1])",
        "inputs_sha256": {str(p): sha256(p) for p in (args.r0, args.queries, args.evaluator)},
        "outputs_sha256": {str(pool_path): sha256(pool_path), str(map_path): sha256(map_path)},
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
