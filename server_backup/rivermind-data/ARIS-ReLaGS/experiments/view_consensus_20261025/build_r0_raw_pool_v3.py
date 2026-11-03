#!/usr/bin/env python3
"""Build the frozen-R0 raw-score pool with GPU text scoring only.

This is preprocessing, not a method experiment: no rendering, GT, candidate
selection, threshold tuning, or NAG reconstruction is performed.  The only
GPU operation is the project's exact ClipSimMeasure text model and similarity
calculation.  SemanticNAG is not instantiated because it is only a storage
container whose constructor forces CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_queries(path: Path):
    rows = load_json(path)
    if not isinstance(rows, list) or len(rows) != 56:
        raise RuntimeError("FAIL_CLOSED: expected exactly 56 frozen queries")
    for row in rows:
        if set(row) != {"image", "prompt"}:
            raise RuntimeError("FAIL_CLOSED: query manifest must contain exactly image,prompt")
    return rows


def make_mapping(labels, level):
    arr = labels[level].detach().cpu().numpy().astype(np.int64, copy=False)
    count = int(arr.max()) + 1
    return {f"{level}:{cid}": np.flatnonzero(arr == cid).astype(np.int64).tolist()
            for cid in range(count)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--repro-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_CLOSED: requested exact text-scoring GPU is unavailable")
    if args.out.exists():
        raise RuntimeError(f"FAIL_CLOSED: refusing existing output {args.out}")

    queries = load_queries(args.queries)
    evaluator = load_exact(args.evaluator, args.repro_root)
    obj = torch.load(args.r0, map_location="cpu")
    labels, feats = obj["nag"], obj["nag_feat"]
    if len(labels) < 4 or len(feats) < 3:
        raise RuntimeError("FAIL_CLOSED: incomplete R0 hierarchy")
    if [int(labels[2].max()) + 1, int(labels[3].max()) + 1] != [467, 179]:
        raise RuntimeError("FAIL_CLOSED: expected 467/179 level-2/3 nodes")

    # Storage-only bypass; no SemanticNAG constructor and no renderer.
    vlm = evaluator.ClipSimMeasure()
    vlm.load_model()
    device = torch.device(args.device)
    gpu_feats = [feats[level - 1].detach().to(device=device, dtype=torch.float32)
                 for level in (2, 3)]
    if any(not x.is_cuda for x in gpu_feats):
        raise RuntimeError("FAIL_CLOSED: exact score features did not reach CUDA")

    pool = []
    with torch.no_grad():
        for qi, query in enumerate(queries):
            vlm.encode_text(query["prompt"])
            candidates = []
            for level, feature in zip((2, 3), gpu_feats):
                scores = vlm.compute_similarity(feature)
                if not getattr(scores, "is_cuda", False):
                    raise RuntimeError("FAIL_CLOSED: exact score output is not CUDA-backed")
                scores = scores.detach().cpu().numpy().reshape(-1)
                expected = 467 if level == 2 else 179
                if scores.shape[0] != expected:
                    raise RuntimeError(f"FAIL_CLOSED: unexpected score width at level {level}")
                for cid, score in enumerate(scores):
                    candidates.append({"level": level, "candidate_id": cid,
                                       "raw_score": float(score)})
            if len(candidates) != 646:
                raise RuntimeError("FAIL_CLOSED: candidate count is not 646")
            pool.append({"query_index": qi, "image": query["image"],
                         "prompt": query["prompt"], "pool": candidates})

    mapping = {"nodes": {**make_mapping(labels, 2), **make_mapping(labels, 3)}}
    args.out.mkdir(parents=True)
    pool_path, map_path = args.out / "r0_raw_score_pool.json", args.out / "r0_node_gaussians.json"
    with pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    manifest = {
        "status": "PREPROCESSING_RAW_POOL_AND_MAPPING_V3",
        "preprocessing_only": True, "gt_opened": False, "rendering": False,
        "semantic_nag_instantiated": False, "cuda_device": args.device,
        "query_count": 56, "candidate_count_per_query": 646,
        "levels": {"2": 467, "3": 179},
        "text_interface": "exact ClipSimMeasure.encode_text(prompt); compute_similarity(nag_feat[level-1])",
        "inputs_sha256": {str(p): sha256(p) for p in (args.r0, args.queries, args.evaluator)},
        "outputs_sha256": {str(pool_path): sha256(pool_path), str(map_path): sha256(map_path)},
        "source": str(Path(__file__).resolve()), "source_sha256": sha256(Path(__file__).resolve()),
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
