#!/usr/bin/env python3
"""Controlled raw-grouping bank. It only changes the grouping space h=f."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evidence_modes import load_observations, norm, weighted_overlap


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_raw(obs_path, nag_path, out_path, max_modes=3, geo_weight=.25):
    f, w, region_ids, view_ids, ptr, visible = load_observations(
        obs_path, require_geometry=True
    )
    f = norm(f)
    # The sole experimental change from debiased B1: h is raw normalized CLIP.
    h = f
    regions = np.unique(region_ids)
    R, K, D = len(regions), max_modes, f.shape[1]
    feat = np.zeros((R, K, D), np.float32)
    residuals = np.zeros_like(feat)
    priors = np.zeros((R, K), np.float32)
    valid = np.zeros((R, K), bool)
    base = np.zeros((R, D), np.float32)
    group_count = np.zeros(R, np.int32)
    memberships = []
    for z, rid in enumerate(regions):
        ix = np.flatnonzero(region_ids == rid)
        a = w[ix] / w[ix].sum()
        ff, hh = f[ix], h[ix]
        base[z] = norm((a[:, None] * ff).sum(0))
        if len(ix) > 1 and K > 1:
            semantic = hh @ hh.T
            overlap = weighted_overlap(ix, ptr, visible)
            similarity = (1 - geo_weight) * semantic + geo_weight * (2 * overlap - 1)
            centers = [int(np.argmax(similarity @ a))]
            for _ in range(min(K, len(ix)) - 1):
                novelty = 1 - np.max(similarity[:, centers], axis=1)
                novelty[centers] = -np.inf
                candidate = int(np.argmax(novelty))
                if novelty[candidate] <= 1e-6:
                    break
                centers.append(candidate)
            assign = np.argmax(similarity[:, centers], axis=1)
        else:
            assign = np.zeros(len(ix), dtype=np.int64)
        groups = []
        for k in range(int(assign.max()) + 1):
            members = np.flatnonzero(assign == k)
            if len(members):
                groups.append((float(a[members].sum()), members))
        if not groups:
            groups = [(1.0, np.arange(len(ix)))]
        groups.sort(key=lambda x: -x[0])
        for k, (prior, members) in enumerate(groups):
            priors[z, k] = prior
            feat[z, k] = norm((a[members, None] * ff[members]).sum(0))
            residuals[z, k] = norm((a[members, None] * hh[members]).sum(0))
            valid[z, k] = True
            memberships.append({
                "region_id": int(rid), "slot": int(k),
                "view_ids": view_ids[ix[members]].tolist(),
                "observation_indices": ix[members].tolist(),
                "n_before_rofa": int(len(members)), "n_after_rofa": int(len(members)),
            })
        priors[z] /= priors[z].sum()
        group_count[z] = len(groups)
    metadata = {
        "observation_sha256": file_sha256(obs_path),
        "nag_sha256": file_sha256(nag_path),
        "source_observation": str(Path(obs_path).resolve()),
        "config": {
            "max_modes": max_modes, "min_mass": 0.0, "b1": True, "b2": False,
            "grouping": "raw", "geo_weight": geo_weight,
            "geometry_required": True, "rofa_tau": 3.0,
        },
        "effective_geometry": True, "b1_enabled": True, "b2_enabled": False,
        "dropped_observation_count": 0,
        "controlled_change": "h=f for mode discovery; final feat uses the same original normalized CLIP f",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, features=feat, residuals=residuals, priors=priors, valid=valid,
        base_features=base, group_count=group_count, region_ids=regions,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    out_path.with_suffix(".members.json").write_text(
        json.dumps({"metadata": metadata, "memberships": memberships}, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-modes", type=int, default=3)
    ap.add_argument("--geo-weight", type=float, default=.25)
    args = ap.parse_args()
    if args.max_modes != 3 or args.geo_weight != .25:
        raise ValueError("This controlled ablation fixes max_modes=3 and geo_weight=.25")
    print(json.dumps(build_raw(args.observations, args.nag, args.out), indent=2))


if __name__ == "__main__":
    main()
