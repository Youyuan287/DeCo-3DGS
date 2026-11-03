#!/usr/bin/env python3
"""Audit Gaussian/NAG/CSR spatial support without changing any inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def nag_labels(path: Path):
    d = torch.load(path, map_location="cpu")
    # The frozen .pt stores the four per-Gaussian label tensors directly.
    # Reading them avoids importing the optional SPT notebook stack.
    return [np.asarray(x.detach().cpu().numpy(), dtype=np.int64) for x in d["nag"]]


def audit_scene(scene, obs_path, nag_paths, members_path, bank_path, cases_path):
    with np.load(obs_path, allow_pickle=False) as z:
        obs_regions = np.asarray(z["region_ids"], dtype=np.int64)
        views = np.asarray(z["view_ids"], dtype=np.int64)
        off = np.asarray(z["visible_gaussian_offsets"], dtype=np.int64)
        gids = np.asarray(z["visible_gaussian_ids"], dtype=np.int64)
    members = json.loads(Path(members_path).read_text())["memberships"]
    with np.load(bank_path, allow_pickle=False) as z:
        bank_regions = np.asarray(z["region_ids"], dtype=np.int64)
    cases = pd.read_csv(cases_path)
    cases = cases[cases.scene == scene]
    records = []
    for nag_path in nag_paths:
        labs = nag_labels(Path(nag_path))
        level = labs[-1]
        for _, c in cases.iterrows():
            rid = int(c.region_id)
            mode = int(c.mode_id)
            mm = [x for x in members if int(x["region_id"]) == rid and int(x["slot"]) == mode]
            mode_obs = np.asarray(mm[0]["observation_indices"], dtype=np.int64) if mm else np.asarray([], dtype=np.int64)
            mode_ids = np.unique(np.concatenate([gids[off[i]:off[i + 1]] for i in mode_obs])) if len(mode_obs) else np.asarray([], dtype=np.int64)
            root_ids = np.flatnonzero(level == rid).astype(np.int64)
            outside = np.setdiff1d(mode_ids, root_ids)
            obs_region_values = np.unique(obs_regions[mode_obs]) if len(mode_obs) else np.asarray([], dtype=np.int64)
            bad_obs = int(np.sum(obs_regions[mode_obs] != rid)) if len(mode_obs) else 0
            valid_ids = int(np.sum((mode_ids >= 0) & (mode_ids < len(level))))
            records.append(dict(scene=scene, case_group=c.case_group, image=c.image, query=c["query"], region_id=rid, mode_id=mode, nag=str(nag_path), nag_sha256=sha256(Path(nag_path)), gaussian_count=len(level), root_gaussian_count=len(root_ids), mode_gaussian_count=len(mode_ids), outside_gaussian_count=len(outside), mode_ids_out_of_range=int(len(mode_ids)-valid_ids), obs_count=len(mode_obs), obs_region_ids=";".join(map(str, obs_region_values.tolist())), obs_region_mismatch=bad_obs, bank_region_present=bool(rid in set(bank_regions)), level_unique_ids=int(len(np.unique(level))), level_min=int(level.min()), level_max=int(level.max())))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", action="append", nargs=6, metavar=("NAME", "OBS", "NAG1", "NAG2", "MEM", "BANK"), help="scene name, obs, two candidate NAG paths, membership, bank")
    args = ap.parse_args()
    rows = []
    for spec in args.scene or []:
        name, obs, nag1, nag2, mem, bank = spec
        rows += audit_scene(name, Path(obs), [nag1, nag2], Path(mem), Path(bank), args.cases)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
