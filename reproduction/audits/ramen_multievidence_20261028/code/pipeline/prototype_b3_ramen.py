"""Explicitly non-certified same-scene, no-GT B3 structural prototype."""
import hashlib
import json
import sys

import numpy as np

from multi_evidence import EvidenceConfig, build_bank, l2n, save_npz


def main(observation_path, raw_path, debiased_path, report_path):
    with np.load(observation_path, allow_pickle=False) as data:
        f, w, r = data["features"], data["weights"], data["region_ids"]
    u = l2n(l2n(f).mean(axis=0))
    cfg = EvidenceConfig(max_evidence=3, min_group_weight=.05, temperature=.08)
    raw = build_bank(f, w, r, u, cfg, mode="raw")
    debiased = build_bank(f, w, r, u, cfg, mode="debiased")
    source_hash = hashlib.sha256(open(observation_path, "rb").read()).hexdigest()
    meta = {"status": "PROTOTYPE_UNVERIFIED_SAME_SCENE_DIRECTION",
            "observation_sha256": source_hash,
            "direction_sha256": hashlib.sha256(u.tobytes()).hexdigest()}
    save_npz(raw_path, raw, metadata={**meta, "mode": "raw"})
    save_npz(debiased_path, debiased, metadata={**meta, "mode": "debiased"})
    out = {**meta, "regions": int(len(raw.region_ids)),
           "raw_avg_groups": float(raw.group_count.mean()),
           "debiased_avg_groups": float(debiased.group_count.mean()),
           "regions_with_different_group_count": int(np.count_nonzero(raw.group_count != debiased.group_count)),
           "u_norm": float(np.linalg.norm(u))}
    with open(report_path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:])
