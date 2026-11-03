"""Non-GT structural check for one production observation file."""
import hashlib
import json
import sys

import numpy as np

from multi_evidence import EvidenceConfig, build_bank, l2n, save_npz


def main(observation_path, output_path, report_path):
    with np.load(observation_path, allow_pickle=False) as data:
        features = data["features"]
        weights = data["weights"]
        regions = data["region_ids"]
        views = data["view_ids"]
    # Canonical non-data vector is irrelevant to raw mode, but makes its input explicit.
    u = np.ones(features.shape[1], dtype=np.float32)
    u /= np.linalg.norm(u)
    k1 = build_bank(features, weights, regions, u, EvidenceConfig(max_evidence=1), mode="raw")
    k3 = build_bank(features, weights, regions, u, EvidenceConfig(max_evidence=3), mode="raw")
    direct = []
    for region_id in k1.region_ids:
        idx = np.where((regions == region_id) & (weights > 0))[0]
        a = weights[idx] / weights[idx].sum()
        direct.append(l2n((a[:, None] * l2n(features[idx])).sum(axis=0)))
    direct = np.stack(direct)
    max_abs = float(np.max(np.abs(direct - k1.base_features)))
    if max_abs > 1e-6:
        raise AssertionError("K=1 differs from direct weighted aggregation: %g" % max_abs)
    save_npz(output_path, k3, metadata={"mode": "raw", "k": 3,
                                          "observation_sha256": hashlib.sha256(open(observation_path, "rb").read()).hexdigest()})
    result = {"status": "STRUCTURAL_ONLY_NO_TEXT_NO_DEV_DIRECTION",
              "observations": int(len(features)), "regions": int(len(k1.region_ids)),
              "views": int(len(np.unique(views))), "weight_min": float(weights.min()),
              "weight_max": float(weights.max()), "k1_max_abs_error": max_abs,
              "b2_avg_groups": float(k3.group_count.mean()),
              "b2_group_count_min": int(k3.group_count.min()), "b2_group_count_max": int(k3.group_count.max())}
    with open(report_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:])
