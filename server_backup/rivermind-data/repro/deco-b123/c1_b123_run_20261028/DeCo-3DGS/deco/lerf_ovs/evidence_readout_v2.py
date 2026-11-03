"""Conservative M1--M3 auxiliary readout for the V2 evaluator.

The DeCo root score remains the primary score.  This module only computes a
non-negative, capped correction from valid multi-view mode evidence; it never
replaces the root score with a mode average.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for z in iter(lambda: f.read(1 << 20), b""):
            h.update(z)
    return h.hexdigest()


def load_bank_v2(path, nag_path, root_count, dim, observations_path=None, members_path=None):
    """Load and validate an existing bank; never mutate or rebuild it."""
    with np.load(path, allow_pickle=False) as d:
        required = ["features", "priors", "valid", "base_features", "region_ids", "metadata"]
        for key in required:
            if key not in d:
                raise ValueError(f"missing bank key {key}")
        bank = {k: d[k].copy() for k in required if k != "metadata"}
        meta = json.loads(str(d["metadata"].item()))
    x, priors, valid, region_ids = (bank["features"], bank["priors"],
                                    bank["valid"].astype(bool), bank["region_ids"])
    if x.ndim != 3 or x.shape[-1] != dim or priors.shape != x.shape[:2] or valid.shape != priors.shape:
        raise ValueError("bank shape mismatch")
    if len(np.unique(region_ids)) != len(region_ids) or (region_ids < 0).any() or (region_ids >= root_count).any():
        raise ValueError("invalid bank region IDs")
    if not np.isfinite(x).all() or not np.isfinite(priors).all() or (priors < 0).any():
        raise ValueError("non-finite or negative bank values")
    if not np.allclose(priors.sum(1), 1, atol=1e-4):
        raise ValueError("bank priors do not sum to one")
    if meta.get("nag_sha256") != sha256(nag_path):
        raise ValueError("NAG SHA256 mismatch")
    if observations_path is not None:
        obs_sha = sha256(observations_path)
        if meta.get("observation_sha256") != obs_sha:
            raise ValueError("observation SHA256 mismatch")
    bank["valid"] = valid
    bank["metadata"] = meta
    if members_path is not None and not Path(members_path).exists():
        raise ValueError("members file missing")
    return bank, meta


def load_mode_support(observations_path, members_path, bank):
    """Return query-independent mode reliability and multi-view eligibility."""
    with np.load(observations_path, allow_pickle=False) as d:
        obs_features = np.asarray(d["features"], dtype=np.float32)
        view_ids = np.asarray(d["view_ids"], dtype=np.int64)
    payload = json.loads(Path(members_path).read_text())
    memberships = payload["memberships"]
    R, K = bank["priors"].shape
    region_to_row = {int(r): i for i, r in enumerate(bank["region_ids"])}
    reliability = np.ones((R, K), dtype=np.float32)
    view_count = np.zeros((R, K), dtype=np.int32)
    consistency = np.ones((R, K), dtype=np.float32)
    for m in memberships:
        rid, slot = int(m["region_id"]), int(m["slot"])
        if rid not in region_to_row or slot >= K:
            continue
        ri = region_to_row[rid]
        idx = np.asarray(m.get("observation_indices", []), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(obs_features))]
        nviews = len(np.unique(view_ids[idx]))
        view_count[ri, slot] = nviews
        if nviews >= 2 and len(idx) >= 2:
            x = obs_features[idx]
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            sim = x @ x.T
            tri = sim[np.triu_indices(len(x), 1)]
            # Mean pairwise semantic agreement, conservatively clamped.
            c = float(np.mean(tri)) if len(tri) else 0.0
            consistency[ri, slot] = float(np.clip(c, 0.0, 1.0))
        else:
            consistency[ri, slot] = 1.0
    return reliability, consistency, view_count, (view_count >= 2)


def aggregate_advantage(mode_scores, main_score, mode_priors, eligible_mask,
                        method="softmax", temperature=0.08, reliability=None):
    """Aggregate positive mode advantages without changing canon scoring.

    Inputs are shaped ``[R,K]``, ``[R]``, ``[R,K]`` and ``[R,K]``.  The
    optional reliability argument is retained for the existing V2 readout;
    DeCo-LI experiments pass no reliability (M2 is off).
    """
    if method not in {"softmax", "maxsim", "prior_maxsim"}:
        raise ValueError(f"unknown aggregation method: {method}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = mode_scores
    main = main_score.reshape(-1)
    priors = mode_priors
    eligible = eligible_mask.to(dtype=torch.bool)
    if scores.ndim != 2 or priors.shape != scores.shape or eligible.shape != scores.shape:
        raise ValueError("mode score, prior and eligible shapes must match [R,K]")
    advantage = torch.clamp(scores - main[:, None], min=0)
    if reliability is not None:
        rel = reliability.to(device=scores.device, dtype=scores.dtype)
        if rel.shape != scores.shape:
            raise ValueError("reliability shape must match [R,K]")
    else:
        rel = torch.ones_like(scores)
    valid = eligible & torch.isfinite(scores) & torch.isfinite(priors)
    valid &= priors >= 0
    if method == "maxsim":
        out = advantage.masked_fill(~valid, -torch.inf).max(dim=1).values
    elif method == "prior_maxsim":
        positive = valid & (priors > 0)
        pmax = priors.masked_fill(~positive, 0).max(dim=1).values
        relative = torch.zeros_like(priors)
        nonzero = pmax > 0
        relative[nonzero] = priors[nonzero] / pmax[nonzero, None]
        weighted = relative * advantage * rel
        out = weighted.masked_fill(~positive, -torch.inf).max(dim=1).values
    else:
        logits = scores / float(temperature) + torch.log(priors.clamp_min(1e-12))
        logits = logits.masked_fill(~valid, -torch.inf)
        weights = torch.zeros_like(logits)
        any_valid = valid.any(dim=1)
        if any_valid.any():
            weights[any_valid] = torch.softmax(logits[any_valid], dim=1)
        out = (weights * rel * advantage).sum(dim=1)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def auxiliary_correction(main_scores, slot_scores, bank, eligible, reliability,
                         temperature=0.08, coefficient=0.25, method="softmax"):
    """Compute `main + capped(nonnegative delta)` without replacing main scores."""
    if temperature <= 0 or coefficient < 0:
        raise ValueError("invalid V2 correction parameters")
    priors = torch.as_tensor(bank["priors"], device=slot_scores.device, dtype=slot_scores.dtype)
    valid = torch.as_tensor(bank["valid"], device=slot_scores.device, dtype=torch.bool)
    elig = torch.as_tensor(eligible, device=slot_scores.device, dtype=torch.bool) & valid
    rel = torch.as_tensor(reliability, device=slot_scores.device, dtype=slot_scores.dtype)
    region_ids = torch.as_tensor(bank["region_ids"], device=slot_scores.device, dtype=torch.long)
    root_main = main_scores[region_ids]
    advantage = torch.clamp(slot_scores - root_main[:, None], min=0)
    delta = aggregate_advantage(slot_scores, root_main, priors, elig,
                                method=method, temperature=temperature,
                                reliability=rel)
    logits = slot_scores / float(temperature) + torch.log(priors.clamp_min(1e-12))
    logits = logits.masked_fill(~elig, -torch.inf)
    weights = torch.zeros_like(logits)
    any_valid = elig.any(dim=1)
    if any_valid.any() and method == "softmax":
        weights[any_valid] = torch.softmax(logits[any_valid], dim=1)
    correction = torch.clamp(float(coefficient) * delta, min=0.0, max=float(coefficient))
    new_root = main_scores.clone()
    new_root[region_ids] = root_main + correction
    return new_root, {
        "advantage": advantage,
        "query_weights": weights,
        "aggregation_method": method,
        "reliability": rel,
        "eligible": elig,
        "delta": delta,
        "correction": correction,
    }


def descriptors_v2(bank, device, center=None, renorm=False):
    """Move bank features to the evaluator device using the production C1 mean."""
    x = torch.from_numpy(bank["features"]).to(device=device, dtype=torch.float32)
    if center is not None:
        x = x - center.reshape(1, 1, -1)
        if renorm:
            x = torch.nn.functional.normalize(x, dim=-1)
    return x.reshape(-1, x.shape[-1])
