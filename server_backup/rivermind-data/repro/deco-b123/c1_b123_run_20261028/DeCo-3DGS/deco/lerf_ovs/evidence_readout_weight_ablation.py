"""Fixed-prior versus query-adaptive M3 readout, with all other V2 terms frozen."""
from __future__ import annotations

import torch

from evidence_readout_v2 import auxiliary_correction


def fixed_weight_correction(main_scores, slot_scores, bank, eligible, reliability,
                            temperature=0.08, coefficient=0.25):
    """Use normalized query-independent priors, but query-dependent advantages."""
    _, detail = auxiliary_correction(
        main_scores, slot_scores, bank, eligible, reliability,
        temperature=temperature, coefficient=coefficient,
    )
    priors = torch.as_tensor(bank["priors"], device=slot_scores.device,
                             dtype=slot_scores.dtype)
    valid = torch.as_tensor(bank["valid"], device=slot_scores.device, dtype=torch.bool)
    elig = torch.as_tensor(eligible, device=slot_scores.device, dtype=torch.bool) & valid
    fixed_weights = priors * elig.to(priors.dtype)
    fixed_weights = fixed_weights / fixed_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    advantage = detail["advantage"]
    delta = (fixed_weights * detail["reliability"] * advantage).sum(dim=1)
    correction = torch.clamp(float(coefficient) * delta, min=0.0, max=float(coefficient))
    region_ids = torch.as_tensor(bank["region_ids"], device=main_scores.device, dtype=torch.long)
    root_main = main_scores[region_ids]
    new_root = main_scores.clone()
    new_root[region_ids] = root_main + correction
    detail.update({
        "query_weights": fixed_weights,
        "delta": delta,
        "correction": correction,
        "weight_mode": "fixed_prior",
    })
    return new_root, detail
