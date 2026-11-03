"""Fixed two-factor M3 ablation built on the certified V2 evidence readout."""
from __future__ import annotations

import torch

from evidence_readout_v2 import auxiliary_correction


def factorial_correction(main_scores, slot_scores, bank, eligible, reliability,
                         scope, gain, selected_root_ids, temperature=0.08,
                         coefficient=0.25):
    """Apply exactly one fixed range/gain combination.

    `all` uses the static scene bank roots as its comparison set.  `selected`
    uses the original DeCo candidates supplied before any correction.  Neither
    set uses GT nor is recomputed after corrected scores are produced.
    """
    if scope not in {"all", "selected"} or gain not in {"absolute", "relative"}:
        raise ValueError((scope, gain))
    _, detail = auxiliary_correction(
        main_scores, slot_scores, bank, eligible, reliability, temperature, coefficient
    )
    bank_ids = torch.as_tensor(bank["region_ids"], device=main_scores.device, dtype=torch.long)
    full_a = torch.zeros_like(main_scores)
    full_a[bank_ids] = detail["delta"]
    if scope == "all":
        target = torch.unique(bank_ids, sorted=True)
    else:
        target = torch.as_tensor(
            sorted({int(x) for x in selected_root_ids}),
            device=main_scores.device,
            dtype=torch.long,
        )
        target = target[(target >= 0) & (target < len(main_scores))]

    full_correction = torch.zeros_like(main_scores)
    full_relative = torch.zeros_like(main_scores)
    if len(target):
        if gain == "absolute":
            full_correction[target] = torch.clamp(
                coefficient * full_a[target], min=0.0, max=coefficient
            )
        else:
            mean_a = full_a[target].mean()
            full_relative[target] = torch.clamp(full_a[target] - mean_a, min=0.0)
            full_correction[target] = torch.clamp(
                coefficient * full_relative[target], min=0.0, max=coefficient
            )
    detail["unconditional_delta"] = detail["delta"].clone()
    detail["delta"] = (full_relative if gain == "relative" else full_a)[bank_ids]
    detail["correction"] = full_correction[bank_ids]
    detail["comparison_count"] = int(len(target))
    return main_scores + full_correction, detail
