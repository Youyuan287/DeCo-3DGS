"""One preregistered M3 rule. No GT, learned thresholds, or bank mutation."""
import torch
from evidence_readout_v2 import auxiliary_correction


def relative_correction(main_scores, slot_scores, bank, eligible, reliability,
                        comparison_ids, temperature=0.08, coefficient=0.25):
    _, detail = auxiliary_correction(main_scores, slot_scores, bank, eligible,
                                     reliability, temperature, coefficient)
    ids = torch.as_tensor(bank['region_ids'], device=main_scores.device).long()
    full_a = torch.zeros_like(main_scores)
    full_a[ids] = detail['delta']
    selected = sorted(set(int(x) for x in comparison_ids))
    correction = torch.zeros_like(main_scores)
    relative = torch.zeros_like(main_scores)
    if selected:
        idx = torch.tensor(selected, device=main_scores.device).long()
        relative[idx] = torch.clamp(full_a[idx] - full_a[idx].mean(), min=0)
        correction[idx] = torch.clamp(coefficient * relative[idx], 0, coefficient)
    detail['unconditional_delta'] = detail['delta'].clone()
    detail['delta'] = relative[ids]
    detail['correction'] = correction[ids]
    return main_scores + correction, detail
