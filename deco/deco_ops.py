"""DeCo: the two operators, in one place.

The ScanNet harness works in numpy, the two LERF harnesses work in torch. Both call the functions
below, so the repository contains exactly one definition of the method.

    C1  centre the region descriptors   f_i <- f_i - mean_i(f)          (per scene, per level)
    C2  centre the text bank and marginalise over its words
            t   <- l2n(t - mean_class_set(t))
            s_i = tau * logmeanexp_{w in B} <f_i, w>                    (tau = 0.02)

`renorm` is off by default. Turning it on (f <- l2n(f - mean)) makes the operator *active* on
single-query retrieval, where the plain version is a constant shift and therefore inert; see
PROTOCOLS.md section 4. The certified ScanNet numbers are 58.55 without it and 57.85 with it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TAU = 0.02


# --------------------------------------------------------------------------- numpy
def l2n_np(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def center_np(feats, renorm=False):
    """C1 for a list of per-level feature matrices, each (N_level, D)."""
    out = [f - f.mean(axis=0, keepdims=True) for f in feats]
    return [l2n_np(f) for f in out] if renorm else out


def text_center_np(bank, mean_over=None, eps=1e-12):
    """C2a. `bank` is (B, D); `mean_over` is the (D,) mean of the class set the bank competes in."""
    bank = l2n_np(np.atleast_2d(bank))
    if mean_over is None:
        return bank
    return l2n_np(bank - np.asarray(mean_over).reshape(1, -1))


def lse_np(sims, tau=DEFAULT_TAU):
    """C2b. `sims` is (B, N) similarities of N regions against B words; returns (N,)."""
    m = sims.max(axis=0, keepdims=True)
    return (tau * np.log(np.exp((sims - m) / tau).mean(axis=0, keepdims=True)) + m).squeeze(0)


def score_np(feats, bank, tau=DEFAULT_TAU):
    """Convenience: C1 + C2 applied to a per-level bank, returns a list of (N_level,) score vectors."""
    return [lse_np(np.stack([f @ w for w in bank]), tau) for f in feats]


# --------------------------------------------------------------------------- torch
def l2n_torch(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def center_torch(feats, renorm=False):
    out = [f - f.mean(dim=0, keepdim=True) for f in feats]
    return [l2n_torch(f) for f in out] if renorm else out


def text_center_torch(bank, mean_over=None, eps=1e-12):
    bank = l2n_torch(bank)
    if mean_over is None:
        return bank
    return l2n_torch(bank - mean_over.reshape(1, -1))


def lse_torch(sims, tau=DEFAULT_TAU):
    """`sims` is (B, N), same convention as lse_np."""
    import torch

    m = sims.max(dim=0, keepdim=True).values
    return (tau * ((sims - m) / tau).exp().mean(dim=0, keepdim=True).log() + m).squeeze(0)


def score_torch(feats, bank, tau=DEFAULT_TAU):
    """"(N, D) @ (B, D)^T -> transpose to (B, N) before marginalising."""
    return [lse_torch((f @ bank.T).T, tau) for f in feats]


if __name__ == "__main__":
    # the two implementations must agree bit for bit on the same input
    import torch

    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 32)).astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rng.normal(size=(5, 32)).astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    M = B.mean(0)

    a1, b1 = center_np([A])[0], text_center_np(B, M)
    a2 = center_torch([torch.from_numpy(A)])[0]
    b2 = text_center_torch(torch.from_numpy(B), torch.from_numpy(M))
    s1 = lse_np(np.stack([a1 @ w for w in b1]))
    s2 = lse_torch((a2 @ b2.T).T).numpy()
    print("center  max|np-torch| =", float(np.abs(a1 - a2.numpy()).max()))
    print("text    max|np-torch| =", float(np.abs(b1 - b2.numpy()).max()))
    print("score   max|np-torch| =", float(np.abs(s1 - s2).max()))