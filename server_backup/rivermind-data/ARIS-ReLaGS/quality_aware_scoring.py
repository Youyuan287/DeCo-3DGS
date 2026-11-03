"""Minimal training-free quality-aware candidate scoring for THGS.

The module is deliberately independent of GT and model training.  It only
uses hierarchy features, child features, and Gaussian support counts already
stored in ``sai_nag.pt``.
"""
from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = _to_numpy(x).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def _size_score(counts: np.ndarray) -> np.ndarray:
    values = np.log1p(_to_numpy(counts).astype(np.float32))
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def quality_from_features(
    candidate_features: np.ndarray,
    child_features: list[np.ndarray],
    gaussian_counts: np.ndarray,
) -> np.ndarray:
    """Return a bounded quality score in [0.5, 1.0] for each candidate.

    ``child_features[i]`` contains the lower-level semantic features belonging
    to candidate ``i``.  The quality score is the fixed equal-weight average
    of log-support size and candidate-to-child feature agreement, mapped to a
    conservative multiplicative factor [0.5, 1].
    """
    candidates = _unit_rows(candidate_features)
    sizes = _size_score(gaussian_counts)
    agreement = np.zeros(len(candidates), dtype=np.float32)
    for i, children in enumerate(child_features):
        children = _to_numpy(children).astype(np.float32)
        if children.size == 0:
            agreement[i] = 0.5
            continue
        child_mean = _unit_rows(children).mean(axis=0, keepdims=True)
        child_mean = _unit_rows(child_mean)[0]
        agreement[i] = float(np.clip((candidates[i] @ child_mean + 1.0) * 0.5, 0.0, 1.0))
    return (0.5 + 0.5 * (sizes + agreement) * 0.5).astype(np.float32)


def quality_from_hierarchy(features: list, labels: list, level: int) -> np.ndarray:
    """Compute quality for one hierarchy level using only frozen ``sai_nag`` data."""
    current_features = _to_numpy(features[level - 1]).astype(np.float32)
    current_labels = _to_numpy(labels[level])
    child_features = _to_numpy(features[level - 2]).astype(np.float32)
    child_labels = _to_numpy(labels[level - 1])
    counts = np.bincount(current_labels.astype(np.int64), minlength=len(current_features))
    child_groups = []
    for candidate_id in range(len(current_features)):
        child_ids = np.unique(child_labels[current_labels == candidate_id]).astype(np.int64)
        child_groups.append(child_features[child_ids])
    return quality_from_features(current_features, child_groups, counts)


def combine_scores(semantic_scores: np.ndarray, quality_scores: np.ndarray, scoring_mode: str) -> np.ndarray:
    semantic_scores = np.asarray(semantic_scores, dtype=np.float32)
    if scoring_mode == "baseline":
        return semantic_scores.copy()
    if scoring_mode == "quality":
        return semantic_scores * np.asarray(quality_scores, dtype=np.float32)
    raise ValueError(f"unknown scoring_mode: {scoring_mode}")


def _ordered(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(i)))


def select_top3_quality(semantic_by_level: dict, quality_by_level: dict, scoring_mode: str, topk: int = 3):
    """Replicate THGS's per-level pool then final top-k selection."""
    final_scores = {
        level: combine_scores(semantic_by_level[level], quality_by_level[level], scoring_mode)
        for level in sorted(semantic_by_level)
    }
    pool = []
    for level in sorted(final_scores):
        pool.extend((level, idx) for idx in _ordered(final_scores[level])[:topk])
    pool_scores = {(level, idx): float(final_scores[level][idx]) for level, idx in pool}
    selected = sorted(pool, key=lambda token: (-pool_scores[token], int(token[0]), int(token[1])))[:topk]
    return selected, pool
