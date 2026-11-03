"""Pure cache-only spatial support primitives. No GT, model loading or I/O."""
import numpy as np


def validate_run_inputs(cache_frames, query_by_image, historical_rows):
    cache_frames = set(cache_frames)
    if not set(query_by_image).issubset(cache_frames):
        raise ValueError('Evaluation frames cannot be matched to cache frames')
    rows = [r for r in historical_rows if r.get('scene') == 'figurines']
    required = {'scene','image','prompt','iou','pred_px'}
    if not rows or any(not required.issubset(r) for r in rows):
        raise ValueError('Missing historical fields')
    keys = [(r['image'],r['prompt']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate historical query keys')
    expected = {(im,q) for im,qs in query_by_image.items() for q in qs}
    if not expected.issubset(set(keys)):
        raise ValueError('Incomplete historical query coverage')
    return sorted(cache_frames - set(query_by_image))


def select_feature(scores, seg):
    scores, seg = np.asarray(scores), np.asarray(seg)
    ids = np.unique(seg[seg >= 0])
    if not len(ids) or ids.max() >= len(scores):
        raise ValueError('No valid proposal masks or invalid feature indices')
    return int(ids[np.argmax(scores[ids])])


def mask_iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Mask shape mismatch')
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def evidence_scores(ious, visible):
    ious, visible = np.asarray(ious, float), np.asarray(visible, bool)
    if ious.shape != visible.shape or ious.ndim != 1:
        raise ValueError('Evidence must be matched one-dimensional arrays')
    if not np.isfinite(ious).all() or ((ious < 0) | (ious > 1)).any():
        raise ValueError('Invalid IoU')
    if not len(ious):
        return 0.0, 0.0
    return float(visible.mean()), float(np.where(visible, ious, 0).mean())


def gap_select(candidates, scores):
    """Original ReLaGS prefix rule, including empty single/tied candidate cases."""
    if len(candidates) != len(scores):
        raise ValueError('Candidates and scores differ in length')
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    gaps = [0.0] + [ranked[i][1] - ranked[i+1][1] for i in range(len(ranked)-1)]
    cut = gaps.index(max(gaps))
    return [item[0] for item in ranked[:cut]]


def rescore(base_scores, evidence, all_unobserved=False):
    if len(base_scores) != len(evidence):
        raise ValueError('Evidence and candidates differ in length')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in evidence):
        raise ValueError('Invalid evidence')
    if all_unobserved:
        return list(base_scores)
    return [float(s * e) for s, e in zip(base_scores, evidence)]


def proposal_permutation(seg, n_features, seed):
    """Feature ID -> mask ID derangement within view/level/log2(area) bins.

    Keeps the mask bank and score bank unchanged, NOT each selected mask's exact
    area. Singleton bins are unchanged and must be included in coverage reports.
    Feature IDs must be level-disjoint as in the actual LangSplat cache.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3 or n_features < 0:
        raise ValueError('Expected level x height x width segmentation')
    if (seg < -1).any() or (seg >= n_features).any():
        raise ValueError('Segmentation ID outside feature table')
    rng = np.random.default_rng(seed)
    mapping = np.arange(n_features)
    seen = set()
    eligible = moved = singleton = 0
    areas = np.zeros(n_features, dtype=np.int64)
    for layer in seg:
        ids, counts = np.unique(layer[layer >= 0], return_counts=True)
        if seen.intersection(ids.tolist()):
            raise ValueError('Feature IDs shared across levels')
        seen.update(ids.tolist())
        areas[ids] = counts
        bins = np.floor(np.log2(counts)).astype(int)
        eligible += len(ids)
        for bin_id in np.unique(bins):
            group = ids[bins == bin_id].copy()
            if len(group) == 1:
                singleton += 1
                continue
            rng.shuffle(group)
            mapping[group] = np.roll(group, 1)
            moved += len(group)
    return mapping, {
        'eligible_features': eligible, 'moved_features': moved,
        'singleton_features': singleton,
        'total_mask_area_preserved': bool(areas.sum() == areas[mapping].sum()),
    }
