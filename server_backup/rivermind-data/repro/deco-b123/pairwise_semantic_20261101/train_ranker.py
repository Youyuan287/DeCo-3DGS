"""Frozen CPU LOSO RankNet probe on the original A candidate pool.

S=real scalar evidence; J=S plus true query/feature product; Q=S plus a
split-local deranged-query product. Every arm keeps the original g_A.
--sanity imports and validates all data and checks a toy numerical gradient;
it neither fits real heads nor writes research outputs. --full fits four folds.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
EXPECTED_VIEWS = dict(zip(SCENES, (56, 71, 59, 22)))
SEED = 20261101
L2 = .1
TOL = 1e-8
DOT_TOL = 2e-6
SCALE_EPS = 1e-12
SCALAR_FEATURES = (
    "s", "r", "b", "m", "rank", "deficit", "is_leaf", "level", "root_score",
    "log1p_n_points", "log1p_root_n_points", "n_points_ratio", "log1p_n_leaf", "leaf_gain",
    "id_top3", "id_mean", "cover_frac", "log1p_cover_views", "mwp_conf", "fallback_seg",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_number(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return float("nan")
    x = float(value)
    require(not np.isinf(x), "infinite scalar input")
    return x


def csv_key(row):
    return row["image"], row["prompt"], int(row["level"]), int(row["cand_id"])


def numeric_equal(a, b, tol=1e-8):
    return bool(np.allclose(a, b, atol=tol, rtol=0, equal_nan=True))


def raw_scalar(row, seg, detail, i):
    values = {"s": float(detail["s"][i]), "r": float(detail["r"][i]),
              "b": float(detail["b"][i]), "m": float(detail["m"][i])}
    for name in ("rank", "deficit", "is_leaf", "level", "root_score", "n_points_ratio", "leaf_gain"):
        values[name] = optional_number(row[name])
    for target, source in (("log1p_n_points", "n_points"), ("log1p_root_n_points", "root_n_points"),
                           ("log1p_n_leaf", "n_leaf")):
        x = optional_number(row[source])
        require(np.isnan(x) or x >= 0, f"negative {source}")
        values[target] = np.log1p(x)
    for name in ("id_top3", "id_mean", "cover_frac", "mwp_conf", "fallback_seg"):
        values[name] = optional_number(seg[name])
    x = optional_number(seg["cover_views"])
    require(np.isnan(x) or x >= 0, "negative cover_views")
    values["log1p_cover_views"] = np.log1p(x)
    return np.array([values[k] for k in SCALAR_FEATURES], dtype=np.float64)


def load_data(root, feature_root):
    workspace = root.parents[1]
    old_root = workspace / "scannet_seed1234_repro" / "lerf_selector_20261030"
    seg_root = workspace / "scannet_seed1234_repro" / "lerf_segidentity_20261030"
    score_root = root.parent / "score_readout_20261101" / "results"
    units = []
    validation = {"candidate_view_rows": 0, "feature_rows": 0, "max_feature_dot_error": 0.,
                  "max_score_join_error": 0., "max_label_join_error": 0.}
    for scene in SCENES:
        original = read_csv(old_root / f"{scene}_candidates.csv")
        segment = read_csv(seg_root / f"{scene}_segidentity.csv")
        old = {csv_key(r): r for r in original}
        seg = {csv_key(r): r for r in segment}
        require(len(old) == len(original) and len(seg) == len(segment), f"duplicate CSV key: {scene}")
        scores = read_csv(score_root / f"{scene}_score.csv")
        require(len(scores) == EXPECTED_VIEWS[scene], f"incomplete score CSV: {scene}")
        score_map = {(r["image"], r["prompt"]): r for r in scores}
        require(len(score_map) == len(scores), f"duplicate query-view: {scene}")
        require(all(r["scene"] == scene for r in scores + original + segment), "scene field mismatch")
        details = [json.loads(line) for line in (score_root / f"{scene}_details.jsonl").read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
        detail_map = {(d["image"], d["prompt"]): d for d in details}
        require(len(detail_map) == len(details) and set(detail_map) == set(score_map), f"detail keys mismatch: {scene}")
        with np.load(feature_root / f"{scene}.npz", allow_pickle=False) as feature:
            f = np.asarray(feature["F"], dtype=np.float64)
            t = np.asarray(feature["T"], dtype=np.float64)
            prompts = feature["prompts"].astype(str)
            levels = feature["levels"].astype(int)
            cids = feature["candidate_ids"].astype(int)
        require(f.shape == t.shape and f.shape == (len(prompts), 512), f"feature shape mismatch: {scene}")
        require(len(levels) == len(cids) == len(prompts), "feature metadata lengths differ")
        require(np.isfinite(f).all() and np.isfinite(t).all(), "nonfinite exported feature")
        feat = {(str(p), int(l), int(c)): (ff, tt) for p, l, c, ff, tt in zip(prompts, levels, cids, f, t)}
        require(len(feat) == len(f), f"duplicate exported feature key: {scene}")
        validation["feature_rows"] += len(f)
        image_queries = Counter(r["image"] for r in scores)
        image_count = len(image_queries)
        groups = defaultdict(list)
        for (image, prompt), d in detail_map.items():
            require(d["scene"] == scene, "detail scene mismatch")
            groups[prompt].append((image, d, score_map[(image, prompt)]))
        used_old, used_seg, used_feat = set(), set(), set()
        for prompt, views in sorted(groups.items()):
            image0, d0, score0 = views[0]
            keys = [tuple(map(int, k)) for k in d0["candidates"]]
            require(len(set(keys)) == len(keys), "duplicate A candidate")
            a_selection = [tuple(map(int, k)) for k in d0["selections"]["A"]]
            require(set(a_selection) <= set(keys), "A selection not contained in pool")
            g = int(float(score0["g_A"]))
            require(g == len(a_selection), "g_A does not match A selection")
            rows_f, rows_t = [], []
            for level, cid in keys:
                fk = (prompt, level, cid)
                require(fk in feat, f"missing exported feature {scene}/{fk}")
                ff, tt = feat[fk]
                rows_f.append(ff); rows_t.append(tt); used_feat.add(fk)
            F, T = np.asarray(rows_f), np.asarray(rows_t)
            require(np.allclose(T, T[0], atol=1e-8, rtol=0), "query embedding differs within prompt")
            scalars, truths, weights, view_info = [], [], [], []
            for image, d, score in sorted(views):
                require([tuple(map(int, k)) for k in d["candidates"]] == keys, "pool order differs across views")
                require([tuple(map(int, k)) for k in d["selections"]["A"]] == a_selection, "A selection differs across views")
                require(int(float(score["g_A"])) == g and float(score["control_set_equal"]) == 1,
                        "A/B positive control or fixed count failed")
                require(float(score["A_iou"]) == float(score["B_iou"]), "A/B IoU mismatch")
                require(abs(float(score["baseline_error"])) <= 1e-6, "baseline reproduction failed")
                X = []
                truth = np.array(d["candidate_ious"], dtype=float)
                require(truth.shape == (len(keys),) and np.isfinite(truth).all() and
                        np.all((truth >= 0) & (truth <= 1)), "invalid candidate IoUs")
                for i, (level, cid) in enumerate(keys):
                    key = image, prompt, level, cid
                    require(key in old and key in seg, f"candidate join missing {scene}/{key}")
                    a, b = old[key], seg[key]
                    used_old.add(key); used_seg.add(key)
                    se = max(abs(float(a["score"]) - float(d["s"][i])), abs(float(b["score"]) - float(d["s"][i])))
                    ie = max(abs(float(a["iou"]) - truth[i]), abs(float(b["iou"]) - truth[i]))
                    validation["max_score_join_error"] = max(validation["max_score_join_error"], se)
                    validation["max_label_join_error"] = max(validation["max_label_join_error"], ie)
                    require(se <= 1e-8 and ie <= 1e-8, "archived candidate score/IoU mismatch")
                    X.append(raw_scalar(a, b, d, i))
                scalars.append(np.asarray(X)); truths.append(truth)
                weights.append(1. / image_count / image_queries[image])
                view_info.append({"image": image, "evaluation_weight_within_scene": weights[-1]})
                validation["candidate_view_rows"] += len(keys)
            require(all(numeric_equal(scalars[0], x) for x in scalars), "non-area query scalar differs across views")
            dot = np.einsum("ij,ij->i", F, T)
            dot_error = float(np.max(np.abs(dot - np.asarray(d0["r"]))))
            validation["max_feature_dot_error"] = max(validation["max_feature_dot_error"], dot_error)
            require(dot_error <= DOT_TOL, f"exported q dot f differs from true raw score: {scene}/{prompt}")
            w = np.array(weights)
            unit = {"scene": scene, "prompt": prompt, "keys": keys, "A": a_selection, "g_A": g,
                "F": F, "T": T[0], "scalar": scalars[0], "mass": float(w.sum()),
                "y": np.average(np.stack(truths), axis=0, weights=w), "n_views": len(views), "views": view_info}
            units.append(unit)
        require(set(old) == used_old and set(seg) == used_seg, f"unused or extra candidate rows: {scene}")
        require(set(feat) == used_feat, f"unused or extra feature rows: {scene}")
        require(abs(sum(u["mass"] for u in units if u["scene"] == scene) - 1) < 1e-12,
                "scene evaluation masses do not sum to one")
    require(len(units) == 67 and validation["feature_rows"] == 366 and
            validation["candidate_view_rows"] == 1097, "frozen dataset count mismatch")
    validation.update(query_views=sum(u["n_views"] for u in units), scene_prompt_units=len(units),
                      scene_prompt_counts=dict(Counter(u["scene"] for u in units)))
    return units, validation


def make_pairs(units):
    """Only the passed split's labels can enter these training pair indices/weights."""
    start, eligible = 0, []
    scene_mass = defaultdict(float)
    for u in units:
        candidates = []
        for i in range(len(u["keys"])):
            for j in range(i + 1, len(u["keys"])):
                delta = float(u["y"][i] - u["y"][j])
                if abs(delta) > TOL:
                    candidates.append((start + (i if delta > 0 else j),
                                       start + (j if delta > 0 else i), abs(delta)))
        if candidates:
            eligible.append((u, candidates))
            scene_mass[u["scene"]] += u["mass"]
        start += len(u["keys"])
    scenes = sorted({u["scene"] for u in units})
    require(all(scene_mass[s] > 0 for s in scenes), "a split scene has no non-tied training pairs")
    high, low, weights = [], [], []
    for u, pairs in eligible:
        normalizer = sum(p[2] for p in pairs)
        group_weight = u["mass"] / scene_mass[u["scene"]] / len(scenes)
        for a, b, delta in pairs:
            high.append(a); low.append(b); weights.append(group_weight * delta / normalizer)
    weights = np.asarray(weights, dtype=np.float64)
    require(abs(weights.sum() - 1) < 1e-12, "pair loss weights are not normalized")
    return np.asarray(high, int), np.asarray(low, int), weights, {
        "candidate_rows": start, "scene_prompt_units": len(units), "eligible_scene_prompt_units": len(eligible),
        "non_tied_pairs": len(high), "weight_sum": float(weights.sum()), "eligible_mass_by_scene": dict(scene_mass)}


def query_name(prompt):
    return " ".join(prompt.casefold().split())


def derange(units, seed):
    """One split-local donor per scene/prompt; no labels or candidate IoUs used."""
    rng = np.random.default_rng(seed)
    cost = rng.random((len(units), len(units)))
    for i, u in enumerate(units):
        for j, donor in enumerate(units):
            if i == j or query_name(u["prompt"]) == query_name(donor["prompt"]):
                cost[i, j] = np.inf
    rows, columns = linear_sum_assignment(cost)
    require(len(rows) == len(units) and np.isfinite(cost[rows, columns]).all(), "no legal text derangement")
    mapping = {int(i): int(j) for i, j in zip(rows, columns)}
    collisions = sum(np.allclose(units[i]["T"], units[j]["T"], atol=1e-8, rtol=0) for i, j in mapping.items())
    records = [{"scene": units[i]["scene"], "prompt": units[i]["prompt"],
        "donor_scene": units[j]["scene"], "donor_prompt": units[j]["prompt"]} for i, j in sorted(mapping.items())]
    return mapping, {"seed": seed, "mapping": records, "fixed_points": 0, "same_normalized_text": 0,
                     "same_embedding_collisions": int(collisions), "n_units": len(units)}


def blocks(units, head, permutation=None):
    scalar = np.concatenate([u["scalar"] for u in units], axis=0)
    if head == "S":
        return [scalar]
    products = []
    for i, u in enumerate(units):
        text = u["T"] if head == "J" else units[permutation[i]]["T"]
        products.append(u["F"] * text[None, :])
    return [scalar, np.concatenate(products, axis=0)]


def fit_preprocessor(raw_blocks, high, low, weights):
    params, outputs = [], []
    for raw in raw_blocks:
        imputer = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                            for col in raw.T])
        x = np.where(np.isfinite(raw), raw, imputer[None, :])
        delta = x[high] - x[low]
        rms = np.sqrt(np.sum(weights[:, None] * delta * delta, axis=0))
        active = rms > SCALE_EPS
        divisor = np.where(active, rms, 1.)
        dim = int(active.sum())
        norm = float(np.sqrt(dim)) if dim else 1.
        transformed = x / divisor[None, :] / norm
        transformed[:, ~active] = 0.
        params.append({"imputer": imputer, "scale": divisor, "active": active,
                       "block_divisor": norm, "active_dimensions": dim})
        outputs.append(transformed)
    return np.concatenate(outputs, axis=1), params


def apply_preprocessor(raw_blocks, params):
    outputs = []
    for raw, p in zip(raw_blocks, params):
        x = np.where(np.isfinite(raw), raw, p["imputer"][None, :])
        x = x / p["scale"][None, :] / p["block_divisor"]
        x[:, ~p["active"]] = 0.
        outputs.append(x)
    return np.concatenate(outputs, axis=1)


def objective(w, differences, weights):
    margins = differences @ w
    loss = float(weights @ np.logaddexp(0., -margins) + L2 / 2 * (w @ w))
    gradient = differences.T @ (-weights * expit(-margins)) + L2 * w
    return loss, gradient


def gradient_test():
    rng = np.random.default_rng(SEED)
    D = rng.normal(size=(7, 5))
    weights = rng.random(7); weights /= weights.sum()
    w = rng.normal(size=5)
    _, analytical = objective(w, D, weights)
    numerical = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = 1e-6
        numerical[i] = (objective(w + e, D, weights)[0] - objective(w - e, D, weights)[0]) / 2e-6
    error = float(np.max(np.abs(numerical - analytical)))
    require(error < 1e-7, f"gradient check failed: {error}")
    return {"max_absolute_gradient_error": error, "passed": True, "real_data_fitted": False}


def split_scores(units, values):
    result, start = [], 0
    for u in units:
        n = len(u["keys"])
        result.append(np.asarray(values[start:start + n]))
        start += n
    require(start == len(values), "score length does not match candidate rows")
    return result


def ranking_diagnostic(units, score_lists):
    """Post-fit labels only. One chosen node is shared over all views of each query."""
    scenes = sorted({u["scene"] for u in units})
    oracle = achieved = 0.
    for u, scores in zip(units, score_lists):
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        weight = u["mass"] / len(scenes)
        oracle += weight * float(np.max(u["y"]))
        achieved += weight * float(u["y"][order[0]])
    high, low, weights, pair_info = make_pairs(units)
    flat = np.concatenate(score_lists)
    delta = flat[high] - flat[low]
    return {"single_candidate_iou": achieved, "query_consistent_single_mask_oracle": oracle,
        "regret": oracle - achieved,
        "weighted_pair_accuracy_ties_half": float(weights @ ((delta > TOL) + .5 * (np.abs(delta) <= TOL))),
        "pair_counts": pair_info,
        "diagnostic_only": True, "not_a_multimask_upper_bound": True}


def fit_head(train, test, head, train_shuffle, test_shuffle):
    high, low, weights, pair_info = make_pairs(train)
    raw = blocks(train, head, train_shuffle)
    x, params = fit_preprocessor(raw, high, low, weights)
    D = x[high] - x[low]
    fit = minimize(objective, np.zeros(x.shape[1]), args=(D, weights), method="L-BFGS-B", jac=True,
        tol=1e-9, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9, "maxls": 50})
    require(bool(fit.success) and np.isfinite(fit.x).all(), f"optimizer did not converge for {head}: {fit.message}")
    test_x = apply_preprocessor(blocks(test, head, test_shuffle), params)
    train_scores, test_scores = split_scores(train, x @ fit.x), split_scores(test, test_x @ fit.x)
    optimizer = {"method": "L-BFGS-B", "success": bool(fit.success), "status": int(fit.status),
        "message": str(fit.message), "iterations": int(fit.nit), "function_evaluations": int(fit.nfev),
        "objective": float(fit.fun), "gradient_infinity_norm": float(np.max(np.abs(fit.jac))),
        "coefficient_l2_norm": float(np.linalg.norm(fit.x)),
        "active_dimensions_per_block": [p["active_dimensions"] for p in params]}
    return {"params": params, "coef": fit.x, "optimizer": optimizer, "pair_counts": pair_info,
        "train_scores": train_scores, "test_scores": test_scores,
        "train_diagnostic": ranking_diagnostic(train, train_scores),
        "test_diagnostic": ranking_diagnostic(test, test_scores)}


def save_model(path, model, heldout, head, train_scenes, train_mapping, test_mapping):
    payload = {"coef": model["coef"], "scalar_feature_names": np.asarray(SCALAR_FEATURES),
        "head": np.asarray(head), "heldout_scene": np.asarray(heldout), "train_scenes": np.asarray(train_scenes),
        "l2": np.asarray(L2), "seed": np.asarray(SEED), "n_blocks": np.asarray(len(model["params"])),
        "train_shuffle_json": np.asarray(json.dumps(train_mapping, ensure_ascii=False)),
        "test_shuffle_json": np.asarray(json.dumps(test_mapping, ensure_ascii=False))}
    for i, p in enumerate(model["params"]):
        for name in ("imputer", "scale", "active", "block_divisor", "active_dimensions"):
            payload[f"block_{i}_{name}"] = np.asarray(p[name])
    np.savez_compressed(path, **payload)


def run(root, feature_root, sanity):
    toy = gradient_test()
    units, validation = load_data(root, feature_root)
    fold_setup = []
    for index, heldout in enumerate(SCENES):
        train = [u for u in units if u["scene"] != heldout]
        test = [u for u in units if u["scene"] == heldout]
        require(set(u["scene"] for u in train) == set(SCENES) - {heldout}, "fold isolation failed")
        require(set(u["scene"] for u in test) == {heldout}, "heldout split failed")
        train_map, train_meta = derange(train, SEED + index * 100 + 1)
        test_map, test_meta = derange(test, SEED + index * 100 + 2)
        pairs = make_pairs(train)[3]
        fold_setup.append((heldout, train, test, train_map, test_map, train_meta, test_meta, pairs))
    if sanity:
        print(json.dumps({"status": "SANITY_PASS", "validation": validation, "toy": toy,
            "folds_checked": len(fold_setup), "real_heads_fitted": 0, "outputs_written": False}))
        return
    folds, predictions, persisted = [], [], []
    all_test_units, all_scores = [], {h: [] for h in "SJQ"}
    for heldout, train, test, train_map, test_map, train_meta, test_meta, pair_info in fold_setup:
        models = {h: fit_head(train, test, h, train_map, test_map) for h in "SJQ"}
        # True scalar values/scaler are identical in all three arms, including Q.
        for h in "JQ":
            for name in ("imputer", "scale", "active", "block_divisor"):
                require(np.array_equal(models["S"]["params"][0][name], models[h]["params"][0][name]),
                        "scalar preprocessing differs across heads")
        train_scenes = [s for s in SCENES if s != heldout]
        folds.append({"heldout_scene": heldout, "train_scenes": train_scenes,
            "optimizer": {h: models[h]["optimizer"] for h in "SJQ"}, "pair_counts": pair_info,
            "train_regret": {h: models[h]["train_diagnostic"] for h in "SJQ"},
            "oof_ranking": {h: models[h]["test_diagnostic"] for h in "SJQ"},
            "shuffle": {"train": train_meta, "test": test_meta},
            "train_scene_prompt_units": len(train), "heldout_scene_prompt_units": len(test)})
        for i, u in enumerate(test):
            selections = {"A": [list(k) for k in u["A"]]}
            scores = {}
            for h in "SJQ":
                values = models[h]["test_scores"][i]
                order = sorted(range(len(values)), key=lambda j: (-float(values[j]), j))
                selections[h] = [list(u["keys"][j]) for j in order[:u["g_A"]]]
                scores[h] = values.tolist()
                all_scores[h].append(values)
            predictions.append({"scene": u["scene"], "prompt": u["prompt"], "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "candidates": [list(k) for k in u["keys"]], "g_A": u["g_A"],
                "selections": selections, "scores": scores})
        all_test_units.extend(test)
        for h in "SJQ":
            persisted.append((heldout, h, models[h], train_scenes, train_meta, test_meta))
    require(len(predictions) == 67 and len({(p["scene"], p["prompt"]) for p in predictions}) == 67,
            "OOF predictions do not cover each query exactly once")
    summary = {"status": "COMPLETE", "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation, "toy_gradient_check": toy, "folds": folds,
        "scalar_feature_names": list(SCALAR_FEATURES), "heads": {"S": "scalar-only", "J": "scalar+q*F", "Q": "scalar+deranged_q*F"},
        "hyperparameters": {"l2": L2, "optimizer": "L-BFGS-B", "maxiter": 1000, "tol": 1e-9,
            "ftol": 1e-9, "gtol": 1e-9, "maxls": 50, "initialization": "zeros", "seed": SEED,
            "pair_tie_tolerance": TOL, "scale_epsilon": SCALE_EPS, "no_sweep": True},
        "label_weight_definition": {"view": "a_v=1/(scene image count * queries in image)",
            "candidate_label": "y_qc=sum_v a_v*IoU_qcv/sum_v a_v", "query_mass": "sum_v a_v",
            "pair": "all abs(delta_y)>1e-8; within-query weights proportional to abs(delta_y)",
            "loss": "query mass renormalized over eligible queries within scene, then equal scene mean; total pair weight=1",
            "preprocessing": "train-only candidate-row median imputation; train-weighted pair-difference RMS per column; inactive columns zero; each block divided by sqrt(active dimensions)",
            "CLIP_vectors_unchanged": True, "area_ratio_excluded": True,
            "GT_and_family_excluded_from_inputs": True, "heldout_GT_not_used_in_training_or_selection": True},
        "out_of_fold": {h: ranking_diagnostic(all_test_units, all_scores[h]) for h in "SJQ"},
        "scope": ["67 scene/prompt decisions, 208 repeated query-view evaluations; not 1097 independent samples.",
            "Ranking diagnostics choose one mask consistently across all query views; they are not bounds on multi-mask predictions.",
            "All deployed selections use the original A pool and original g_A without heldout labels.",
            "Fixed original A order breaks exact learned-score ties; no threshold, gate, or head is chosen post hoc.",
            "Same four scenes have been explored previously; LOSO is not a fresh independent confirmation."]}
    # Write only after every real fold succeeds. Prediction records contain no GT labels.
    json.dumps(summary, allow_nan=False)
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for heldout, h, model, train_scenes, train_meta, test_meta in persisted:
        save_model(model_dir / f"{heldout}_{h}.npz", model, heldout, h, train_scenes, train_meta, test_meta)
    (root / "predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for p in predictions), encoding="utf-8")
    (root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "folds": len(folds), "predictions": len(predictions),
        "models": len(persisted), "validation": validation,
        "oof_single_mask_regret_diagnostic": {h: summary["out_of_fold"][h]["regret"] for h in "SJQ"}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--features", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sanity", action="store_true")
    mode.add_argument("--full", action="store_true", help="fit all four frozen LOSO folds (default)")
    args = parser.parse_args()
    try:
        run(args.root.resolve(), (args.features or args.root / "features").resolve(), args.sanity)
    except FileNotFoundError as exc:
        print(json.dumps({"status": "WAITING_FOR_INPUTS", "missing": str(exc.filename), "outputs_written": False}))
        raise SystemExit(2)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc), "completed_results_claimed": False}))
        raise


if __name__ == "__main__":
    main()
