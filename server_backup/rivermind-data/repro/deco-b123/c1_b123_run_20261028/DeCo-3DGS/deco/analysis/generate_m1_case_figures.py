#!/usr/bin/env python3
"""Render six selected M1 cases from existing Gaussian/camera data.

No feature extraction, NAG construction, bank generation, or evaluator is run.
The support panels are the same Gaussian renderer semantics path used by the
existing root-region support render; each mode uses its real visible Gaussian
CSR union from the frozen observation file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def poly_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1)
    return m


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_final_labels(path: Path):
    """Load the frozen level-3 labels without constructing or changing NAG."""
    payload = torch.load(path, map_location="cpu")
    labels = np.asarray(payload["nag"][-1].detach().cpu().numpy(), dtype=np.int64)
    return labels, sha256(path)


def camera_rgb(cam, source: Path, image: str):
    # Prefer the camera tensor used by the existing evaluator.  It is the
    # original RGB from the source dataset, not a fabricated mask.
    img = getattr(cam, "original_image", None)
    if img is not None:
        arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
        return np.clip(arr, 0, 1)
    for ext in (".jpg", ".png", ".jpeg"):
        p = source / "images" / (image + ext)
        if p.exists():
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return None


def gt_mask(gt_root: Path, scene: str, image: str, query: str, shape):
    p = gt_root / scene / (image + ".json")
    if not p.exists():
        return None
    anno = json.loads(p.read_text())
    out = np.zeros(shape, np.uint8)
    found = False
    for obj in anno.get("objects", []):
        if obj.get("category") == query:
            out = np.maximum(out, poly_mask(shape, obj["segmentation"]))
            found = True
    return out.astype(bool) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--nag", required=True, type=Path)
    ap.add_argument("--gt-root", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--observations", required=True, type=Path)
    ap.add_argument("--members", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    pargs, _ = parser.parse_known_args(["-s", str(args.source), "-m", str(args.model)])
    ds = lp.extract(pargs)
    pipe = pp.extract(pargs)
    gauss = GaussianModel(ds.sh_degree, 20)
    scene = Scene(ds, gauss, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    n_gauss = int(gauss.get_xyz.shape[0])

    labels, nag_sha = load_final_labels(args.nag)
    if len(labels) != n_gauss:
        raise RuntimeError(
            f"Gaussian/NAG mismatch: renderer has {n_gauss}, "
            f"frozen NAG has {len(labels)}"
        )

    with np.load(args.observations, allow_pickle=False) as z:
        ptr = np.asarray(z["visible_gaussian_offsets"], dtype=np.int64)
        gids = np.asarray(z["visible_gaussian_ids"], dtype=np.int64)
        obs_regions = np.asarray(z["region_ids"], dtype=np.int64)
    if len(ptr) == 0 or ptr[-1] != len(gids):
        raise RuntimeError("invalid visible Gaussian CSR offsets")
    members = json.loads(args.members.read_text())["memberships"]
    mmap = {(int(x["region_id"]), int(x["slot"])): x for x in members}
    cases = pd.read_csv(args.cases)
    cases = cases[cases.scene == args.scene].copy()
    # Only the requested three categories, one case per category.  The CSV is
    # constructed from the existing FULL_EVIDENCE_TABLE, with no new scoring.
    order = ["target_purity_gt50", "visible_background", "target_low_purity"]
    cases = cases[cases.case_group.isin(order)].sort_values(["case_group", "case_index"] if "case_index" in cases else ["case_group"])
    if len(cases) > 3:
        cases = cases.groupby("case_group", sort=False).head(1)

    manifest = []
    with torch.no_grad():
        for _, row in cases.iterrows():
            image = str(row.image)
            key = f"{args.scene}_{int(row.region_id)}_{image}_{str(row.query).replace(' ', '_')}"
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
            cam = cams.get(image)
            item = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in row.to_dict().items()}
            item.update({"figure": safe + ".png", "rgb_available": False, "gt_available": False,
                         "root_mask_available": False, "mode_support_available": [],
                         "nag_sha256": nag_sha, "gaussian_count": n_gauss,
                         "renderer_projection_normalization": "none"})
            if cam is None:
                item["error"] = "camera not found"
                manifest.append(item)
                continue
            h, w = int(cam.image_height), int(cam.image_width)
            rgb = camera_rgb(cam, args.source, image)
            # The panel must use the original camera RGB.  Do not issue an
            # extra RGB render: the existing support renderer is the only
            # projection path we need, and avoiding an unrelated RGB pass
            # keeps this audit independent of the evaluator output state.
            if rgb is None:
                item["rgb_available"] = False
                rgb = np.zeros((h, w, 3), dtype=np.float32)
            else:
                item["rgb_available"] = True

            gt = gt_mask(args.gt_root, args.scene, image, str(row.query), (h, w))
            item["gt_available"] = gt is not None
            region_id = int(row.region_id)
            root_ids = np.flatnonzero(labels == region_id).astype(np.int64)
            if len(root_ids) == 0:
                raise RuntimeError(f"frozen NAG has no Gaussian for region {region_id}")
            if root_ids.min() < 0 or root_ids.max() >= n_gauss:
                raise RuntimeError(f"root Gaussian IDs out of range for region {region_id}")

            def render_support(ids):
                sem = torch.zeros((n_gauss, 20), dtype=torch.float32, device="cuda")
                if len(ids):
                    sem[torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(device="cuda"), 0] = 1.0
                gauss._semantics = sem
                rendered = render(cam, gauss, pipe, bg)["semantics"][0]
                arr = rendered.detach().cpu().float().numpy()
                if not np.isfinite(arr).all():
                    raise RuntimeError("non-finite semantic projection")
                if float(arr.min()) < -1e-6:
                    raise RuntimeError("semantic renderer is not non-negative")
                return arr

            # Root and all mode panels use the same camera, Gaussian model,
            # point order, renderer, and raw (unnormalized) float output.
            root_float = render_support(root_ids)
            root_mask = root_float > 0.5
            item.update({"root_mask_available": True, "root_gaussian_count": int(len(root_ids)),
                         "root_projection_max": float(root_float.max()),
                         "root_projection_positive_px": int((root_float > 1e-6).sum())})

            mode_maps = []
            intersection_maps = []
            mode_stats = []
            for mode in range(3):
                mem = mmap.get((int(row.region_id), mode))
                if mem is None:
                    mode_maps.append(None)
                    intersection_maps.append(None)
                    continue
                obs_ids = [int(x) for x in mem.get("observation_indices", [])]
                if not obs_ids:
                    mode_maps.append(None)
                    intersection_maps.append(None)
                    continue
                if min(obs_ids) < 0 or max(obs_ids) >= len(obs_regions):
                    raise RuntimeError(f"observation index out of range for region={region_id}, mode={mode}")
                obs_region_values = np.unique(obs_regions[np.asarray(obs_ids, dtype=np.int64)])
                if not np.all(obs_region_values == region_id):
                    raise RuntimeError(
                        f"observation region mismatch: region={region_id}, mode={mode}, "
                        f"obs_regions={obs_region_values.tolist()}"
                    )
                vis = np.unique(np.concatenate([gids[ptr[i] : ptr[i + 1]] for i in obs_ids])).astype(np.int64)
                if len(vis) and (vis.min() < 0 or vis.max() >= n_gauss):
                    raise RuntimeError(f"visible Gaussian IDs out of range for region={region_id}, mode={mode}")
                outside = np.setdiff1d(vis, root_ids)
                if len(outside):
                    raise RuntimeError(
                        f"Invalid spatial support: region={region_id}, mode={mode}, outside={len(outside)}"
                    )
                intersection_ids = np.intersect1d(vis, root_ids)
                smap = render_support(vis)
                intersection_float = render_support(intersection_ids)
                violation = float(np.max(smap - root_float)) if smap.size else 0.0
                if violation > 1e-5:
                    raise RuntimeError(
                        f"projection containment failed: region={region_id}, mode={mode}, "
                        f"max(mode-root)={violation:.9g}"
                    )
                mode_maps.append(smap)
                intersection_maps.append(intersection_float)
                maxv = float(np.nanmax(smap)) if smap.size else 0.0
                support05 = smap > 0.5
                gt_inter = int((support05 & gt).sum()) if gt is not None else None
                px05 = int(support05.sum())
                prec05 = float(gt_inter / px05) if (gt_inter is not None and px05) else None
                mode_stats.append({"mode_id": mode, "visible_gaussian_count": int(len(vis)),
                                   "outside_gaussian_count": int(len(outside)),
                                   "observation_count": len(obs_ids),
                                   "observation_region_ids": obs_region_values.tolist(),
                                   "projection_max": maxv,
                                   "projection_positive_px": int((smap > 1e-6).sum()),
                                   "projection_positive_px_at_0.5": px05,
                                   "projection_gt_intersection_at_0.5": gt_inter,
                                   "projection_precision_at_0.5": prec05,
                                   "containment_max_violation": violation,
                                   "mode_float_le_root": bool(violation <= 1e-5),
                                   "intersection_projection_max": float(intersection_float.max()) if intersection_float.size else 0.0,
                                   "intersection_gaussian_count": int(len(intersection_ids))})
                if mode == int(row.mode_id):
                    item["selected_mode_outside_gaussian_count"] = int(len(outside))
                    item["selected_mode_containment_max_violation"] = violation
                    item["selected_mode_float_le_root"] = bool(violation <= 1e-5)
                    item["selected_mode_gaussian_count"] = int(len(vis))
                    item["selected_mode_observation_region_ids"] = obs_region_values.tolist()
                item["mode_support_available"].append(mode_stats[-1])

            fig, ax = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
            ax = ax.ravel()
            ax[0].imshow(np.clip(rgb, 0, 1)); ax[0].set_title("Original RGB")
            if gt is not None:
                ax[1].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"GT: {row.query}")
            else:
                ax[1].text(.5, .5, "GT unavailable", ha="center", va="center"); ax[1].set_title("GT")
            if root_mask is not None:
                ax[2].imshow(root_mask, cmap="magma", vmin=0, vmax=1); ax[2].set_title(f"Root region {int(row.region_id)}")
            else:
                ax[2].text(.5, .5, "root mask unavailable", ha="center", va="center"); ax[2].set_title("Root mask")
            for mode, smap in enumerate(mode_maps):
                a = ax[3 + mode]
                if smap is None:
                    a.text(.5, .5, f"mode {mode}\nunavailable", ha="center", va="center")
                else:
                    a.imshow(smap, cmap="viridis", vmin=0, vmax=1)
                a.set_title(f"Mode {mode} support (raw float)")
            selected = int(row.mode_id)
            a = ax[6]
            selected_intersection = intersection_maps[selected] if selected < len(intersection_maps) else None
            if selected_intersection is None:
                a.text(.5, .5, "selected intersection unavailable", ha="center", va="center")
            else:
                a.imshow(selected_intersection, cmap="viridis", vmin=0, vmax=1)
            a.set_title(f"Mode {selected} ∩ root (raw float)")
            a = ax[7]
            if selected < len(mode_maps) and mode_maps[selected] is not None:
                a.imshow(np.maximum(root_float - mode_maps[selected], 0), cmap="magma", vmin=0, vmax=1)
            else:
                a.text(.5, .5, "difference unavailable", ha="center", va="center")
            a.set_title("root − selected mode (raw float)")
            ax[8].axis("off")
            ax[8].text(0.02, 0.95,
                       f"frozen NAG: {nag_sha[:12]}…\n"
                       f"root IDs: {len(root_ids)}\n"
                       f"selected mode IDs: {item.get('selected_mode_gaussian_count', 'n/a')}\n"
                       f"outside IDs: {item.get('selected_mode_outside_gaussian_count', 'n/a')}\n"
                       f"max(mode−root): {item.get('selected_mode_containment_max_violation', float('nan')):.3g}",
                       va="top", family="monospace")
            for a in ax:
                a.axis("off")
            fig.suptitle(f"{args.scene} | {image} | query={row.query} | region={int(row.region_id)} | M1 mode={int(row.mode_id)}\n"
                         f"class={row.case_group} | gain={float(row.recover_gain):.6f} | precision={float(row.precision):.4f}")
            fig.savefig(args.out / (safe + ".png"), dpi=140)
            plt.close(fig)
            np.savez_compressed(args.out / (safe + ".npz"),
                                gt=(gt.astype(np.uint8) if gt is not None else np.zeros((h, w), np.uint8)),
                                root_float=root_float.astype(np.float32),
                                root=(root_mask.astype(np.uint8)),
                                **{f"intersection_{i}": (m.astype(np.float32) if m is not None else np.zeros((h, w), np.float32)) for i, m in enumerate(intersection_maps)},
                                **{f"mode_{i}": (m.astype(np.float32) if m is not None else np.zeros((h, w), np.float32)) for i, m in enumerate(mode_maps)})
            (args.out / (safe + ".json")).write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf8")
            manifest.append(item)
    pd.DataFrame(manifest).to_json(args.out / "CASE_FIGURES_MANIFEST.json", orient="records", indent=2, force_ascii=False)
    print("rendered", len(manifest), "cases to", args.out)


if __name__ == "__main__":
    main()
