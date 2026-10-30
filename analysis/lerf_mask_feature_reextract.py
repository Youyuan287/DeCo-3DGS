"""Minimal validation: re-extract mask-level CLIP features from the actual images.

For a few queries per scene, compute two mask-conditioned CLIP features
  maskfull : full image with non-mask pixels set to gray, standard CLIP preprocess
  bbox     : crop to the mask's bounding box, background gray, resize 224
and compare the rank of the GT-best mask against the dataset-provided features
(rank_dataset read from the maskdiag CSV). GT is used only to identify the best mask.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser
from PIL import Image
import open_clip
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "gt_root", "diag_csv", "preds_csv", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--per-family", type=int, default=1)
args = ap.parse_args()

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
model = model.cuda().eval()
tokenizer = open_clip.get_tokenizer("ViT-B-16")
to224 = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                   T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                               std=(0.26862954, 0.26130258, 0.27577711))])

diag = {}
with open(args.diag_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        diag[(r["image"], r["prompt"])] = r

cats = {}
with open(args.preds_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["scene"] == args.scene:
            cats[(r["image"], r["prompt"])] = r["cat"]

sel = {}
for (im, pr), c in cats.items():
    if c not in sel or len(sel[c]) < args.per_family:
        sel.setdefault(c, []).append((im, pr))
wanted = {(im, pr) for lst in sel.values() for im, pr in lst}
print("selected queries:", len(wanted), flush=True)

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
cams = {c.image_name: c for c in scene.getTrainCameras()}
img_dir = os.path.join(args.source, "images")
gt_dir = os.path.join(args.gt_root, args.scene)
rows = []


def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)


with torch.no_grad():
    for name in sorted(set(im for im, _ in wanted)):
        img_path = os.path.join(img_dir, name + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, name + ".png")
        img = np.array(Image.open(img_path).convert("RGB"))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        if img.shape[0] != h or img.shape[1] != w:
            img = np.array(Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR))
        seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        labels = np.unique(seg[fg])
        mask_arrs = {int(L): ((seg == L) & fg) for L in labels}
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        for im_key, prompt in sorted(wanted):
            if im_key != name:
                continue
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            ious = {}
            for L, m in mask_arrs.items():
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            best_L = max(ious, key=ious.get)
            toks = tokenizer([prompt]).cuda()
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats = {"maskfull": {}, "bbox": {}}
            for L, m in mask_arrs.items():
                mf = (img * m[..., None] + 128 * (~m)[..., None]).astype(np.uint8)
                feats["maskfull"][L] = preprocess(Image.fromarray(mf))
                ys, xs = np.where(m)
                if len(ys) == 0:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = img[y0:y1, x0:x1]
                cm = m[y0:y1, x0:x1]
                cf = (crop * cm[..., None] + 128 * (~cm)[..., None]).astype(np.uint8)
                feats["bbox"][L] = to224(Image.fromarray(cf))
            out = {}
            for tag, fd in feats.items():
                if not fd:
                    out["rank_" + tag] = -1
                    continue
                Ls = sorted(fd)
                batch = torch.stack([fd[L] for L in Ls]).cuda()
                emb = model.encode_image(batch)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ tf.T).squeeze(-1).cpu().numpy()
                order = [Ls[i] for i in np.argsort(-sims)]
                out["rank_" + tag] = order.index(best_L) + 1 if best_L in order else -1
            d = diag.get((name, prompt), {})
            rows.append(dict(scene=args.scene, image=name, prompt=prompt, cat=cats.get((name, prompt), "?"),
                             n_masks=len(mask_arrs), best_label=int(best_L),
                             iou_best=ious[best_L],
                             rank_dataset=int(d.get("rank_raw", -1)) if d else -1,
                             rank_maskfull=out["rank_maskfull"], rank_bbox=out["rank_bbox"]))
        print("IMG", name, flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_reextract.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))