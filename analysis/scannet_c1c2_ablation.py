#!/usr/bin/env python3
"""C1 x C2 ablation on ScanNet-10, with the shipped scorer's own bank code.

C1 = deco_ops.center_np (image side: subtract the mean over superpoints)
C2 = text side: bare class name vs templates + hand-written lexicon, tau-LSE

The shipped score_scannet.py always applies C1, so this script extracts its module prefix
(constants and bank construction) by exec'ing everything before `def main():`, then toggles C1.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
_DECO = os.path.dirname(CODE)
if _DECO not in sys.path:
    sys.path.insert(0, _DECO)

from eval_scannet_3d import NYU40, SUBSETS, metrics, text_features  # noqa: E402
from deco_ops import center_np  # noqa: E402

SCORER = os.path.join(CODE, "score_scannet.py")


def load_constants():
    src = open(SCORER).read().split("def main():")[0]
    ns = {"__name__": "score_prefix", "__file__": SCORER}
    exec(compile(src, SCORER, "exec"), ns)
    return ns["NAMES"], ns["PROMPTS"], ns["TAU"], ns["LEX"], ns["classes"]


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    NAMES, PROMPTS, TAU, LEX, classes = load_constants()
    print("classes", len(classes), "names", len(NAMES), file=sys.stderr)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb(words):
        with torch.no_grad():
            f = model.encode_text(tok(list(words)))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    banks = {}
    banks["one_word"] = l2n(emb(NAMES) - emb(NAMES).mean(0, keepdims=True))
    V = max(len(LEX[n]) for n in NAMES)
    rows = [emb(LEX[n] + [LEX[n][0]] * (V - len(LEX[n]))) for n in NAMES]
    lex = np.transpose(np.stack(rows), (1, 0, 2))
    lex = l2n(lex - lex.mean(1, keepdims=True))
    generic = ("{}", "a photo of a {}", "the {}", "a {} in a room", "a close-up of a {}",
               "a {} seen from above", "part of a room: {}", "a large {}", "a small {}")
    gen = np.stack([emb([p.format(n) for n in NAMES]) for p in generic])
    gen = l2n(gen - gen.mean(1, keepdims=True))
    tpl = np.stack([emb([p.format(n) for n in NAMES]) for p in PROMPTS])
    tpl_c = l2n(tpl - tpl.mean(1, keepdims=True))
    both = np.concatenate([tpl_c, lex], 0)

    def lse(bank):
        sc = np.stack([Xc @ bank[i].T for i in range(len(bank))])
        m = sc.max(0)
        return TAU * np.log(np.exp((sc - m) / TAU).mean(0)) + m

    cfg_names = [("none", 0, "one_word"), ("none", 0, "both_lse"),
                 ("image", 0, "one_word"), ("image", 0, "both_lse"),
                 ("image", 1, "one_word"), ("image", 1, "both_lse")]
    scorer = {}
    per_scene = {}
    for fn in sorted(os.listdir(args.cache)):
        if not fn.endswith(".npz"):
            continue
        scene = fn[:-4]
        z = np.load(os.path.join(args.cache, fn))
        gt = z["gt"].astype(np.int64)
        keep = z["keep"] if "keep" in z.files else np.ones(gt.shape[0], bool)
        nag = torch.load(os.path.join(args.models, scene, "sai_nag_3.0.pt"), map_location="cpu")
        lv = [np.asarray(x).ravel().astype(np.int64) for x in nag["nag"]]
        li = len(nag["nag_feat"]) - 1
        X = l2n(np.asarray(nag["nag_feat"][li].float().cpu().numpy()))
        per_scene[scene] = {}
        for center, renorm, bank_name in cfg_names:
            Xc = X if center == "none" else center_np([X], renorm=bool(renorm))[0]
            if bank_name == "one_word":
                sc = Xc @ banks["one_word"].T
            else:
                sc = lse(both)
            pred = np.array([classes[i] for i in sc.argmax(1)])[lv[li + 1]]
            mm = metrics(gt[keep], pred[keep], classes)
            key = "C1=%s_C2=%s_r%d" % ("off" if center == "none" else "on",
                                       "off" if bank_name == "one_word" else "on", renorm)
            per_scene[scene][key] = 100 * mm["mIoU"]
            acc = scorer.setdefault(key, {"iou": [], "acc": [], "inter": np.zeros(19),
                                          "union": np.zeros(19), "tot": np.zeros(19)})
            acc["iou"].append(100 * mm["mIoU"])
            acc["acc"].append(100 * mm["mAcc"])
            for i, c in enumerate(classes):
                pc = mm["per_class"].get(NYU40[c])
                if pc is None:
                    continue
                acc["inter"][i] += pc["inter"]
                acc["union"][i] += pc["union"]
                acc["tot"][i] += pc["gt"]

    out = {"per_scene": per_scene, "summary": {}}
    for k, v in scorer.items():
        present = v["tot"] > 0
        ious = v["inter"][present] / np.maximum(v["union"][present], 1)
        out["summary"][k] = dict(pooled_miou=100 * float(np.mean(ious)),
                                scene_mean=float(np.mean(v["iou"])),
                                scene_mean_acc=float(np.mean(v["acc"])),
                                classes_present=int(present.sum()))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "c1c2_ablation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("%-24s %10s %10s %10s" % ("config", "pooled", "scene-mean", "mAcc"))
    for k in sorted(out["summary"]):
        s = out["summary"][k]
        print("%-24s %10.2f %10.2f %10.2f" % (k, s["pooled_miou"], s["scene_mean"],
                                              s["scene_mean_acc"]))


if __name__ == "__main__":
    main()