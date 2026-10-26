"""The vocabulary is a distribution over words, not a point.

A class name in these benchmarks is one string, and the pipeline embeds one string (or the mean of four
templates). But a class is a region of lexical space: "sofa" is couch/settee/loveseat, "counter" is
worktop/bar/sideboard. Treating it as a point makes the decision depend on which word happened to be
written down; treating it as a distribution and marginalising makes the decision depend on the class.

This compares, on the same features and the same protocol:
    one word            the bare class name
    templates           the four templates averaged into one vector (pipeline behaviour)
    templates lse       the four templates marginalised
    lexicon lse         a hand-written lexicon per class, marginalised
    both lse            everything marginalised

All at the coarsest level with the scene background removed, so only the lexical side varies.

Usage: python cli_syn.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("RIVERMIND_ROOT", ".")
CACHE = ROOT + "/clip_cache"
sys.path.insert(0, ROOT + "/scripts")
from eval_scannet_3d import NYU40, SUBSETS, metrics  # noqa: E402

classes = SUBSETS[19]
NAMES = [NYU40[c] for c in classes]
PROMPTS = ("{}", "a photo of a {}", "the {}", "a {} in a room")
TAU = 0.02

# a small lexical distribution per class, written from NYU40 semantics
LEX = {
    "wall": ["wall", "walls", "wall surface", "room wall", "partition wall"],
    "floor": ["floor", "floors", "flooring", "ground", "room floor"],
    "cabinet": ["cabinet", "cabinets", "cupboard", "shelf unit", "wardrobe"],
    "bed": ["bed", "beds", "bedstead", "mattress on a frame"],
    "chair": ["chair", "chairs", "seat", "armchair", "office chair"],
    "sofa": ["sofa", "couch", "settee", "loveseat", "sectional sofa"],
    "table": ["table", "tables", "dining table", "desk table", "coffee table"],
    "door": ["door", "doors", "doorway", "door frame"],
    "window": ["window", "windows", "window pane", "glass window"],
    "picture": ["picture", "pictures", "painting", "poster", "framed photo", "wall art"],
    "counter": ["counter", "counters", "countertop", "worktop", "bar counter"],
    "desk": ["desk", "desks", "writing desk", "work desk"],
    "curtain": ["curtain", "curtains", "drape", "drapes", "window curtain"],
    "refrigerator": ["refrigerator", "fridge", "freezer", "refrigerator unit"],
    "shower curtain": ["shower curtain", "shower curtains", "bath curtain"],
    "toilet": ["toilet", "toilets", "wc", "lavatory", "toilet bowl"],
    "sink": ["sink", "sinks", "washbasin", "basin", "kitchen sink"],
    "bathtub": ["bathtub", "bath tub", "bath", "tub"],
    "bookshelf": ["bookshelf", "bookshelves", "bookcase", "shelving unit", "shelf"],
}


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb(words):
        with torch.no_grad():
            f = model.encode_text(tok(list(words)))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    banks = {}
    banks["one_word"] = l2n(emb(NAMES) - emb(NAMES).mean(0, keepdims=True))
    tpl = np.stack([emb([p.format(n) for n in NAMES]) for p in PROMPTS])
    tpl_c = l2n(tpl - tpl.mean(1, keepdims=True))
    banks["templates_mean"] = l2n(l2n(tpl.mean(0)) - l2n(tpl.mean(0)).mean(0, keepdims=True))
    # per-variant banks: shape (V, K, D), each centred over the vocabulary
    V = max(len(LEX[n]) for n in NAMES)
    rows = [emb(LEX[n] + [LEX[n][0]] * (V - len(LEX[n]))) for n in NAMES]
    lex = np.transpose(np.stack(rows), (1, 0, 2))       # (V, K, D)
    lex = l2n(lex - lex.mean(1, keepdims=True))         # centre each variant over the vocabulary
    generic = ("{}", "a photo of a {}", "the {}", "a {} in a room", "a close-up of a {}",
               "a {} seen from above", "part of a room: {}", "a large {}", "a small {}")
    gen = np.stack([emb([p.format(n) for n in NAMES]) for p in generic])
    gen = l2n(gen - gen.mean(1, keepdims=True))
    both = np.concatenate([tpl_c, lex], 0)
    agg = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "scene*.npz"))):
        S = os.path.basename(p)[:-4]
        z = np.load(p, allow_pickle=False)
        gt = z["gt"].astype(np.int64)
        keep = z["keep"] if "keep" in z.files else np.ones(len(gt), bool)
        nag = torch.load(os.path.join(ROOT, "output", "scannet", S, "sai_nag_3.0.pt"), map_location="cpu")
        lv = [np.asarray(x).ravel().astype(np.int64) for x in nag["nag"]]
        li = len(nag["nag_feat"]) - 1
        X = l2n(np.asarray(nag["nag_feat"][li].detach().float().cpu().numpy()))
        Xc = l2n(X - X.mean(0, keepdims=True))

        def lse(bank):
            sc = np.stack([Xc @ bank[i].T for i in range(len(bank))])
            m = sc.max(0)
            return TAU * np.log(np.exp((sc - m) / TAU).mean(0)) + m

        res = {"one_word": Xc @ banks["one_word"].T, "templates_mean": Xc @ banks["templates_mean"].T,
               "templates_lse": lse(tpl_c), "generic_lse": lse(gen), "lexicon_lse": lse(lex),
               "both_lse": lse(both)}
        out = {}
        for k, sc in res.items():
            pred = np.array([classes[i] for i in sc.argmax(1)])[lv[li + 1]]
            mm = metrics(gt[keep], pred[keep], classes)
            out[k] = 100 * mm["mIoU"]
            if args.emit and k == "both_lse":
                d = dict(scene=S, center="both_lse", subsets={})
                for kk, cls in SUBSETS.items():
                    d["subsets"][str(kk)] = metrics(gt[keep], pred[keep], cls)
                od = os.path.join(ROOT, "results", "scannet_digs")
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, S + ".json"), "w") as fh:
                    json.dump(d, fh)
        for k, v in out.items():
            agg.setdefault(k, []).append(v)
        print("%-16s " % S + " ".join("%s %.1f" % (k, out[k]) for k in out), flush=True)
    print("\n  " + "  ".join("%s %.2f" % (k, np.mean(v)) for k, v in agg.items()))


if __name__ == "__main__":
    main()
