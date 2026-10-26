"""Region as a distribution: E[f(x)] instead of f(E[x]).

A region descriptor is the mean of its members, and the pipeline classifies that mean. The label
of a region is what its members support, which is the mixture of members' posteriors. Jensen
says the two differ; the measurement says it matters: 47.15 mIoU against 48.55 for classifying
the mean (renorm=1, K19, scene mean). Reported in RESULTS.md section 6 as a negative result.

Shipped verbatim from the working tree (script `cli_post.py`), with paths parameterised.

Usage: python ablation_region_posterior.py
"""

import glob
import os
import sys

import numpy as np

ROOT = os.environ.get("RIVERMIND_ROOT", ".")
CACHE = ROOT + "/clip_cache"
sys.path.insert(0, os.environ.get("SCANNET_SCRIPTS", os.path.join(ROOT, "scripts")))
from eval_scannet_3d import SUBSETS, metrics  # noqa: E402

classes = SUBSETS[19]
TAU = 0.05


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def post(s):
    z = (s - s.max(1, keepdims=True)) / TAU
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    import torch
    from clip_diag import text_emb
    T = l2n(text_emb().astype(np.float32))
    Tc = l2n(T - T.mean(0, keepdims=True))
    agg = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "scene*.npz"))):
        S = os.path.basename(p)[:-4]
        z = np.load(p, allow_pickle=False)
        gt = z["gt"].astype(np.int64)
        keep = z["keep"] if "keep" in z.files else np.ones(len(gt), bool)
        nag = torch.load(os.path.join(ROOT, "output", "scannet", S, "sai_nag_3.0.pt"), map_location="cpu")
        lv = [np.asarray(x).ravel().astype(np.int64) for x in nag["nag"]]
        feats = [l2n(np.asarray(f.detach().float().cpu().numpy())) for f in nag["nag_feat"]]
        Ps, Ns, pars = [], [], []
        for li, X in enumerate(feats):
            Xc = l2n(X - X.mean(0, keepdims=True))
            Ps.append(post(Xc @ Tc.T))
            Ns.append(np.maximum(np.bincount(lv[li + 1]).astype(np.float64), 1.0))
            if li < len(feats) - 1:
                par = np.full(len(X), -1, dtype=np.int64)
                for a, b in np.unique(np.stack([lv[li + 1], lv[li + 2]], 1), axis=0):
                    par[a] = b
                pars.append(par)

        def mix_up(li):
            """mixture of a node's children posteriors, combined with its own"""
            child_p, child_n, par = Ps[li], Ns[li], pars[li]
            K = child_p.shape[1]
            num = np.zeros((Ps[li + 1].shape[0], K))
            den = np.zeros(Ps[li + 1].shape[0])
            np.add.at(num, par, child_p * child_n[:, None])
            np.add.at(den, par, child_n)
            mix = num / np.maximum(den, 1e-9)[:, None]
            w = den / (den + Ns[li + 1])                      # weight by how much evidence the children carry
            return w[:, None] * mix + (1 - w[:, None]) * Ps[li + 1]

        def emit(posterior, level):
            lab = posterior.argmax(1)
            pred = np.array([classes[i] for i in lab])[lv[level + 1]]
            return 100 * metrics(gt[keep], pred[keep], classes)["mIoU"]

        res = {"arg_l1": emit(Ps[0], 0), "arg_l2": emit(Ps[1], 1), "arg_l3": emit(Ps[2], 2)}
        m3 = mix_up(1)                                        # level-3 node from its level-2 children
        m2 = mix_up(0)
        res["mix_l3"] = emit(m3, 2)
        res["mix_l2"] = emit(m2, 1)
        # push the coarse mixture back down and let it reweight the fine posteriors
        for beta in (0.3, 0.5, 0.7):
            up = m3[pars[1]]                                  # each level-2 node's group context
            q2 = (1 - beta) * Ps[1] + beta * up
            back = q2[pars[0]]                                # each level-1 node's group context
            q1 = (1 - beta) * Ps[0] + beta * back
            res["down_l1_b%.1f" % beta] = emit(q1, 0)
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
        print("%-16s " % S + " ".join("%s %.1f" % (k, res[k]) for k in
              ("arg_l1", "arg_l3", "mix_l3", "down_l1_b0.5")), flush=True)
    print("\n  " + "  ".join("%s %.2f" % (k, np.mean(v)) for k, v in agg.items()))


if __name__ == "__main__":
    main()
