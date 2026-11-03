"""Is `l2n(f - m)` a no-op for the decision? With a single word per class, yes. With a tau-LSE over
several words per class, no -- and that asymmetry is the whole answer.

    LSE_tau(c * a) = c * LSE_{tau/c}(a)

Scaling the similarities by a row constant changes the *effective temperature* of the marginalisation,
so the average over the class's words is taken differently, and the argmax over classes can move.
"""
import os
import sys

import numpy as np
import torch

ROOT = "/root/rivermind-data"
sys.path.insert(0, ROOT + "/scripts")
from eval_scannet_3d import NYU40, SUBSETS  # noqa: E402

SCENE = sys.argv[1] if len(sys.argv) > 1 else "scene0000_00"
classes = SUBSETS[19]
TAU = 0.02
PROMPTS = ("{}", "a photo of a {}", "the {}", "a {} in a room")


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


nag = torch.load(os.path.join(ROOT, "output", "scannet", SCENE, "sai_nag_3.0.pt"), map_location="cpu")
li = len(nag["nag_feat"]) - 1
X = l2n(np.asarray(nag["nag_feat"][li].detach().float().cpu().numpy()))
m = X.mean(0, keepdims=True)
A = X - m
B = l2n(X - m)

# a stand-in bank with V variants per class, all unit norm: the shape the real lexicon has
rng = np.random.default_rng(0)
V, K, D = 4, len(classes), X.shape[1]
bank = l2n(rng.normal(size=(V, K, D)).astype(np.float32))


def lse_scores(Z, bank, tau=TAU):
    sc = np.stack([Z @ bank[v].T for v in range(len(bank))])   # (V, N, K)
    mx = sc.max(0)
    return tau * np.log(np.exp((sc - mx) / tau).mean(0)) + mx


for V_used in (1, 4):
    bk = bank[:V_used]
    ia = lse_scores(A, bk).argmax(1)
    ib = lse_scores(B, bk).argmax(1)
    diff = int((ia != ib).sum())
    print("V=%d  argmax differs %3d / %d  (%.2f%%)" % (V_used, diff, len(ia), 100 * diff / len(ia)))

# the margin distribution: are the decisions close enough for the effective-temperature shift to matter?
sc = lse_scores(A, bank)
srt = np.sort(sc, axis=1)
margin = srt[:, -1] - srt[:, -2]
print("top1-top2 margin: median %.4f  p25 %.4f  p10 %.4f  frac<1e-3 %.1f%%"
      % (np.median(margin), np.percentile(margin, 25), np.percentile(margin, 10),
         100 * (margin < 1e-3).mean()))

# norm spread: how much of a per-row temperature change renormalisation introduces
nrm = np.linalg.norm(X - m, axis=1)
print("||f - m|| per superpoint: median %.3f  p10 %.3f  p90 %.3f  (1/norm ratio %.1fx)"
      % (np.median(nrm), np.percentile(nrm, 10), np.percentile(nrm, 90), nrm.max() / nrm.min()))

# homogeneous? check LSE directly on one row
a = (A[0] @ bank[0].T)
c = 1.0 / nrm[0]
l_raw = TAU * np.log(np.exp((a - a.max()) / TAU).mean())
l_scaled = TAU * np.log(np.exp((c * a - (c * a).max()) / TAU).mean())
print("row0 LSE(a) = %.4f   c*LSE(a) = %.4f   LSE(c*a) = %.4f   ->  ratio %.3f (c = %.3f)"
      % (l_raw, c * l_raw, l_scaled, l_scaled / l_raw, c))