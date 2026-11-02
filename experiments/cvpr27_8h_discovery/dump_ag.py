"""Dump the region-axis bias a_g (z-scored) per LERF scene.

a_g = <f_g, tbar_V>, f_g = mean of the region's per-view features,
tbar_V = normalized mean of a FIXED generic vocabulary. Pure offline statistic:
no query, no label, no GT.
"""
import os, sys, json
import numpy as np

REPO = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "deco", "lerf_ovs"))
from lerf_axis_bias import VOCAB_A, VOCAB_B  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402

SAS = "/root/rivermind-data/repro/deco-b123/m1_sas_20261029"
OUT = "/root/rivermind-data/repro/deco-b123/axis_20261103"
SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]

vlm = ClipSimMeasure()
vlm.load_model()


def vocab_dir(words):
    T = []
    for w in words:
        vlm.encode_text(w)
        T.append(vlm.text_feature[0].detach().cpu().numpy().astype(np.float64).ravel())
    T = np.stack(T)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-8)
    m = T.mean(0)
    return m / max(np.linalg.norm(m), 1e-9)


ta, tb = vocab_dir(VOCAB_A), vocab_dir(VOCAB_B)
os.makedirs(OUT, exist_ok=True)
for sc in SCENES:
    b = os.path.join(SAS, sc + "_new")
    if not os.path.isdir(b):
        b = os.path.join(SAS, "waldo_new")
    d = np.load(os.path.join(b, "observations_level3.npz"))
    F = d["features"].astype(np.float32)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    reg = d["region_ids"].astype(np.int64)
    n = int(reg.max()) + 1
    freg = np.zeros((n, F.shape[1]))
    cnt = np.bincount(reg, minlength=n).astype(np.float64)
    for i in range(len(F)):
        freg[reg[i]] += F[i]
    freg /= np.maximum(cnt, 1)[:, None]
    freg /= np.maximum(np.linalg.norm(freg, axis=1, keepdims=True), 1e-8)
    a1, a2 = freg @ ta, freg @ tb
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    np.savez(os.path.join(OUT, "%s_ag.npz" % sc),
             a1z=z(a1), a2z=z(a2), nviews=np.bincount(reg, minlength=n),
             corr=float(np.corrcoef(a1, a2)[0, 1]))
    print("DUMP", sc, n, "corr %.4f" % np.corrcoef(a1, a2)[0, 1], flush=True)