# ScanNet-10 baseline reproduction (certified)

The upstream repository ships no ScanNet evaluation code, so the baseline used by `deco/` was
reimplemented. This note records what it reproduces and how to re-run it.

## Protocol

* ten scenes: 0000, 0062, 0070, 0097, 0140, 0200, 0347, 0400, 0590, 0645;
* each Gaussian takes the label of its nearest GT point (`<scene>_vh_clean_2.labels.ply`, NYU40);
* prediction: finest-level superpoint feature x class-name text embedding, `argmax` over the class
  subset, then every Gaussian of that superpoint inherits the label;
* metric: per-class intersection/union **pooled over the ten scenes**, averaged over the classes
  with ground truth. `bookshelf` (id 10) has zero annotated points in all ten scenes, so the K19
  divisor is **18** (K15 -> 14, K10 -> 9). See `deco/PROTOCOLS.md`.

## Result

| variant | K19 mIoU | mAcc |
|---|---|---|
| ReLaGS, published (Table 9) | 41.26 | 57.29 |
| **this reproduction** | **40.79** | **56.69** |

Under the macro protocol (per-scene mean, then mean over scenes) the same run gives 39.94, which is
what OpenGaussian-style evaluations report; the published 41.26 is closer to the pooled number, so
we assume ReLaGS uses pooled.

## Code

`deco/scannet/score_scannet.py` (scoring, all ablation arms) and `deco/scannet/aggregate.py`
(dataset-level aggregation). Run with `deco/scripts/run_scannet.sh`; it needs the `sai_nag_3.0.pt`
artefacts produced by the upstream pipeline for each scene.
