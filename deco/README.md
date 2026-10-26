# DeCo-3DGS

Reference implementation of **DeCo** (*De-Collapse*), two training-free operators that remove the
scene-wide common mode from open-vocabulary 3D Gaussian Splatting representations.

The operators act on the **representation**, not on the network: no training, no new parameters, no
labels. They are applied once, at test time, to the per-superpoint language features and to the text
embeddings that a ReLaGS-style pipeline already computes.

```
C1  image side :  f_i  <-  f_i - mean_i(f)                       (per scene, per hierarchy level)
C2  text  side :  t    <-  l2n(t - mean_c(t)),  s(i,c) = tau * logmeanexp_{w in L(c)} <f_i, w>
```

This directory contains the code that produced the numbers in `RESULTS.md` on ScanNet-10 and
LERF-Mask, plus the LERF-OVS harness used to test the scope of the method.

## Headline numbers

| benchmark | readout | baseline | + DeCo | delta |
|---|---|---|---|---|
| ScanNet-10, K19, dataset-pooled | 18-class argmax | 40.79 | **58.55** | **+17.76** |
| ScanNet-10, K19, macro | 18-class argmax | 39.94 | **53.30** | **+13.36** |
| LERF-Mask (Gaussian Grouping) | single-query retrieval | 0.6800 | 0.7112 | +3.12 |
| LERF-OVS | single-query retrieval | 0.6401 | 0.6393 | −0.08 |

The first two rows are the claim. The last two are a *scope check*, not a gain: a single-query
retrieval score is `<f_i - m, t> = <f_i, t> - <m, t>`, and the subtracted term does not depend on the
region index, so C1 is a constant shift and cannot change the ranking. It is measured to be inert
(173/208 LERF-OVS queries bit-identical, 57/61 on LERF-Mask). See `PROTOCOLS.md` §4.

## Layout

```
deco_ops.py              the two operators, one implementation (numpy + torch)
scannet/                 ScanNet-10 scoring, metrics, dataset-level aggregation
lerf_mask/               LERF-Mask evaluation (Gaussian Grouping protocol)
lerf_ovs/                LERF-OVS evaluation (+ the per-query lexicon used for the C2 arms)
scripts/                 runners
PROTOCOLS.md             evaluation conventions, including the 18-vs-19 class divisor
RESULTS.md               full tables, ablations, and the negative results
```

## Requirements

The harnesses are thin wrappers around a [ReLaGS](https://github.com/) installation: they import
`scene`, `gaussian_renderer`, `arguments` and `nag_data` from it, and they expect the usual
`output/<scene>/sai_nag_3.0.pt` artefacts (per-level superpoint labels + CLIP features).

```bash
export RELAGS_REPO=/path/to/ReLaGS          # repo root that contains scene/, gaussian_renderer/
export RIVERMIND_ROOT=/path/to/artefacts    # where output/<scene>/sai_nag_3.0.pt live
export HF_HUB_OFFLINE=1
```

The CLIP encoder is `ViT-B-16 laion2b_s34b_b88k` (512-d). Text features are L2-normalised.

## Quick start

```bash
# ScanNet-10: score 10 scenes, then aggregate to dataset level
python scannet/score_scannet.py            # per-scene JSON -> $RIVERMIND_ROOT/results/scannet_digs
python scannet/aggregate.py                # dataset-pooled mIoU / mAcc, per class

# LERF-Mask, image-side operator only
python lerf_mask/eval_lerf_mask.py --scene figurines ... --center image

# LERF-OVS
python lerf_ovs/eval_lerf_ovs.py --scene ramen ... --center image --text canon
```

See `scripts/run_*.sh` for complete command lines.

## What this is not

* Not a new reconstruction method: the geometry, superpoints and features are ReLaGS's.
* Not SOTA on LERF-Mask: our +3.12 comes from 4 of 61 queries, and the other 57 are bit-identical.
* Not comparable to instance-bridged numbers (DIGS and friends report detection-bridged results on
  LERF-Mask; on the pure-CLIP path our reproduction of ReLaGS is 0.6800 against their 0.68–0.69).

## Citation

Paper in preparation. If you use this code before it appears, please cite the ReLaGS pipeline it
builds on.