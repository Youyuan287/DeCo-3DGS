# Evaluation protocols

Every number in `RESULTS.md` was produced by the code here, under one of the protocols below. This
file exists because the conventions differ between the papers we compare against, and the differences
are larger than some of the effects being reported.

## 1. ScanNet-10: which classes, and which average

The evaluated scenes are the ten used by ReLaGS / OpenGaussian:

```
scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00
scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00
```

**Class list (K19), 19 NYU40 ids:**

```
1 wall, 2 floor, 3 cabinet, 4 bed, 5 chair, 6 sofa, 7 table, 8 door, 9 window,
10 bookshelf, 11 picture, 12 counter, 14 desk, 16 curtain, 24 refrigerator,
28 shower curtain, 33 toilet, 34 sink, 36 bathtub
```

**`bookshelf` has zero annotated points in all ten scenes.** Verified twice: in the per-Gaussian GT
and in the raw `<scene>_vh_clean_2.labels.ply` (0 points in every scene). The divisor is therefore
**18**, not 19.

Subsets keep the same class list minus ids:

| subset | ids | classes with GT |
|---|---|---|
| K19 | 19 ids | **18** |
| K15 | `1,2,3,4,5,6,7,8,9,10,12,14,16,33,34` | **14** |
| K10 | `1,2,4,5,6,7,8,9,10,33` | **9** |

If you prefer a fixed 19-class divisor (absent class counted as zero), multiply the K19 means by
18/19: 58.55 -> 55.47 for our method, 40.79 -> 38.64 for the reproduced baseline. The gain is then
+16.82 instead of +17.76.

### pooled vs macro

Two aggregations are in use in the literature and they are not interchangeable:

* **dataset-pooled** (`scannet/aggregate.py`): sum intersection and union per class over the ten
  scenes, then average over the 18 present classes. This is what ReLaGS's published 41.26 matches:
  our reproduction is 40.79 pooled, and 39.94 macro.
* **macro / scene-mean** (OpenGaussian's `calculate_metrics`): per scene, average IoU over the classes
  present *in that scene*, then average the ten scene means.

We report both. OpenGaussian additionally drops Gaussians with `sigmoid(opacity) < 0.1` and forces
`pred = 0` where `gt == 0`; we do neither, so our numbers are not directly comparable to theirs even
under the macro convention.

## 2. LERF-Mask (Gaussian Grouping protocol)

* 3 scenes (figurines, ramen, teatime), 61 queries, test views from the GG split.
* retrieval: ReLaGS multi-level superpoint search (`topk=5`, `level_until=3`, `filter_small=50`).
* render: per-Gaussian binary selection -> rasterised semantics, threshold 0.5.
* metric: IoU and Boundary-IoU with `dilation_ratio=0.02`, code identical to GG's
  `script/eval_lerf_mask.py`.
* aggregate: mean over views per prompt, then mean over prompts.

This is a **single-query retrieval** protocol: there is no class denominator at all.

## 3. LERF-OVS

* 4 scenes (figurines, ramen, teatime, waldo_kitchen), 208 (image, query) samples over 22 frames.
* metric: per (image, query) binary IoU -> mean over queries within the image -> mean over images ->
  mean over scenes. This is `eval_seg.py`'s `mIoU`, and it is what the reported number means; the
  `flat=` column in our logs is the mean over queries and differs by ~0.6 points.
* no class denominator.

## 4. Why the retrieval benchmarks show no gain (and must be reported as such)

For a single query, with `m = mean_i(f)`:

```
<f_i - m, t> = <f_i, t> - <m, t>
```

The second term does not depend on `i`, so **image-side mean removal cannot change the ranking of
regions**. In a multi-class readout, the same term becomes `<m, t_c>`, which *does* depend on the
class, and it changes `argmax_c`. This is the whole difference between ScanNet (18 classes competing)
and LERF (one query at a time).

Measured consequences:

* direct check, teatime, 5 queries x 3 levels: `max_i |(s1 - s0) - const|` = 5e-8..1e-7, and the
  selected top-5 superpoint sets are identical 15/15;
* LERF-OVS: 173/208 queries bit-identical, mIoU 0.6401 -> 0.6393;
* LERF-Mask: 57/61 queries bit-identical, mIoU 0.6800 -> 0.7112 (the +3.12 comes from 4 queries).

With the released non-linear canon readout the shift is no longer exactly constant (subtracting `m`
adds a *different* constant to each canon word's logit), which is the residual ±3 point wobble.

**Text-side (C2) is active on retrieval and there it hurts**: in a classification readout every class
gets the same "max over its lexicon" advantage, but in a single-query readout widening the bank only
widens the false-positive surface. Measured on LERF-OVS: LSE over a 63-entry lexicon −14.7, mean over
the lexicon −11.0, lexicon with the canon readout preserved −7.2 (all against 0.6401). C2 is
therefore only applied where the readout is multi-class.

## 5. Reproduction anchors

| what | value | how |
|---|---|---|
| ScanNet reproduced baseline, pooled | 40.79 / 56.69 | `scannet/` with all operators off |
| ScanNet published (ReLaGS) | 41.26 / 57.29 | their Table 9 |
| LERF-Mask baseline, pure CLIP | 0.6800 / 0.6444 | `lerf_mask/` with `--center none` |
| LERF-OVS baseline | 0.6401 | `lerf_ovs/` with `--center none`; their paper reports 64.40 |

If your run does not reproduce these four numbers, the harness is misconfigured; do not read the
deltas.


## 6. The applicability gate is code, not prose

```python
from deco_ops import cmr, should_center

cmr(feats)                     # ||mean(F)|| / rms(F - mean(F)), scale-free in the norm and in D
should_center(feats, 1.6)      # the decision
```

Measured: ScanNet superpoint features CLIP ViT-B-16 **2.14** (1.60-3.12 across scenes), OpenSeg-768
**1.13** (0.93-2.04). LERF-Mask: CLIP 1.92 / 2.06 / 1.84, OpenSeg 1.05 / 1.24 / 1.09.

**What the shipped harnesses actually do with it: nothing automatic.** `score_scannet.py` and the two
LERF harnesses apply the operator whenever `--center image` is passed; the gate is a policy function
you call, not a switch that fires inside the loop. On the ScanNet CLIP arm that is exactly right — CMR
is 2.14 in every scene, so the gate would always open and the explicit flag changes nothing. On the
OpenSeg arm it would close, which is why that arm is reported as out of scope (the operator forced on
takes 61.64 to 47.05) rather than as a failure of the method.

To let the gate decide, call it:

```python
X = feats[level]                     # (N, D) for one scene, one level
if should_center(X):
    X = X - X.mean(axis=0)
```
