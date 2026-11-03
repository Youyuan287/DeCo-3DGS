# Results

All numbers were produced by the code in this directory on the ten certified ScanNet-10 scenes and
the LERF-Mask / LERF-OVS split. Protocols are in `PROTOCOLS.md`; the class divisor is 18, not 19.

## 1. ScanNet-10, K19 (dataset-pooled, CLIP ViT-B-16)

| variant | mIoU | mAcc | note |
|---|---|---|---|
| ReLaGS, published | 41.26 | 57.29 | their Table 9 |
| ReLaGS, reproduced here | 40.79 | 56.69 | certified baseline |
| + C1 image side only | 53.67 | 66.67 | |
| + C2 templates only | 56.32 | 68.11 | |
| + C1 + C2, `--renorm 1` | 57.85 | 69.07 | +17.06 over the reproduction |
| **+ C1 + C2, `--renorm 0` (unified, the default)** | **58.55** | **69.82** | **+17.76** |

`--renorm 0` is the default of `scannet/score_scannet.py` and the definition in
`deco_ops.center_np`: `f <- f - mean(f)`, with **no** L2 renormalisation afterwards. `--renorm 1`
adds `l2n(f - mean)`, costs 0.70 mIoU, and survives only because the earlier ablations were run with
it. Both come out of the same entry point: `REGRESSION.md` §1 has the commands and the expected
output verbatim.

The 0.70 gap is **not** cosmetic and **not** noise: with a single word per class the renormalisation
would be invisible to `argmax`, but the tau-LSE over a class's words is not homogeneous
(`LSE_tau(c*a) = c * LSE_{tau/c}(a)`), so it changes the effective temperature. Measured: 0/366
superpoints move with one word per class, **109/366 (29.8%)** with four. See PROTOCOLS.md §4.1.

Subsets, renormalised variant: K19 57.85 / 69.07, K15 57.13, K10 61.13. K10 is 0.95 lower than the
templates-only arm (62.08); with a 9-class denominator that is a single-class effect and we report it.

### Per-class IoU (dataset-pooled, unified operator)

| class | GT points | baseline | DeCo | delta |
|---|---|---|---|---|
| wall | 1,284,233 | 3.20 | 52.85 | +49.65 |
| floor | 766,350 | 58.11 | 80.22 | +22.11 |
| cabinet | 521,315 | 57.26 | 67.71 | +10.45 |
| bed | 354,671 | 80.18 | 86.74 | +6.56 |
| chair | 342,623 | 74.71 | 65.12 | −9.59 |
| door | 300,697 | 42.91 | 44.11 | +1.20 |
| counter | 269,716 | 27.43 | 22.30 | −5.13 |
| table | 204,988 | 33.76 | 57.28 | +23.52 |
| curtain | 200,566 | 66.50 | 69.51 | +3.01 |
| refrigerator | 165,103 | 70.78 | 67.94 | −2.84 |
| window | 133,902 | 0.93 | 1.23 | +0.30 |
| desk | 126,691 | 35.21 | 48.86 | +13.65 |
| picture | 108,967 | 13.76 | 39.56 | +25.80 |
| sofa | 95,744 | 26.20 | 76.39 | +50.19 |
| toilet | 56,553 | 72.47 | 84.96 | +12.49 |
| sink | 42,067 | 8.57 | 49.80 | +41.23 |
| shower curtain | 17,962 | 0.00 | 79.41 | +79.41 |
| bathtub | 4,212 | 62.25 | 59.94 | −2.31 |
| **mean (18)** | | **40.79** | **58.55** | **+17.76** |

14/18 classes improve. The four that drop are lexical collisions, not mechanism failures: `chair`
(seat/armchair vs sofa), `counter` (absorbed by table/desk), `refrigerator` (freezer/cabinet),
`bathtub` (bath). `window` at 1.23 is a disaster class in both rows and is reported as such.

## 2. ScanNet-10, macro protocol (per-scene mean, then mean over scenes)

| variant | mIoU |
|---|---|
| baseline (four templates averaged, no operator) | 39.94 |
| + C1 + C2, renormalised | 53.62 |
| **+ C1 + C2, unified** | **53.30** |

Both protocols agree on the direction and differ by ~4.4 points in magnitude, because pooling
rewards the large classes and that is where the gain is concentrated.

## 3. Ablations (ScanNet, dataset-pooled unless stated)

**2x2 over the two operators**

| | text: bare prompt | text: lexicon + templates |
|---|---|---|
| image: raw | 39.02 | 46.82 |
| image: centred | 53.67 | **57.85** |

Interaction is **−3.62**, i.e. sub-additive: each operator overlaps with the other's information.

**Lexicon ladder (scene-mean, image side centred)**

| text representation | mIoU |
|---|---|
| bare class name | 42.53 |
| 4 templates averaged | 48.55 |
| 4 templates, tau-LSE | 50.76 |
| 9 generic templates, tau-LSE | 49.63 |
| hand-written lexicon, tau-LSE | 53.17 |
| templates + lexicon | 53.62 |

**Other region-side operators** (all at the same operating point, base 40.79): image mean 49.32;
image + text 51.23; text only 38.19; mean + drop PC1 47.51; projection removal 49.04; excess-only
50.06; top-1 hub penalised 43.53; **hub direction only 40.95**; hub + text 40.97; mean -> hub 50.82;
partial removal at 25/50/75/100% = 45.04 / 46.31 / 48.38 / 51.23.

The ablation localises the effect to the representation: only removing the *shared* component on both
sides helps; every other correction of the region side is neutral or worse.

**Encoder boundary.** CLIP has a common-mode ratio (CMR = ||mean(F)|| / rms(F - mean)) of 2.14
(1.60-3.12 across scenes); OpenSeg-768 has 1.13 (0.93-2.04). On OpenSeg the same operator goes
61.64 -> 47.05, so the operator is gated on CMR > 1.6 rather than applied unconditionally.

## 4. LERF-Mask (Gaussian Grouping protocol, 61 queries)

| arm | mIoU | BIoU |
|---|---|---|
| baseline, pure CLIP | 0.6800 | 0.6444 |
| **+ C1** | **0.7112** | **0.6655** |

Per scene: figurines 0.7268 -> 0.7268, ramen 0.5645 -> 0.7198, teatime 0.7487 -> 0.6871.
**57 of 61 queries are bit-identical**; the whole gain is `pork belly` (+0.936) and the whole loss is
`stuffed bear` (−0.616). This is consistent with the no-op argument in `PROTOCOLS.md` §4 and is
reported as a scope check, not as a SOTA claim.

For reference, with a detection bridge (GroundingDINO + SAM) instead of the text retrieval this
benchmark is usually reported on: oracle GT boxes 0.9299 / 0.9108; off-the-shelf detector 0.6606 /
0.6399; SAM + CLIP 0.5448 / 0.5264. Published instance-bridged numbers are not comparable to the
pure-CLIP rows.

## 5. LERF-OVS (208 samples, per-image mean)

| arm | mIoU | delta | bit-identical |
|---|---|---|---|
| baseline (reproduces 64.01; paper reports 64.40) | 0.6401 | — | — |
| + C1 | 0.6393 | −0.08 | 173/208 |
| + C1 with an added L2 renormalisation | 0.4686 | −17.15 | — |
| + C1 + z-score per level | 0.6326 | −0.75 | — |
| C2, 4 templates tau-LSE | 0.6003 | −4.0 | — |
| C2, 63-entry lexicon tau-LSE | 0.4929 | −14.7 | — |
| C2, lexicon averaged | 0.5304 | −11.0 | — |
| C2, lexicon with canon readout kept | 0.5679 | −7.2 | — |

**Analysis (not a benchmark result): closed-set readout.** Replacing the per-query retrieval with the
ScanNet-style argmax over the categories present in the image, on the same features:

| readout | baseline | + C1 | delta |
|---|---|---|---|
| single-query retrieval | 0.6401 | 0.6393 | −0.08 |
| closed-set argmax | 0.2558 | 0.2829 | **+2.71** |

Empty assignments drop 30.3% -> 3.4%, and IoU on non-degenerate queries rises 0.287 -> 0.463. The
closed-set readout is much weaker in absolute terms (masks are ~2.5x oversized), so this is evidence
about *which axis the operator acts on*, not a number to put in a comparison table.

## 6. Negative results (kept because they constrain the story)

These are scene-mean with `--renorm 1`, so the reference point is 48.55 (four templates averaged),
**not** 58.55.

| attempt | mIoU | where it lives | what it shows |
|---|---|---|---|
| region as a distribution: `E[f(x)]` over children posteriors | 47.15 | `scannet/ablation_region_posterior.py` | mixing posteriors loses to classifying the mean |
| coarse posterior pushed down to the fine level | 44.46 | same script | the coarse advantage is region count, not context |
| tree smoothing, noise ~ 1/members | 30.59 (fine 25.66) | working tree `cli_tree.py` | no clean parameterisation, not shipped |
| oracle scene histogram | 55.57 (+7.02) | working tree | an upper bound, kept as future work |
| naive EM / leave-one-scene prior / MAP-EM | 42.36 / 47.53 / 43.36 | working tree | priors self-reinforce |
| auto-mined lexicon, k = 3/5/8/12 | 52.67 / 47.30 / 45.11 / 39.64 | working tree `cli_prune.py` | text-space neighbours are co-occurrence, not synonymy |

**Scope limit on the first row, stated explicitly.** 47.15 tests *posterior mixing*: average the
children's class posteriors, then decide. It does **not** test *retaining the raw multi-view evidence*
(the `(Gaussian, view, mask)` triples, voted into class space at the very end). That experiment
exists in the working tree as `obs_bank.py` / `vote_final.py`, is **not shipped here, and has no
shipped number**. The two differ in where the averaging happens, so 47.15 cannot be cited as
evidence against evidence retention, and `RESULTS.md` should not be read as having closed that
direction.

Leave-one-word-out over the 18 classes: worst slot −2.86, and removing slot 4 *helps* (+0.69), so the
lexicon is not individually tuned.

Bootstrap over scenes: +17.06 with 95% CI [5.47, 23.06], P(improvement) = 1.000.

## 7. Cost

2DGS reconstruction 15-36 min/scene, SAM + CLIP encoding 22-100 min/scene, and the operator itself
< 10 ms/scene (one mean subtraction; the text encodings are cached). No training, no new parameters,
no labels.