# Regression anchors

Four numbers, four commands, and the exact code that produced them. If a run does not reproduce these,
the harness is misconfigured and no delta from it means anything.

```
export RIVERMIND_ROOT=/path/to/artefacts      # output/scannet/<scene>/sai_nag_3.0.pt, clip_cache/, scripts/
export RELAGS_REPO=/path/to/ReLaGS            # scene/, gaussian_renderer/, nag_data.py
export HF_HUB_OFFLINE=1
PY=python3
```

## 1. ScanNet-10, K19 (the two operator variants)

One entry point, one flag. `--renorm 0` is the unified operator that every table in `RESULTS.md` uses;
`--renorm 1` is the renormalised variant, kept because it is the one the earlier ablations were run with.

```bash
export SCANNET_OUT=/tmp/deco_reg               # both scorer and aggregator read this
$PY deco/scannet/score_scannet.py --renorm 0   # -> 58.55 pooled
$PY deco/scannet/score_scannet.py --renorm 1   # -> 57.85 pooled
$PY deco/scannet/aggregate.py --results /tmp/deco_reg
```

Expected output (2026-10-26, ten scenes, CLIP ViT-B-16, 18-class divisor):

```
=== center=both_lse  renorm=0  scenes=10  K19  divisor=18   <- unified (58.55)
    pooled  mIoU  58.55   mAcc  69.82
    macro   mIoU  53.30

=== center=both_lse  renorm=1  scenes=10  K19  divisor=18   <- renormalised (57.85)
    pooled  mIoU  57.85   mAcc  69.07
    macro   mIoU  53.62
```

The per-scene print from the scorer also reproduces the scene-mean lexicon ladder
(`one_word 42.53 / templates_mean 48.55 / templates_lse 48.10 or 50.76 / generic_lse 49.95 or 49.63 /
lexicon_lse 53.81 or 53.17 / both_lse 53.30 or 53.62`), where the first value of each pair is
`--renorm 0`.

## 2. LERF-OVS baseline and the C1 arm

`--renorm 0` is the default; it is also the only setting under which the operator is a constant shift
(PROTOCOLS.md §4). `--renorm 1` walks into a different branch and lands near 0.47.

```bash
$PY deco/lerf_ovs/eval_lerf_ovs.py --scene ramen ... --center none  --renorm 0   # 0.6401 overall
$PY deco/lerf_ovs/eval_lerf_ovs.py --scene ramen ... --center image --renorm 0   # 0.6393 overall
```

| arm | overall mIoU | queries unchanged |
|---|---|---|
| `--center none` | 0.6401 | — |
| `--center image --renorm 0` | 0.6393 | 173 / 208 |
| `--center image --renorm 1` | 0.4686 | — |

## 3. LERF-Mask

```bash
$PY deco/lerf_mask/eval_lerf_mask.py --scene ramen ... --center image --renorm 0
```

| arm | mIoU | BIoU |
|---|---|---|
| `--center none` | 0.6800 | 0.6444 |
| `--center image --renorm 0` | 0.7112 | 0.6655 |

57 of 61 queries are bit-identical between the two rows, so this row is a scope check, not a gain.

## 4. The gate

```bash
$PY deco/deco_ops.py
```

prints the numpy/torch agreement (8.9e-8 on the score path) and nothing else. The gate itself is
`deco_ops.cmr` / `deco_ops.should_center`; measured values are in PROTOCOLS.md §6.

## What is *not* covered by these anchors

* the ablation arms behind `RESULTS.md` §3 (operator zoo, lexicon ladder, CMR table): they live in the
  working tree's `cli_syn.py`, not in this package, and are reproduced by that script;
* the LERF-OVS `--text` arms (C2 on retrieval): same, `eval_lerf_ovs.py --text lexicon` reproduces the
  negative, but the lexicon files in `lerf_ovs/lexicon.json` are the ones used;
* the detection-bridge numbers (RESULTS.md §4): GroundingDINO + SAM, not shipped here.