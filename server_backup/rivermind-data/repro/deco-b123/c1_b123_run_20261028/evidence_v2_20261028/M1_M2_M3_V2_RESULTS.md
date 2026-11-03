# M1–M3 V2 robust complementary readout results

本轮固定使用官方 NAG、Gaussian、B1 bank、observations 和原始 DeCo evaluator 协议。
`off` 是原始 DeCo 主路径；`semantic` 为 M1+M3；`reliable` 为 M1+M2+M3。
所有辅助修正均为 `main + clamp(0.25 * delta, 0, 0.25)`，没有根据 GT 调参。

## SHA256 provenance

```json
{
  "ramen": {
    "nag_sha256": "10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324",
    "bank_sha256": "73e4511df01725fb7b6e6ed235971c509434aa6603d1aaf233eba5cdd6f54aa1",
    "observation_sha256": "6ae30f6e03b01177b14e900a447d71ee788ef57a31cac9909b0abf0d33358232"
  },
  "waldo_kitchen": {
    "nag_sha256": "6cd39db45d60031ab04eac18630a918c311890de22220e07678118a6da903098",
    "bank_sha256": "8bf6b51bce581045c588a5967b9219c7b67fdf818dc10e8193754c90e70a7ad3",
    "observation_sha256": "2586d87fb86eb4dee8885889b6e057363450ef710e06d4e2ca8d9fa72702fbf8"
  }
}
```

## Scene summary

| scene | config | scene mIoU | flat mIoU | improved | degraded | unchanged | total gain | total loss | triggered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ramen | off | 0.518076 | 0.511650 | — | — | — | — | — | 0 |
| ramen | semantic | 0.529587 | 0.522873 | 6 | 1 | 64 | 0.811264 | 0.014447 | 71 |
| ramen | reliable | 0.529587 | 0.522873 | 6 | 1 | 64 | 0.811264 | 0.014447 | 71 |
| waldo_kitchen | off | 0.659209 | 0.681240 | — | — | — | — | — | 0 |
| waldo_kitchen | semantic | 0.659209 | 0.681240 | 0 | 0 | 22 | 0.000000 | -0.000000 | 22 |
| waldo_kitchen | reliable | 0.659209 | 0.681240 | 0 | 0 | 22 | 0.000000 | -0.000000 | 22 |

## Protocol and interpretation

V2 `off` first passed strict per-query equality against the existing A1 results: ramen 71/71 and waldo_kitchen 22/22 query keys matched, with zero IoU, predicted-pixel, or candidate-count differences.
The auxiliary branch never replaces a root score with a mode average. It computes `advantage=max(mode_score-main_score,0)`, masks modes with fewer than two independent view IDs, applies the fixed 0.08 temperature only over eligible modes, and adds a capped non-negative correction.
`semantic` uses reliability 1 for eligible modes; `reliable` uses query-independent group semantic consistency in [0,1]. Single-view modes remain in the bank but cannot trigger auxiliary correction.

## M2 support counts

| scene | valid bank modes | single-view modes | multi-view modes |
|---|---:|---:|---:|
| ramen | 722 | 189 | 533 |
| waldo_kitchen | 1152 | 327 | 825 |
## Negative-result safeguard

- ramen/semantic: positive queries=6, negative queries=1, unchanged=64, net ΔIoU=0.796817.
- ramen/reliable: positive queries=6, negative queries=1, unchanged=64, net ΔIoU=0.796817.
- waldo_kitchen/semantic: positive queries=0, negative queries=0, unchanged=22, net ΔIoU=0.000000.
- waldo_kitchen/reliable: positive queries=0, negative queries=0, unchanged=22, net ΔIoU=0.000000.

No coefficient, temperature, mode budget, NAG, observation, or bank was changed after seeing GT. This report stops here; no follow-up tuning was started.
