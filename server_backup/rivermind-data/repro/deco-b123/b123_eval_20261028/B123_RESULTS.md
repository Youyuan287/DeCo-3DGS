# B123 frozen ramen results

All five arms use the same frozen NAG and the newly exported observation file. No historical 304-region bank is used.

## Protocol

- scene: ramen; center=image; text=canon; renorm=0; selection=gap; topk=5; level_until=3; filter_small=50; temperature=0.08.
- NAG SHA256: `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535`.
- observation SHA256: `3c857c05d101029fd332fbdbb4e499b4d0c8d1b67a643121a9b7fdb7d32a7fc6`.

## Overall metrics

| Arm | mIoU | flat mIoU | queries | mean candidates | mean predicted px |
|---|---:|---:|---:|---:|---:|
| B0 | 0.457138 | 0.449149 | 71 | 2.070 | 23057.3 |
| K0 | 0.457138 | 0.449149 | 71 | 2.141 | 23057.3 |
| B1 | 0.457135 | 0.449144 | 71 | 1.789 | 14998.9 |
| B2 | 0.457135 | 0.449144 | 71 | 1.789 | 14998.9 |
| B3 | 0.437974 | 0.426904 | 71 | 2.042 | 21903.7 |

## Paired IoU differences (first arm minus second arm)

| Comparison | n | mean | min | max | positive / negative |
|---|---:|---:|---:|---:|---:|
| K0 - B0 | 71 | 0.00000000 | 0.00000000 | 0.00000000 | 0 / 0 |
| B1 - K0 | 71 | -0.00000510 | -0.00036200 | 0.00000000 | 0 / 1 |
| B2 - B1 | 71 | 0.00000000 | 0.00000000 | 0.00000000 | 0 / 0 |
| B3 - B2 | 71 | -0.02224021 | -0.56645600 | 0.05264500 | 6 / 16 |

## B2 ROFA audit

- K0: dropped observations = `0`, effective geometry = `False`, bank SHA256 = `1e12a4acea743bcae25bf43ad7d8757337156e9547ec16644fc0a22fc9397555`.
- B1: dropped observations = `0`, effective geometry = `True`, bank SHA256 = `07962f0c387c217fad224b337aee97febde4c5187712c1e2376552a192baec07`.
- B2: dropped observations = `0`, effective geometry = `True`, bank SHA256 = `7978967c6709c7c424b7fb9568be47014ea32032837c43224994184cbe793a37`.
- B3 directly loads `B2.npz`; its bank SHA256 is identical to B2.
- Candidate and per-query records are in `B0/per_query.csv`, `K0/per_query.csv`, `B1/per_query.csv`, `B2/per_query.csv`, and `B3/per_query.csv`.

## Interpretation

B1-K0 isolates debiased semantic plus geometry-supported grouping under the supplied constants. B2-B1 isolates the supplied mode-internal ROFA. B3-B2 isolates query-conditioned weighting. Any near-zero B2-B1 is reported as the measured net ROFA effect; no threshold was changed.
