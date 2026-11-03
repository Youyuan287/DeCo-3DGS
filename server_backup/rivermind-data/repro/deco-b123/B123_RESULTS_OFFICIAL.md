# Official-NAG B123 frozen ramen results

Previous frozen-NAG results are invalidated. These five arms use the official `sai_nag.pt`, one newly exported observation file, and one protocol.

## Inputs

- NAG SHA256: `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324`.
- Observation SHA256: `6ae30f6e03b01177b14e900a447d71ee788ef57a31cac9909b0abf0d33358232`.
- Protocol: center=image, text=canon, renorm=0, sel=gap, topk=5, level_until=3, filter_small=50, temperature=0.08.

## Overall metrics

| Arm | mIoU | flat mIoU | queries | mean candidates | mean predicted px |
|---|---:|---:|---:|---:|---:|
| B0 | 0.518076 | 0.511650 | 71 | 2.113 | 20602.0 |
| K0 | 0.494300 | 0.483218 | 71 | 1.746 | 24459.4 |
| B1 | 0.494300 | 0.483218 | 71 | 1.676 | 24459.4 |
| B2 | 0.494300 | 0.483218 | 71 | 1.676 | 24459.4 |
| B3 | 0.492156 | 0.481220 | 71 | 2.465 | 24693.4 |

## Paired IoU differences

| Comparison | n | mean | min | max | positive / negative |
|---|---:|---:|---:|---:|---:|
| K0 - B0 | 71 | -0.02843215 | -0.92936700 | 0.23062600 | 11 / 7 |
| B1 - K0 | 71 | 0.00000000 | 0.00000000 | 0.00000000 | 0 / 0 |
| B2 - B1 | 71 | 0.00000000 | 0.00000000 | 0.00000000 | 0 / 0 |
| B3 - B2 | 71 | -0.00199866 | -0.09013900 | 0.04880700 | 3 / 9 |

## ROFA and bank audit

- K0: dropped observations=`0`, effective geometry=`False`, bank SHA256=`844c4b04aa2494d64baab9b67ca2fdaa438c8e84cad6bbcfa76c74847b32a8c1`.
- B1: dropped observations=`0`, effective geometry=`True`, bank SHA256=`73e4511df01725fb7b6e6ed235971c509434aa6603d1aaf233eba5cdd6f54aa1`.
- B2: dropped observations=`0`, effective geometry=`True`, bank SHA256=`68f1b7ad3eee33670539caba1d49c853cc6bbef587f3bf4e4b0ec80235bdc81e`.
- B3 directly loads B2.npz; B2/B3 bank SHA256 is identical.
- All per-query CSVs contain 71 queries.

## Conclusion

B0 now matches the historical official DeCo value (`0.5181`). K0/B1/B2/B3 are controlled ablations against this same official-NAG baseline and the same newly exported observation source.
