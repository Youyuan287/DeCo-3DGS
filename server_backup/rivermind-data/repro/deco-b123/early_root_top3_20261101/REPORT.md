# Early-root top-3 paired experiment

Status: **COMPLETE**. Rows: 208/208.

Arm A is this experiment's original main/gap baseline. The historical gateE 0.6760 is an independent anchor; neither arm is identified as gateE by this summary.

Primary metric: query mean within image, image mean within scene, then equal mean over the four scenes. Partial inputs are explicitly provisional.

| Metric | A | B | B - A |
|---|---:|---:|---:|
| Image/scene macro (primary) | 0.639262 | 0.546909 | -0.092354 |
| All-query mean | 0.625692 | 0.555821 | -0.069870 |
| Mean of scene-query means | 0.639623 | 0.551741 | -0.087882 |
| Pool oracle, image/scene macro | 0.707397 | 0.661103 | -0.046294 |

B-A primary paired 95% CI: [-0.152159, -0.040303]; 35 wins / 46 losses / 127 ties.
Pool oracle is GT-based diagnostic candidate capacity, not deployable accuracy.

## Per scene

| Scene | n | A | B | Delta | A pool oracle | B pool oracle | Roots changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| figurines | 56 | 0.585869 | 0.526451 | -0.059418 | 0.734317 | 0.699030 | 56 |
| ramen | 71 | 0.518076 | 0.407993 | -0.110083 | 0.627519 | 0.580864 | 71 |
| teatime | 59 | 0.793896 | 0.799475 | +0.005579 | 0.801153 | 0.822173 | 59 |
| waldo_kitchen | 22 | 0.659209 | 0.453716 | -0.205493 | 0.666598 | 0.542344 | 22 |

## Families

Family rows below use within-family query means, not the primary global macro. Families are taken unchanged from the input CSV.

| Family | n | A query mean | B query mean | Delta | Wins | Losses |
|---|---:|---:|---:|---:|---:|---:|
| deep | 19 | 0.226267 | 0.541651 | +0.315384 | 9 | 0 |
| found | 124 | 0.816010 | 0.708588 | -0.107422 | 6 | 27 |
| rank2 | 36 | 0.663525 | 0.403896 | -0.259629 | 5 | 18 |
| recall | 29 | 0.026642 | 0.100494 | +0.073852 | 15 | 1 |

## Integrity and intervention

- Maximum baseline reproduction error: 0.
- Changed root sets: 208 query-views, 67 distinct scene/prompt units.
- Mean pool size A/B: 5.274/6.572.
- Mean B observation roots: 370.760.
- Mean B fallback slots: 0.000; 0 rows have a fallback slot.
- Bootstrap: 2000 paired image-cluster draws within scenes, seed 20261101.
- Statistical intervals describe these paired scenes, not new-scene generalization.
