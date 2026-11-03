# M1-RCS Results

All four scenes use the same certified V2 evaluator, M2 OFF, Softmax M3, coefficient 0.25, temperature 0.08, and frozen observations/NAG. RCS changes only M1 seed selection. CSV IoUs are recorded to six decimal places; the scene summaries retain higher precision.

| scene | queries | DeCo off | Original V2 | M1-RCS V2 | RCS−V2 |
|---|---:|---:|---:|---:|---:|
| figurines | 56 | 0.585868976 | 0.593810503 | 0.593810503 | +0.000000000 |
| ramen | 71 | 0.518075618 | 0.529587430 | 0.529587430 | +0.000000000 |
| teatime | 59 | 0.793896209 | 0.793896209 | 0.793896209 | +0.000000000 |
| waldo_kitchen | 22 | 0.659209087 | 0.659209087 | 0.659209087 | +0.000000000 |
| macro | 208 | 0.639262472 | 0.644125807 | 0.644125807 | +0.000000000 |

Compared with Original V2 at the CSV precision: improved 0, degraded 0, unchanged 208. Candidate ID set changed in 5/208 queries; rendered mask hash changed in 0/208. Gaussian ID sets are not serialized in these cached CSVs, so no Gaussian-set change is claimed. The `Gaussian +/−` and `mask XOR` values in the focus lines are each evaluator row's auxiliary-vs-main trace, not an RCS-vs-Original-V2 difference.

## Focus queries

- figurines / frame_00041 / tesla door handle: V2=0.067802, RCS=0.067802, Δ=+0.000000, Gaussian +21 / -2445, mask XOR=467 px.
- figurines / frame_00105 / tesla door handle: V2=0.305141, RCS=0.305141, Δ=+0.000000, Gaussian +21 / -2445, mask XOR=9055 px.
- ramen / frame_00006 / nori: V2=0.819620, RCS=0.819620, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=3498 px.
- ramen / frame_00024 / corn: V2=0.000000, RCS=0.000000, Δ=+0.000000, Gaussian +14024 / -0, mask XOR=167693 px.
- ramen / frame_00024 / nori: V2=0.979292, RCS=0.979292, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=20299 px.
- ramen / frame_00060 / corn: V2=0.000000, RCS=0.000000, Δ=+0.000000, Gaussian +14024 / -0, mask XOR=106519 px.
- ramen / frame_00065 / corn: V2=0.000000, RCS=0.000000, Δ=+0.000000, Gaussian +14024 / -0, mask XOR=73624 px.
- ramen / frame_00065 / nori: V2=0.802344, RCS=0.802344, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=11274 px.
- ramen / frame_00081 / nori: V2=0.972114, RCS=0.972114, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=2282 px.
- ramen / frame_00119 / corn: V2=0.000000, RCS=0.000000, Δ=+0.000000, Gaussian +14024 / -0, mask XOR=171946 px.
- ramen / frame_00119 / nori: V2=0.966265, RCS=0.966265, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=10878 px.
- ramen / frame_00128 / corn: V2=0.000307, RCS=0.000307, Δ=+0.000000, Gaussian +14024 / -0, mask XOR=57472 px.
- ramen / frame_00128 / nori: V2=0.953370, RCS=0.953370, Δ=+0.000000, Gaussian +0 / -2881, mask XOR=4877 px.
- waldo_kitchen / frame_00053 / ottolenghi: V2=0.975669, RCS=0.975669, Δ=+0.000000, Gaussian +0 / -0, mask XOR=0 px.
