# C1 × B123 controlled ablation

Fixed protocol: `text=canon`, `renorm=0`, `sel=gap`, `topk=5`, `level_until=3`, `filter_small=50`, `temperature=0.08`. Each scene uses its own NAG, observation and matching banks.

## ramen

| Config | mIoU | flat mIoU | queries | mean candidates | mean predicted px |
|---|---:|---:|---:|---:|---:|
| A0 | 0.511494 | 0.504576 | 71 | 2.423 | 20364.5 |
| A1 | 0.518076 | 0.511650 | 71 | 2.113 | 20602.0 |
| K0_off | 0.471401 | 0.459251 | 71 | 1.915 | 25076.8 |
| K0_on | 0.494300 | 0.483218 | 71 | 1.746 | 24459.4 |
| A2 | 0.471401 | 0.459251 | 71 | 1.817 | 25076.8 |
| A3 | 0.494300 | 0.483218 | 71 | 1.676 | 24459.4 |
| A4 | 0.473882 | 0.459858 | 71 | 2.338 | 25124.3 |
| A5 | 0.492156 | 0.481220 | 71 | 2.465 | 24693.4 |

Paired IoU deltas (second minus first):
- `A1-A0`: mean=0.00707414, n=71, positive/negative/unchanged=4/8/59, range=[-0.20702100, 0.47267300]
- `K0_on-K0_off`: mean=0.02396768, n=71, positive/negative/unchanged=11/1/59, range=[-0.12882700, 0.47267300]
- `A3-A2`: mean=0.02396768, n=71, positive/negative/unchanged=11/1/59, range=[-0.12882700, 0.47267300]
- `A5-A4`: mean=0.02136155, n=71, positive/negative/unchanged=7/1/63, range=[-0.12882700, 0.47267300]
- `A2-K0_off`: mean=0.00000000, n=71, positive/negative/unchanged=0/0/71, range=[0.00000000, 0.00000000]
- `A3-K0_on`: mean=0.00000000, n=71, positive/negative/unchanged=0/0/71, range=[0.00000000, 0.00000000]
- `A4-A2`: mean=0.00060746, n=71, positive/negative/unchanged=6/5/60, range=[-0.37777500, 0.16017500]
- `A5-A3`: mean=-0.00199866, n=71, positive/negative/unchanged=3/9/59, range=[-0.09013900, 0.04880700]

Derived effects:
- C1_effect_base = 0.00658181
- C1_effect_B1 = 0.02289953
- C1_effect_full = 0.01827440
- B1_gain_no_C1 = 0.00000000
- B1_gain_with_C1 = 0.00000000
- Full_gain_no_C1 = 0.00248083
- Full_gain_with_C1 = -0.00214430
- Interaction_B1 = 0.01631772
- Interaction_Full = 0.01169259

## waldo_kitchen

| Config | mIoU | flat mIoU | queries | mean candidates | mean predicted px |
|---|---:|---:|---:|---:|---:|
| A0 | 0.606221 | 0.613749 | 22 | 2.091 | 65264.6 |
| A1 | 0.659209 | 0.681240 | 22 | 2.500 | 66210.1 |
| K0_off | 0.572682 | 0.580355 | 22 | 2.864 | 66626.6 |
| K0_on | 0.518620 | 0.531229 | 22 | 2.909 | 71283.0 |
| A2 | 0.572607 | 0.580230 | 22 | 2.818 | 66496.5 |
| A3 | 0.517367 | 0.529911 | 22 | 2.545 | 71423.4 |
| A4 | 0.564571 | 0.574799 | 22 | 2.364 | 63969.3 |
| A5 | 0.530180 | 0.543501 | 22 | 1.909 | 65090.7 |

Paired IoU deltas (second minus first):
- `A1-A0`: mean=0.06749118, n=22, positive/negative/unchanged=4/4/14, range=[-0.00946100, 0.97566900]
- `K0_on-K0_off`: mean=-0.04912545, n=22, positive/negative/unchanged=1/10/11, range=[-0.94929100, 0.04459300]
- `A3-A2`: mean=-0.05031905, n=22, positive/negative/unchanged=0/6/16, range=[-0.94929100, 0.00000000]
- `A5-A4`: mean=-0.03129886, n=22, positive/negative/unchanged=2/5/15, range=[-0.94929100, 0.82028800]
- `A2-K0_off`: mean=-0.00012423, n=22, positive/negative/unchanged=1/1/20, range=[-0.00615100, 0.00341800]
- `A3-K0_on`: mean=-0.00131782, n=22, positive/negative/unchanged=3/1/18, range=[-0.04117500, 0.01190900]
- `A4-A2`: mean=-0.00543095, n=22, positive/negative/unchanged=8/3/11, range=[-0.80978200, 0.49411000]
- `A5-A3`: mean=0.01358923, n=22, positive/negative/unchanged=9/2/11, range=[-0.09222200, 0.18501300]

Derived effects:
- C1_effect_base = 0.05298827
- C1_effect_B1 = -0.05523973
- C1_effect_full = -0.03439035
- B1_gain_no_C1 = -0.00007512
- B1_gain_with_C1 = -0.00125343
- Full_gain_no_C1 = -0.00803603
- Full_gain_with_C1 = 0.01281335
- Interaction_B1 = -0.10822800
- Interaction_Full = -0.08737862

