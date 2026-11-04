**ScanNet-10 ablation. Pooled (P5) and macro (P6) rows are marked and must not be mixed.**

| ID | Ablation | P | Base | Value | $\Delta$ | Status | R |
|---|---|---|---|---|---|---|---|
| S-02 | baseline | P5 | -- | 40.79 | -- | anchor | R3 |
| S-03 | M1-C1 | P5 | 40.79 | 53.67 | +12.88 | active | R3 |
| S-04 | M2-C2 | P5 | 40.79 | 56.32 | +15.53 | active | R3 |
| S-05 | M1+M2-renorm1 | P5 | 40.79 | 57.85 | +17.06 | active | R3 |
| S-06 | M1+M2-renorm0 | P5 | 40.79 | 58.55 | +17.76 | active | R3 |
| S-07 | baseline-macro | P6 | -- | 39.94 | -- | anchor | R3 |
| S-08 | M1+M2-macro | P6 | 39.94 | 53.30 | +13.36 | active | R3 |
| S-09 | rebuilt-baseline | P5 | -- | 41.05 | -- | anchor | R1 |
| S-10 | M1+M2-rebuilt | P5 | 41.05 | 56.05 | +15.00 | active | R1 |
| S-11 | rebuilt-macro | P6 | 38.87 | 49.89 | +11.02 | active | R1 |
| S-12 | M1+M2-rebuilt-renorm1 | P5 | 41.05 | 56.59 | +15.54 | active | R1 |
| S-13 | zR | P5 | 40.36 | 45.98 | +5.62 | superseded | R2 |
| S-15 | zR-plus-E0-wrong-text | P5 | 45.98 | 48.77 | +2.79 | superseded | R2 |
| S-16 | zR-plus-VCLS | P5 | 40.36 | 49.16 | +8.80 | negative | R2 |
| S-17 | certified-chain-C1C2 | P5 | 40.36 | 50.18 | +9.82 | superseded | R2 |
| S-29 | C1-C2-on-openseg | P7 | 43.80 | 44.57 | +0.77 | negative | R1 |
| S-34 | VCLS | P5 | 45.98 | -- | +0.39 | negative | R2 |
| S-35 | certificate-on-class-axis | P5 | 50.18 | 45.91 | -4.27 | negative | R2 |
| S-36 | local-contrast | P5 | 48.77 | -- | -- | negative | R2 |
| S-37 | matrix-normalisation | P5 | 48.77 | 45.54 | -- | negative | R2 |
| S-38 | ICM-neighbourhood | P5 | 51.46 | 31.38 | -- | negative | R2 |
| S-39 | decision-vote | P5 | 47.97 | 41.12 | -- | negative | R2 |

S-02/S-03--S-08 are the shipped-field anchor (R3: the 366-cluster field is missing). S-09--S-12 are the rebuilt 386-cluster field (R1) and sit 1.3--3.4 points below the anchor. S-13/S-15/S-17 are superseded: score-side $zR$, wrong text space, and the early per-view-mean implementation of C1. S-37/S-38/S-39 are summarised from the ledger.
