**Diagnostic ceilings and reference points. These are not deployable methods.**

| ID | Bench. | Diagnostic | Metric | Value | What it bounds |
|---|---|---|---|---|---|
| O-16 | LERF-OVS | option-oracle | mIoU | 67.5 | Per-query best of released/C/mask; auditor consumes 88.6% |
| O-17 | LERF-OVS | single-candidate-rank1 | mIoU | 55.6 | In-harness single-candidate readout; NOT comparable to P1 |
| O-18 | LERF-OVS | pool-oracle | mIoU | 70.3 | Upper bound of candidates already in the pool (0.6948-0.7031) |
| O-19 | LERF-OVS | non-empty-flag-single-candidate | mIoU | 59.8 | Single-candidate diagnostic readout only; the +4.2 does not transfer to the official readout (O-34) |
| O-20 | LERF-OVS | score-plus-logarea | mIoU | 60.8 | Still single-candidate and below the released union readout |
| O-28 | LERF-OVS | M8-grounded-sam-median | median-IoU | 72.2 | CLIP mask ranking vs detector-top1-box -> SAM, LERF-OVS 208 queries; overall a wash, GSAM slightly worse |
| O-29 | LERF-OVS | M8-grounded-sam-ge50 | rate | 57.2 | Fraction of queries with IoU>=0.5: 57.7% CLIP vs 57.2% GSAM |
| O-30 | LERF-OVS | M8-gsam-waldo | median-IoU | 78.8 | CLIP weakest scene: the only place GSAM wins big |
| O-31 | LERF-OVS | M8-gsam-figurines | median-IoU | 33.3 | CLIP strongest scene: GSAM destroys it. The sign swap is the whole story |
| M-10 | LERF-MASK | prefix-oracle | mIoU | 73.0 | Per-query best k; gap rule already equals the global best fixed k |
| M-11 | LERF-MASK | CMR | ratio | 1.92 | CLIP 1.92/2.06/1.84 vs OpenSeg 1.05/1.24/1.09; gate threshold 1.6 |
| M-12 | LERF-MASK | detection-bridge | mIoU | 93.0 | Oracle GT boxes 0.9299/0.9108; off-the-shelf detector 0.6606/0.6399; SAM+CLIP 0.5448/0.5264 |
| M-13 | LERF-MASK | M8-gsam-2d-probe | median-IoU | 92.5 | CLIP-failed 22 queries on LERF-MASK data, not LERF-OVS; must not be paired with 64.01 |
| S-01 | ScanNet | published-ReLaGS | mIoU | 41.26 | ReLaGS Table 9 |
| S-30 | ScanNet | partition-ceiling | mIoU | 76.90 | Perfect root partition semantics; current 51.7 => 25.2 points are read-side loss |
| S-31 | ScanNet | level-ceilings | mIoU | 90.90 | L0 0.909 / L1 0.855 / L2 0.816 / L3 0.769; L1/L0 have over-segmentation artefacts |
| S-33 | ScanNet | single-query-ap | AP | 0.58 | 30 class instances; -a_g 0.5568, -b_c 0.5797 (exact null), centering 0.5042 |
| D-02 | 3D-OVS | published-ReLaGS | mIoU | 92.50 | Paper Table 13 mean; Bed 95.4 / Bench 94.4 / Lawn 91.2 / Room 94.8 / Sofa 86.9 |
| D-03 | 3D-OVS | same-family-repro | mIoU | 66.70 | THGS reproduction, same family as our 70.05 |
| D-04 | 3D-OVS | gaussian-oracle | mIoU | 95.94 | bed; all gaussians inside GT projection; matches paper 94.4 => protocol is correct |
| D-05 | 3D-OVS | level0-oracle | mIoU | 92.45 | bed, 7192 clusters |
| D-06 | 3D-OVS | level1-oracle | mIoU | 77.35 | bed, 2211 clusters; finest layer that has features |
| D-07 | 3D-OVS | level2-oracle | mIoU | 77.76 | bed, 149 clusters |
| D-09 | 3D-OVS | level0-retrieval | mIoU | 52.17 | bed, z2 threshold; finer level retrieves worse |
| D-12 | 3D-OVS | masks-per-view | count | 53.00 | bed 53 / bench 412 / lawn 598 / room 72 / sofa 131; healthy line >100 |

An oracle may select with ground truth or may pick the best option post hoc; neither is a method. O-28 is the only row measured on 2D masks without a 3D lift.
