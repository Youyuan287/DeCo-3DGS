# Frozen ramen NAG observation export

Date: 2026-10-28  
Status: **completed and validated**

This is a new observation export for the frozen `sai_nag_3.0.pt`; it is not a recovery of the historical 304-region observation file and is not mixed with any historical bank.

## Inputs and provenance

| Item | Path | SHA-256 |
|---|---|---|
| Frozen NAG | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt` | `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535` |
| Gaussian PLY (iteration 0) | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/point_cloud/iteration_0/point_cloud.ply` | `e7d2fb7a725b882efe3c8f4cc9f0f64fc5cbe5a36866f0d8bee353fbd9b68276` |
| `nag-l1.pt` | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/nag-l1.pt` | `878b36dbace822d27197e6371d11361380d973d9182460de8bafc2c383c01288` |
| Exported observations | `/root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz` | `3c857c05d101029fd332fbdbb4e499b4d0c8d1b67a643121a9b7fdb7d32a7fc6` |

The isolated ReLaGS copy is `/root/rivermind-data/repro/deco-b123/frozen_ramen_relags`; its adapted `merge_proj.py` hash is `2b2d78e397e62f6b19960b4f5e62c6300eb124ec744790559ba196e1a0542c3a`. The user-supplied export script hash is `2ce043dc6f7791311cd9418ad8a0b2c080ef63cf902221400bfe024b5f9baa5c`. Production ReLaGS was not modified.

## Command executed

```bash
HF_HUB_OFFLINE=1 /root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3 \
  /root/rivermind-data/repro/deco-b123/export_frozen_ramen_observations.py \
  --source /root/rivermind-data/repro/deco-b123/frozen_ramen_source \
  --model /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen \
  --nag /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt \
  --output /root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz \
  --iteration 0 --seed 1234
```

The isolated sink preserves the original Gaussian ray tracing, mask-to-Gaussian projection, visibility weighting and CLIP feature computation. It only records frozen-NAG observations and CSR arrays. No `assign_label()`, `everything_seg()`, clustering, `merge_proj.py` regeneration or training was run. A float32 cast was added only at the existing matrix multiplication boundary because the stored language maps are float16 while the projection matrix is float32; projection math and normalization are unchanged.

## Exported data and validation

- Gaussian count: `372270`; all four NAG label vectors have length `372270`.
- NAG region counts (levels 0/1/2/3): `7936 / 1860 / 490 / 245`.
- `nag[0]` equals `nag-l1.pt` element-by-element.
- Training views: `131`; language feature files: `262` (`131 *_f.npy` and `131 *_s.npy`).
- Observations: `8486`; unique level-3 region IDs: `244`; views represented: `131`.
- Features: `(8486, 512)`, float32, all finite.
- Weights: all finite and strictly positive; minimum `2.4933925e-05`.
- Member CSR: offsets length `8487`, ID length/tail `33421977`.
- Visible CSR: offsets length `8487`, ID length/tail `14674402`.
- All offsets are monotonic and terminate at their corresponding ID-array length.
- Region IDs are within the frozen level-3 range; Gaussian IDs are within `[0, 372270)`.

The output is therefore accepted as a valid frozen-NAG production observation export. B123 uses this file exclusively.
