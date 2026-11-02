# REPRODUCE

## 环境
- 服务器 fj01（4090 24G），`/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3`
- 代码：`/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS`
- LERF 观测：`/root/rivermind-data/repro/deco-b123/m1_sas_20261029/<scene>_new/observations_level3.npz`
- LERF 官方读权：`/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/<scene>`
- GT：`/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label`
- ScanNet 认证场：`/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029`

## 运行
```bash
# LERF 行偏置（4 场景，约 3 分钟/场景）
bash run_axis.sh           # 在 fj01 的 deco-b123 目录下
# 审计器基线（本地）
python analysis/lerf_witness_certificate.py analysis/results/20261101/cnone_gate_tf \
  analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_ev \
  analysis/results/20261101/witness none 0.5 sc
# 审计器 + a_g（本地；先重建 shift）
python experiments/cvpr27_8h_discovery/build_n1_shifted.py analysis/results/20261101 \
  analysis/results/20261103_axis/n1 analysis/results/20261103_axis/ag 1.0 a1z
python analysis/lerf_witness_certificate.py analysis/results/20261101/cnone_gate_tf \
  analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_ev \
  analysis/results/20261103_axis/n1/witness_ag none 0.5 sc
# 诊断
python experiments/cvpr27_8h_discovery/probe_signal.py
python experiments/cvpr27_8h_discovery/probe_views.py
python experiments/cvpr27_8h_discovery/probe_rank.py
```

## 未提交的大文件（可再生成）
`analysis/results/20261103_axis/n1*/` 是 shift 后的 witness CSV（约 12.6 MB），
由 `build_n1_shifted.py` 从 `analysis/results/20261101/` + `ag/*.npz` 重建，故不入库。

## 文献占位检查
```bash
cd experiments/cvpr27_8h_discovery
python tools/litcheck.py --cat cs.CV --q 'all:"conformal" AND all:segmentation' --since 2025-01-01
python tools/litcheck.py --ids 2608.16103 --abstracts
```
依赖：`curl`（内置 WebSearch/WebFetch 在本机不可用）。退出码 2 = 未取到响应，判 UNVERIFIED。
