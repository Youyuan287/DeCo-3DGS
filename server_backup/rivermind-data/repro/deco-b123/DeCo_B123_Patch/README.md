# DeCo B1/B2/B3 增量代码

这是**完整 DeCo-3DGS 仓库的增量补丁**，不是独立 ReLaGS 工程。原始 `deco/lerf_ovs/eval_lerf_ovs.py`、`deco/deco_ops.py`、`nag_data.py` 不修改。

- `evidence_modes.py`: B1（去偏 + 视角空间支持联合分组）及 B2（模式内 ROFA）。
- `build_bank.py`: 从已存在、ROFA 前的生产观测生成 K0/B1/B2 银行；含冻结 NAG 与观测 SHA。
- `evidence_readout.py`: B3 查询自适应加权，并供 B1/B2 使用固定先验融合。
- `patch_deco_evaluator.py`: 自动基于仓库**现有完整 DeCo 评测脚本**产生单独的 `eval_lerf_ovs_b123.py`；禁止覆写已有文件。
- `install.sh`: 保守安装；检测同名文件拒绝覆盖。
- `run_b123.sh`: 同一 NAG、同一 DeCo 配置、五臂嵌套消融 B0/K0/B1/B2/B3；自动校验银行来源 SHA。
- `tests/test_b123.py`: CPU 单元测试。
- `CODEX_TASK.md`: 详细执行步骤、验收条件和科学结论边界。

**尚未进行 GPU 端到端测试。** 仅保留真实已认证 DeCo 文本语义与 NAG 搜索规则，未保证 GPU 渲染结果。
