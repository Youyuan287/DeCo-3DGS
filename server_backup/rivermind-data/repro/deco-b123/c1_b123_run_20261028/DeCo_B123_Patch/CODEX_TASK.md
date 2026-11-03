# Codex 执行任务书：完整 DeCo-3DGS 上的 B1/B2/B3 嵌套消融

**唯一上游**：`https://github.com/Youyuan287/DeCo-3DGS` 当前已检出的完整仓库。首先记录 `git rev-parse HEAD` 和工作树状态。`deco/lerf_ovs/eval_lerf_ovs.py`、`deco/deco_ops.py`、`nag_data.py`、`merge_proj.py` 不允许覆盖或改写。该 ZIP 包为增量补丁，不是完整的仓库替代品。

## 核心定义和开关

| 组 | --control-k1 | --b1 | --b2 | --b3 | 比较含义 |
|---|---:|---:|---:|---:|---|
| B0 | 0 | 0 | 0 | 0 | 完整 DeCo 生产基线（已经过全区域 ROFA） |
| K0 | 1 | 0 | 0 | 0 | 同一份 ROFA 前观测、单证据，不经过原生产 ROFA |
| B1 | 0 | 1 | 0 | 0 | 3.1+3.2：场景共享方向残差分组，几何支持参与模式组织，模式内原始 CLIP 加权均值；查询先验固定 |
| B2 | 0 | 1 | 1 | 0 | B1 + 每个模式内部独立进行原版 ROFA；不使用跨模式共识删除少数模式 |
| B3 | 0 | 1 | 1 | 1 | B2 + query-conditioned evidence weighting；**银行与 B2 完全相同** |

B1 是否有效：比较 `B1-K0`，同时单独报告 `B0` 与 `K0` 差（原生产 ROFA 对照）。B2 净收益：`B2-B1`。B3 净收益：`B3-B2`。所有结果必须来自同一个冻结 NAG / 原始观测 / DeCo 评测协议。

**三个核心算法**：

1. B1：从 production observation 原始 CLIP 特征 `f` 估计无标签全场景公共方向 `u=norm(mean(norm(f)))`，计算 `h=norm(norm(f)-<norm(f),u>u)`；模式间相似度 `S=(1-gamma)*(h_i·h_j)+gamma*(2*Jaccard(V_i,V_j)-1)`，`V_i` 是真实 `visible_gaussian_ids`，`gamma=0.25` 固定首版；使用确定性加权 medoid + farthest-first 选择不超过 3 个证据模式；组内按原始 visibility 权重加权原始 `f`。不直接用残差代替完整语义。每个模式记录成员观测、视角来源与有效支持 CSR。`K0` 使用同一观测单向量聚合。
2. B2：**只在 B1 每个模式内部**应用原版 ROFA 均值两两余弦阈值（`tau=3.0`），模式间不互删。报告每模式被删除观测数和变化区域数。由于小组内 tau=3 可能不触发，若 B2 与 B1 相同须如实标注为筛选无作用，不进行阈值扫描。
3. B3：先让每条证据走**完整 DeCo 已有** `text_sims`（含 C1/C2、canon / lexicon / text templates、原检索协议），得到 `s_{rk}(q)`；B1/B2 使用固定 `prior` 加权，B3 使用 `softmax(s/T+log(prior))` 加权；只替换 root 层分数，其余层级使用原 DeCo 输出。不得替换为原始余弦 + 硬 top-5。固定 `T=0.08`，不调参。

## 工程安装

1. 从 ZIP 解压 `DeCo_B123_Patch/`。先确认仓库没有同名新增文件，然后：

   ```bash
   bash /path/to/DeCo_B123_Patch/install.sh /path/to/DeCo-3DGS
   python -m unittest discover -s /path/to/DeCo_B123_Patch/tests -v
   ```

2. **只允许读取已有的** `observations_level3.npz` 和与其对应的 `sai_nag_3.0.pt`；严禁重新运行 `merge_proj.py`、`sp_partition.py`、CLIP 提取、3DGS 重建等历史步骤。NAG 的 SHA 必须固定，银行文件自动记录 NAG 和观测 SHA。旧缓存如果没有 visible Gaussian CSR，必须停止并汇报 keys，不准悄悄把几何约束关闭。要跑仅语义退化对照可显式设置 `GEO_WEIGHT=0`，并标注 `NO_GEOMETRY`，但它不算完整 B1。

3. 环境变量（按服务器真实位置填写，绝对路径），示例：

   ```bash
   export DECO_ROOT=/root/rivermind-data/DeCo-3DGS
   export RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
   export PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python
   export NAG=/absolute/frozen/ramen/sai_nag_3.0.pt
   export OBS=/absolute/frozen/observations_level3.npz
   export SRC=/absolute/ramen/input
   export MODEL=/absolute/ramen/model
   export GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
   export OUT=/absolute/new/nonexistent/b123_20261028
   export DECO_CENTER=image DECO_TEXT=canon DECO_RENORM=0 DECO_SEL=gap
   bash /path/to/DeCo_B123_Patch/run_b123.sh
   ```

`DECO_CENTER/DECO_TEXT/DECO_RENORM/DECO_SEL` 必须由**已认证的、同一 NAG 的 DeCo 固定配置**决定；示例并不是声称 canon 是 C2 最强配置。不能为不同实验臂使用不同 DeCo 参数。B0 是必要的同 NAG 检查，不重复历史跨 NAG 对比。

## 必须先通过的验收

- 基础兼容：生成的 `eval_lerf_ovs_b123.py` 可以加载 DeCo C1/C2、`SemanticNAG`；B0 与原始 `eval_lerf_ovs.py` 在相同参数下输出逐查询完全一致（如已有同条件认证缓存可直接对照，不重跑）。
- 输入绑定：`K0 / B1 / B2` 的 `observation_sha256` 相同，三者绑定同一 `nag_sha256`，`B3` 直接复用 `B2.npz`，不是再生成银行。
- 来源一致性：观测 `region_ids` 必须与固定 NAG 的 root ID 空间对应；缺失根节点沿用原始 DeCo 分数。不能用相同根节点数量替代 SHA 校验。`K0` 的逐区域向量需与同输入人工加权均值严格一致；与已保存生产 root 的差异可能来自原 ROFA，必须分别报告。
- B1 证据核验：检查是否实际读取 `visible_gaussian_offsets/ids`，模式组别与成员 view ID 是否完整、`valid/prior` 是否正确；不要用仅仅张量 `[R,3,512]` 就宣称空间支持已实现。**本补丁只用几何支持参与分组，尚未实现按空间子区域渲染的 query routing。**
- B2 过滤核验：与 B1 保持同样的分组/几何参数，确认变化只来自模式内 ROFA；检查实际删掉多少观测。没有删除说明本数据设置未触发，不是 bug。
- B3 归因：B2 / B3 读取的银行 SHA 必须相同，唯一变化在 `combine(..., mode='fixed'|'adaptive')`，禁止改变 topk、候选下探、text mode。
- B3 零效应及边界：对单模式区域，B2 与 B3 查询分数应完全相同；多模式区域需保留权重分解以复核。
- 每组保存查询 IoU / top candidate IDs / predicted Gaussian count 和必要的输出 SHA；异常时只定位首个差异，不做 CLIP 特征统计与扫参。

## 注意

**最重要的限制**：补丁经过 Python 语法检查和小型 CPU 单元测试，尚未在用户服务器上完成真正的 DeCo 运行验证；运行结果不能预先声称提升。固定 `K=3`、`gamma=0.25`、`T=0.08` 是原型常数，不是最优参数。B2 是按模式执行已有 ROFA，不等于已经建立新的几何可靠性估计。不要为了获得正结果自动调整超参数。

先提交 `INTEGRATION_CHECK.md`，通过验证后一次性运行新四组 + 同 NAG B0 对照，最后提交 `B123_RESULTS.md`，停止。
