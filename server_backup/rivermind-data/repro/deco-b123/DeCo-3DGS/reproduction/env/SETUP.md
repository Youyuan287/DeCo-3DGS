# 环境搭建

## 实测环境

| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10.x（venv） |
| PyTorch | 2.1.2 + cu118 |
| PyG | 2.3.0（`torch_geometric`） |
| CUDA toolkit | 11.8（conda 前缀布局） |
| GPU | 单卡，复现时占用约 8–12 GB 显存 |

完整依赖：`pip-freeze.txt`。

## 关键坑

1. **CUDA include 路径**：conda 布局的 toolkit 头文件在 `$CUDA_HOME/targets/x86_64-linux/include`，
   需加入 `CPATH`，否则 `cub/cub.cuh`、`thrust/*`、`cuda/std/*` 找不到。
2. **子模块包名**：`submodules/diff-surfel-rasterization-*/` 的 `setup.py` 里 `packages=[...]`
   必须与目录名一致，否则 `pip install -e` 后 import 失败（见 `patches/03`）。
3. **PyG 2.3.0 的 `Data.keys`**：是属性而非方法，上游按方法调用 → 需要 `patches/02`。
4. **fp16 特征**：LangSplat `.npy` 为 fp16，`merge_proj.py` 里矩阵乘会 dtype 冲突 → `patches/01`。
5. **二进制扩展**：`grid_graph`、`cp_d0_dist_cp` 若为 py38 编译产物，py310 下需替换或重编。
6. `**/*.egg-info` 若残留旧条目会导致 `import spt` 命中错误路径，建议清掉。

## 编译

```bash
export CUDA_HOME=/path/to/conda/envs/<env>
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
cd submodules/diff-surfel-rasterization-*   && pip install -e .
cd submodules/simple-knn                    && pip install -e .
cd submodules/grid_graph-*                 && pip install -e .
cd submodules/diff-gaussian-rasterization-* && pip install -e .   # 如需要
```
