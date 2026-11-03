# 10 · ScanNet 数据获取与 Blender/2DGS 格式转换（实测）

> 服务器：TZJ `root@hn01-ssh.gpuhome.cc -p 30946`；根目录 `$R=/root/rivermind-data/ReLaGS_Reproduction`
> 复现目标：ReLaGS 论文 Table 9（`ref/paper.txt:1239`）。本文件只记录**数据层**，真实数字均来自服务器日志（`logs/*.log`）。

## 1. 结论（先看这里）

| 项 | 结果 |
|---|---|
| `.sens` 下载 | 可跑通；**python urllib 只有 ~0.2 MB/s，curl 有 ~2.8 MB/s**（同一台机、同一时段，13 倍差），批量下载必须用 curl |
| 解帧 | 自写 py3 ParseNet 解析器，5578 帧的 scene0000_00 解帧 **7.7 s** |
| 位姿/内参 | 直接用 `.sens` 里的 c2w 与内参，**不需要 COLMAP**，也就没有 SfM 漂移 |
| Blender 化 | 刻意**不写 `camera_angle_x`，只写 `fl_x/fl_y`**；`c2w[:,1:3] *= -1`（y/z 轴翻转） |
| 点云初值 | 用 ScanNet `_vh_clean_2.ply` 的 mesh 顶点（带法向+RGB），不是 SfM 稀疏点 |
| 验收 | `transforms_train.json` 相机数 == 目录里 jpg 数；2DGS 训练 PSNR 见 11 号报告 |

## 2. 每场景清单（真实字节数/帧数）

| scene | `.sens` 字节 | 下载耗时 | 原始帧 | stride | 使用帧 | train/test | 点云初值 | 数据集占用 |
|---|---|---|---|---|---|---|---|---|
| scene0000_00 | 3,699,853,377 | 570 s (≈6.5 MB/s) | 5578 | 20 | 279 | 251/28 | 81,369 | 765 M |
| scene0062_00 | 373,729,315 | ≈90 s | 730 | 3 | 244 | 219/25 | 51,610 | 224 M→(含特征) |
| scene0070_00 | 未记录（python 下载，已删） | — | 1322 | 5 | 265 | 238/27 | 109,380 | 23 M |
| scene0097_00 | 227,122,404 | 70 s | 756 | 3 | 252 | 226/26 | 72,007 | — |
| scene0140_00 | 1,536,852,688 | 463 s | 4294 | 15 | 287 | 258/29 | 372,941 | — |

- `sha256(scene0000_00.sens) = 0fbc82546d6c0daf50bd2af4567d3c57c366765a0ed6944a1583aa790bd2f8cd`
- stride 规则：`stride = ceil(原始帧 / 300)`，保证每个场景 250–290 帧、10 场景量级一致（论文未给帧数，这是 deviation D2）。
- 分辨率 **648×484**（ScanNet 原图 1296×968 的一半），对应内参两套：
  - scene0000_00 / scene0062_00：`fx=584.8105 fy=583.5526 cx=323.1475 cy=244.9635`
  - scene0070_00 / 0097 / 0140：`fx=fy=585.0940 cx=323.875 cy=241.875`
  （ScanNet 两种相机型号，内参直接从 `.sens` 读，不写死。）

## 3. `.sens` 解析（`scripts/sens2frames.py`）

ParseNet 布局，逐字段读（py3，无 `scannetutils` 依赖）：

```
int32 version(4) | uint64 nameLen(8) | name | 4×float64[16] intrinsics/extrinsics(256)
int32 colorCompression, depthCompression(8) | uint32 colorW,colorH,depthW,depthH(16)
float32 depthShift(4) | uint64 numFrames(8) | numFrames × Frame
```

- 导出 `color/%06d.jpg`、`pose/%06d.txt`（**4×4 c2w**）、`intrinsic/%06d.txt`（按 648/1296 缩放后的 3×3）、`extract_meta.json`
- `--list_only` 只探帧数（用于自动定 stride）
- 帧计数踩过的坑：`numFrames` 前要跳过 **284 字节**（256+4+4+16+4），early 版本多读 4 字节，得到 `13794926641077551104` 这种垃圾帧数 → stride 算成 1。

## 4. Blender 转换（`scripts/make_blender.py`）

- 输出 `transforms_train.json`（251 相机）/ `transforms_test.json`（28 相机），每帧带 `file_path`、`transform_matrix`（4×4 c2w，`c2w[:3,1:3] *= -1`）
- 顶层相机参数：**`w,h,fl_x,fl_y,cx,cy`**，故意不写 `camera_angle_x`
  - 原因：2DGS 只认 `camera_angle_x`；而 ReLaGS 的 `readCamerasFromTransforms` 在 `load_img=False` 时读 `fl_x/fl_y`。为了**一份 transforms 两边共用**，给 2DGS 打了 `fov_y_direct` 补丁（见 11 号报告 §2）。
- `points3d.ply`：ScanNet `_vh_clean_2.ply` 的顶点（x,y,z,nx,ny,nz,red,green,blue），**不是**随机初始化；对齐 FlashSplat/OpenGaussian 的做法（它们另有 `scannet2blender.py`，固定 1296×968 + COLMAP→Blender 翻转；我们直接用 `.sens` 位姿，少一层误差）
- 与 OpenGaussian 版差异：OpenGaussian 固定内参、把 COLMAP 世界系翻到 Blender；我们保持 `.sens` 的 c2w，只做 y/z 翻转，因此**同一场景两种转换的世界系不同**，但 3D 评测是「最近 GT 点」赋标签，与绝对坐标系无关。

## 5. 验收证据

- `transforms_train.json` 相机数 == `color/` 帧数（例：251 == 251，日志 `Reading cameras from transforms: ... 251/251`，无 drop）
- 2DGS 用这套 transforms 训练 30k 步：scene0000_00 PSNR test **18.04**、scene0062_00 **21.29**（ScanNet 室内短序列 + 自动曝光，量级正常；见 11 号报告 §1）
- `points3d.ply` 顶点数 = mesh 顶点数（81369 / 51610 / 109380 / 72007 / 372941）

## 6. 磁盘账（50 G 盘）

- `.sens` 单场景 0.2–3.7 GB，**解帧后立刻删除**（已删 scene0000_00 / 0062 / 0070 / 0097 / 0140），否则 10 场景放不下
- 每场景常驻：图 30–60 MB + 语言特征 200–750 MB + 2DGS ply 300 MB（30k 未剪）+ 语义产物 ~50 MB
- 峰值策略：一次只保留「当前场景 + 下一个场景」的 `.sens`

## 7. 复现命令

```bash
bash scripts/batch_prep.sh                      # 8 个场景：curl 下载 + 解帧 + Blender 化
bash scripts/run_scene.sh scene0000_00 prep 0 300   # 单场景（stride 自动）
```