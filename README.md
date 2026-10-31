# DeCo-3DGS

**DeCo** (*De-Collapse*) removes the scene-wide common mode from the language representation of an
open-vocabulary 3D Gaussian Splatting pipeline. Two training-free operators, no new parameters, no
labels, < 10 ms per scene.

| benchmark | readout | baseline | + DeCo |
|---|---|---|---|
| ScanNet-10 K19 (dataset-pooled) | 18-class argmax | 40.79 | **58.55** (+17.76) |
| ScanNet-10 K19 (macro) | 18-class argmax | 39.94 | **53.30** (+13.36) |
| LERF-Mask (Gaussian Grouping) | single-query retrieval | 0.6800 | 0.7112 |
| LERF-OVS | single-query retrieval | 0.6401 | 0.6393 |

The first two rows are the claim; the last two are a scope check, and `deco/PROTOCOLS.md` explains
why image-side mean removal is provably inert for a single-query readout.

* implementation: [`deco/`](deco/) — `deco_ops.py`, the three harnesses, both lexicons
* evaluation conventions: [`deco/PROTOCOLS.md`](deco/PROTOCOLS.md) (class divisor is 18, not 19)
* full tables and negative results: [`deco/RESULTS.md`](deco/RESULTS.md)

### Repository map

| path | what |
|---|---|
| `deco/` | the two DeCo operators, the three evaluation harnesses, both lexicons — everything behind the numbers above |
| `reproduction/` | third-party reproduction of the upstream ReLaGS pipeline on LERF-OVS (64.01 vs 64.40) and of the ScanNet-10 baseline (40.79 vs 41.26) |
| `reproduction/3dovs/` | **extension**: 3D-OVS end-to-end on 5 scenes (2DGS 30k → SAM+CLIP → superpoints → `merge_proj` → eval), 7 arms per scene. DeCo publishes no 3D-OVS number, so this is new territory, not a table reproduction |
| `reproduction/audits/figurines_feature_encoding_20261031/` | can the released semantic field be rebuilt from scratch? Yes — per-cluster cosine **0.9999** against the release, mIoU 66.10 vs 63.29, plus the symlink path-escape bug that had hidden it |
| `scene/`, `gaussian_renderer/`, `nag_data.py`, `scripts/`, `ext/`, `submodules/` | upstream ReLaGS code, unmodified except for the patches listed in `reproduction/patches/` |

This repository is built on the ReLaGS codebase; its README follows unchanged.

---

# ReLaGS: Relational Language Gaussian Splatting

**CVPR 2025**

[Yaxu Xie](https://scholar.google.com/citations?user=3ZKuh9EAAAAJ&hl=en)\* · [Abdalla Arafa](https://abdallaarafa.github.io/)\* · [Alireza Javanmardi](https://scholar.google.com/citations?user=SR_4n3kAAAAJ&hl=en) · [Christen Millerdurai](https://chris10m.github.io/) · [Jia Cheng Hu](https://scholar.google.com/citations?user=KxUF6BUAAAAJ&hl=it) · [Shaoxiang Wang](https://shaoxiang777.github.io/) · [Alain Pagani](https://www.dfki.de/en/web/about-us/employee/person/alpa02) · [Didier Stricker](https://www.dfki.de/en/web/about-us/employee/person/dist01)

<sup>\*Equal contribution</sup>

**¹ German Research Center for Artificial Intelligence (DFKI) · ² RPTU University Kaiserslautern-Landau · ³ University of Modena and Reggio Emilia**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2603.17605)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://dfki-av.github.io/ReLaGS/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/dfki-av/ReLaGS)

---

## 😊 TL;DR

ReLaGS enables **hierarchical 3D scene understanding with language guidance** for Gaussian-splatted scenes. It provides open-vocabulary semantic segmentation, multi-level object hierarchies, and optional 3D scene graph generation—**all without scene-specific training**.

> **Note:** Evaluation scripts will be published soon.

> **🔁 Third-party reproduction (2026-10).** A complete LERF-OVS four-scene reproduction is
> included in [`reproduction/`](reproduction/), filling in the evaluation harness that is not
> shipped here: **Mean 64.01 mIoU vs. the paper's 64.40** (figurines 63.29 / ramen 51.15 /
> teatime 80.97 / waldo_kitchen 60.62), on HuggingFace-only inputs.
> Start at [`reproduction/README.md`](reproduction/README.md) (protocol + how to run) and
> [`reproduction/FINAL_REPORT.md`](reproduction/FINAL_REPORT.md) (full report, provenance,
> protocol sensitivity, limitations).
>
> **🔁 Third-party extension (2026-10).** Two follow-ups live next to it:
> (a) [`reproduction/audits/figurines_feature_encoding_20261031/`](reproduction/audits/figurines_feature_encoding_20261031/) —
> the released `sai_nag.pt` **is** rebuildable: re-encoding all 299 figurines frames with the
> shipped encoder and re-running the five-stage pipeline reproduces the author's per-cluster
> features at **cosine 0.9999** (mIoU 66.10 vs 63.29 on release weights); it also documents the
> symlink path-escape bug that made an earlier rebuild silently read LangSplat's OpenAI-CLIP
> features instead of the freshly encoded ones;
> (b) [`reproduction/3dovs/`](reproduction/3dovs/) — 3D-OVS from raw data, 5 scenes × 7 arms
> (base mean 70.05), where the two DeCo operators turn out to be a wash on this readout, plus an
> oracle control (95.20) that localises the 22-point literature gap to cluster *selection*, not
> training or the metric.

---

![Teaser](assets/images/teaser.jpg)

***Relational Language Gaussian Splatting.*** We build a platform with multi-hierarchical language Gaussian field and open-vocabulary 3D scene graph, to support various tasks such as object selection via click, open vocabulary 3D object segmentation across semantic granularity, spatial relationship reasoning between objects and querying object with relation-guidance.

---

## 🧭 Framework Overview

ReLaGS combines three core components for unified 3D scene understanding:

1. **Language-Distilled Gaussian Fields**: Multi-level hierarchical semantic representations on 2D Gaussian splatting with language embeddings (scenes → objects → parts)

2. **Multi-View Language Alignment**: Robust aggregation of Vision Language Model features (CLIP/DINO) from multiple views into accurate 3D semantic embeddings via ray-tracing

3. **Graph-Based Relationago wigggggl Reasoning**: Two complementary approaches for understanding inter-object relationships:
   - **GNN-based** (default): Graph Neural Network trained on Scan3R predicts 27 relationship types, scene-agnostic and no API keys needed
   - **VLM+SoM** (alternative): ChatGPT-powered Set-of-Marks prompting for flexible, custom relationship definitions

For technical details, see the [paper](https://arxiv.org/pdf/2603.17605).

---

## Installation

**Requirements:** Python 3.10.13, PyTorch 2.2.0, CUDA 11.8, GPU with >20GB VRAM

**Step 1:** Clone and create conda environment
```bash
git clone https://github.com/dfki-av/ReLaGS.git
cd ReLaGS
conda env create -f environment.yml
conda activate ReLaGS

# install additional dependencies
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python scripts/setup_dependencies.py build_ext
```

**Step 3:** Verify installation
```bash
python -c "from gaussian_renderer import render; print('✓ ReLaGS installed successfully')"
```

---

## Datasets

### Download Example Data

We provide LERF example scenes with pre-trained checkpoints and extracted semantics on HuggingFace:

- [Download from HuggingFace](https://huggingface.co/dfki-av/ReLaGS)

Original datasets:
- [LERF-OVS](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing)
- [3DOVS](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing)
- [ScanNet](https://github.com/ScanNet/ScanNet)
- [ScanNet++](https://github.com/aliaksandrsiarohin/scannet-plus)

### Prepare Your Own Data (COLMAP Format)

For custom datasets, ReLaGS requires multi-view RGB images and camera poses via COLMAP:

```bash
mkdir -p my_scene/images
cp /path/to/your/images/*.jpg my_scene/images/

# Preprocess with COLMAP
python convert.py --source_path my_scene --camera OPENCV --resize
```

Output structure:
```
my_scene/
├── images/
├── distorted/sparse/0/  # COLMAP reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images_resized/
```

### Extract 2D Semantic Features

To use language-guided semantics (same preprocessing as LangSplat), extract 2D CLIP features and SAM segmentation masks:

```bash
python scripts/image_encoding.py --source_path my_scene
```

Optional arguments:
- `--save_dir` – output directory (default: `language_features/`)
- `--resolution` – rescale images to specific width (default: auto-downscale if >1080p)
- `--sam_ckpt_path` – SAM checkpoint path (default: `ckpts/sam_vit_h_4b8939.pth`)

Outputs:
```
my_scene/language_features/
├── {image_name}_f.npy  # CLIP embeddings (300 features × 512-dim)
└── {image_name}_s.npy  # SAM segmentation maps
```

> **Note:** Download SAM checkpoint from [Meta Research](https://github.com/facebookresearch/segment-anything) if not present.

### Optional - Extract 2D Scene Graphs with SoM

To extract relational structure from images before lifting to 3D, optionally generate 2D scene graphs using ChatGPT Set-of-Marks prompting:

1. Generate SoM annotations with GPT
2. Lift 2D annotations to 3D scene graph

See [scene_graph/som/README.md](scene_graph/som/README.md) for complete setup, usage instructions, and advanced options including custom relationship types and embedding conversion.

---

## Training

### Step 1: Prepare 2D Gaussian Splatting Scene

Follow the official [2D Gaussian Splatting](https://github.com/plumerai/2d-gaussian-splatting) implementation to train a scene:

```bash
cd /path/to/2d-gaussian-splatting
python train.py \
  --source_path /path/to/colmap/preprocessed/scene \
  --model_path /path/to/output \
  --iterations 30000
```

### Step 2: Run ReLaGS Pipeline

The easiest way is to use the provided shell scripts:

```bash
# For geometry-only reconstruction
bash scripts/run_lerf.sh configs/lerf.yml /path/to/scene

# For full pipeline with scene graphs
bash scripts/run_scannet.sh configs/scannet.yml /path/to/scene
```

Or run key steps manually:

```bash
# Initial render and pruning
python max_weight_pruning.py -m output_dir

# Hierarchical partitioning (2 passes)
python sp_partition.py -m output_dir -a
python merge_proj.py -m output_dir
python sp_partition.py -m output_dir -k

# Edge reweighting and final mesh
python graph_weight.py -m output_dir --config configs/scannet.yml
```
---

## Relational Reasoning

ReLaGS supports two approaches for 3D scene graph generation:

### GNN-Based (Default)
Predicts 27 relationship types using a Graph Neural Network trained on Scan3R. Scene-agnostic and requires no API keys.

```bash
python predict_3d_scene_graph.py --model_path output_dir --root_pred output_dir
```

**Outputs:** `jina_edges_predicted.pt`, `combined_edges_predicted.pt`

**Requires:** Download `graph_transformer_512_80.pth` from [HuggingFace](https://huggingface.co/dfki-av/ReLaGS) and place in `ckpts/`

(Optional) To train GNN on 3DSSG dataset, see [scene_graph/data_anno/README.md](scene_graph/data_anno/README.md) for training data preprocessing, then run:

```bash
python scene_graph/train_gnn.py --output_dir ckpts
```

### VLM+Set-of-Marks (Alternative)
ChatGPT-powered approach for flexible, custom relationship definitions. Requires OpenAI API key.

See [scene_graph/som/README.md](scene_graph/som/README.md) for setup and usage.

---

## Citation

If you find ReLaGS useful for your research, please cite:

```bibtex
@inproceedings{xiearafa2026relags,
  title     = {ReLaGS: Relational Language Gaussian Splatting},
  author    = {Xie, Yaxu and Arafa, Abdalla and Javanmardi, Alireza and Millerdurai, Christen and Hu, Jia Cheng and Wang, Shaoxiang and Pagani, Alain and Stricker, Didier},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This work has been partially funded by the EU projects [dAIEDGE](https://daiedge-project.eu/) (GA Nr 101120726) and [LUMINOUS](https://luminous-project.eu/) (GA Nr 101135724).

We thank the contributors to [THGS](https://arxiv.org/abs/2504.13153), [3D Gaussian Splatting](https://repo.polimi.it/xie/3dgs), [2D Gaussian Splatting](https://github.com/plumerai/2d-gaussian-splatting), and [Segment Anything](https://github.com/facebookresearch/segment-anything) for their foundational work.

---

## License

This project is licensed under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.
