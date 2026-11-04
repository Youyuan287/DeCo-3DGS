# 论文矢量图

5 张图，每张有 `.svg`（可编辑母版）与 `.pdf`（矢量，直接进 LaTeX）。
数字全部来自 `../registry.csv`，由 `../export_figures.py` 生成；改数只需改 CSV 再跑一次。

```bash
python reports/PAPER_EXPERIMENTS_20261104/export_figures.py
```

| 文件 | 尺寸 (in) | 内容 | 对应登记行 |
|---|---|---|---|
| `F1_benchmarks` | 5.40 × 2.50 | 四基准：基线 → 最好可部署结果 | O-01/O-04, M-01/M-02, S-02/S-06, D-01 |
| `F2_operator_transfer` | 5.40 × 3.00 | 同一算子跨基准的符号翻转（热图） | O-21, M-02, S-03, O-23, S-04, S-13, O-26, S-14, O-04, M-09, S-35, D-11 |
| `F3_lerf_ladder` | 5.40 × 2.50 | LERF-OVS：released → C_tf → mask → auditor → oracle | O-01…O-04, O-16 |
| `F4_headroom` | 5.40 × 2.30 | 剩余余量：oracle − current | O-16/O-04, M-10/M-04, S-30/S-40, D-04/D-01 |
| `F5_sign_swap` | 5.40 × 2.00 | 救困难、伤容易：证书与 Grounded-SAM | O-10/O-11, O-30/O-31 |

## 放进论文

```latex
\usepackage{graphicx}
% 双栏通栏
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/F2_operator_transfer.pdf}
  \caption{The same operator flips sign across benchmarks ...}
  \label{fig:transfer}
\end{figure*}
```

单栏用 `\columnwidth`；F1/F4/F5 也可以缩到单栏，但 5.4in 宽的设计按 `\textwidth` 最清晰。

## 口径说明（写图注时必须带）

- **F1**：每行一个协议，绝对数不可跨行比较。3D-OVS 没有可部署模块，只有一个复现基线。
- **F2**：色阶裁到 ±16 分，格内数字是真值（LERF-MASK 证书为 −27.1）。证书在三个基准上
  都违反「等大小、同 query」前提，所以负号不是实现问题。
- **F3**：审计器的须是配对 bootstrap 的 95% CI（+1.14…+5.48），oracle 为空心点、不可当方法。
- **F4**：ScanNet 用 K10 逐场景均值（51.7 → 76.9），与 K19 pooled 的 58.55 不是同一口径。
- **F5**：两侧是不同协议的子群（mIoU 与 2D median IoU），只比较「同一干预内的符号互换」，
  不做跨组数值比较。

## 字体与配色

- 导出时字体固定为 Arial / Helvetica，PDF 内嵌；SVG 母版可直接在 Illustrator / Inkscape 改字。
- 单序列用蓝 `#2a78d6`，负值用红 `#e34948`，中点灰 `#f0efec`；浅色底 `#fcfcfb`。
  这组配色通过了色觉障碍与对比度校验（CVD ΔE 21.6、常规视觉 ΔE 32.3）。
- 图内已带直接数值标签，不依赖颜色单独传达信息。