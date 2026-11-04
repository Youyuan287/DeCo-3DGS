#!/usr/bin/env python3
"""Build a one-page dashboard (self-contained HTML + inline SVG) from registry.csv.

Design follows the project's dataviz rules: one hue per single-series panel,
diverging blue/red only for signed values, direct labels, no dual axes,
light/dark from the same validated steps, and a table view behind a <details>.

Usage:  python reports/PAPER_EXPERIMENTS_20261104/build_dashboard.py
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dashboard.html"

BLUE = "#2a78d6"
RED = "#e34948"
GRAY_FILL = "#f0efec"


def load() -> dict[str, dict]:
    with open(HERE / "registry.csv", newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def pct(row: dict, key: str = "value") -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    x = float(v)
    if row["metric"] in {"mIoU", "BIoU", "AP", "rate", "median-IoU"}:
        x *= 100.0 if row["bench"].startswith("LERF") else 1.0
    return x


def d(v: float, nd: int = 2) -> str:
    s = f"{v:+.{nd}f}"
    return s.replace("-", "−")


def n(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


def esc(s: str) -> str:
    return html.escape(str(s))


def div_color(v: float, cap: float = 16.0) -> tuple[str, str]:
    t = min(1.0, abs(v) / cap)
    target = RED if v < 0 else BLUE
    base = tuple(int(GRAY_FILL[i:i + 2], 16) for i in (1, 3, 5))
    dst = tuple(int(target[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(round(a + (b - a) * t) for a, b in zip(base, dst))
    fill = "#%02x%02x%02x" % mix
    ink = "#ffffff" if t > 0.62 else "#0b0b0b"
    return fill, ink


# --------------------------------------------------------------------------
def panel_main(r: dict[str, dict]) -> str:
    rows = [
        ("LERF-OVS", pct(r["O-01"]), pct(r["O-04"]), "P1", "O-04"),
        ("LERF-MASK", pct(r["M-01"]), pct(r["M-02"]), "P3", "M-02"),
        ("ScanNet-10", pct(r["S-02"]), pct(r["S-06"]), "P5", "S-06"),
        ("3D-OVS", pct(r["D-01"]), None, "P8", None),
    ]
    x0, x1, vmax = 128, 508, 80.0
    X = lambda v: x0 + v / vmax * (x1 - x0)
    out = [f'<svg viewBox="0 0 540 250" role="img" aria-label="Baseline and best deployable module per benchmark">']
    for tick in (0, 20, 40, 60, 80):
        out.append(f'<line x1="{X(tick)}" y1="30" x2="{X(tick)}" y2="212" class="grid"/>')
        out.append(f'<text x="{X(tick)}" y="232" class="tick" text-anchor="middle">{tick}</text>')
    for i, (name, base, best, proto, rid) in enumerate(rows):
        y = 52 + i * 42
        out.append(f'<text x="116" y="{y + 4}" class="lab" text-anchor="end">{name}</text>')
        out.append(f'<text x="116" y="{y + 18}" class="sub" text-anchor="end">{proto}</text>')
        if base is not None:
            out.append(f'<circle cx="{X(base):.1f}" cy="{y}" r="5" class="dot-base"><title>{name} baseline {n(base)}</title></circle>')
        if best is not None:
            out.append(f'<line x1="{X(base):.1f}" y1="{y}" x2="{X(best):.1f}" y2="{y}" class="conn"/>')
            out.append(f'<circle cx="{X(best):.1f}" cy="{y}" r="7" class="dot-best"><title>{name} best {n(best)}</title></circle>')
            out.append(f'<text x="{X(best) + 14:.1f}" y="{y + 4}" class="val">{n(best)}</text>')
            out.append(f'<text x="{X(best) + 50:.1f}" y="{y + 4}" class="delta">{d(pct(r[rid], "delta"), 1)}</text>')
        else:
            out.append(f'<text x="{X(base) + 14:.1f}" y="{y + 4}" class="val">{n(base)}</text>')
            out.append(f'<text x="508" y="{y + 18}" class="note" text-anchor="end">no module</text>')
    out.append('</svg>')
    return "".join(out)


def panel_transfer(r: dict[str, dict]) -> str:
    ops = [
        ("C1 region-axis", {"OVS": "O-21", "MASK": "M-02", "SCAN": "S-03"}),
        ("C2 text lexicon", {"OVS": "O-23", "SCAN": "S-04"}),
        ("score-side zR", {"SCAN": "S-13"}),
        ("per-view evidence E0", {"OVS": "O-26", "SCAN": "S-14"}),
        ("decision certificate", {"OVS": "O-04", "MASK": "M-09", "SCAN": "S-35", "D3": "D-11"}),
    ]
    cols = [("OVS", "LERF-OVS"), ("MASK", "LERF-MASK"), ("SCAN", "ScanNet"), ("D3", "3D-OVS")]
    cw, ch, x0, y0 = 88, 42, 176, 64
    out = ['<svg viewBox="0 0 540 300" role="img" aria-label="Same operator across benchmarks: signed delta heatmap">']
    for j, (_, label) in enumerate(cols):
        out.append(f'<text x="{x0 + j * cw + cw / 2}" y="46" class="lab" text-anchor="middle">{label}</text>')
    for i, (name, cells) in enumerate(ops):
        y = y0 + i * ch
        out.append(f'<text x="164" y="{y + ch / 2 + 4}" class="lab" text-anchor="end">{name}</text>')
        for j, (key, _) in enumerate(cols):
            x = x0 + j * cw
            rid = cells.get(key)
            if rid is None:
                out.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" class="empty"/>')
                out.append(f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" class="na" text-anchor="middle">--</text>')
                continue
            row = r[rid]
            v = pct(row, "delta")
            fill, ink = div_color(v)
            out.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" fill="{fill}">'
                       f'<title>{name} / {row["bench"]}: {d(v, 2)} mIoU points ({rid})</title></rect>')
            out.append(f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" class="cell" fill="{ink}" text-anchor="middle">{d(v, 1)}</text>')
    out.append('<text x="20" y="288" class="sub">blue = gain, red = loss, clipped at ±16; numbers are true values</text>')
    out.append('</svg>')
    return "".join(out)


def panel_lerf(r: dict[str, dict]) -> str:
    ids = ["O-01", "O-02", "O-03", "O-04", "O-16"]
    names = {"O-01": "released", "O-02": "C_tf", "O-03": "mask rerank",
             "O-04": "auditor (pi=0.5)", "O-16": "option oracle"}
    x0, x1, lo, hi = 150, 470, 63.4, 70.0
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    base = pct(r["O-01"])
    out = ['<svg viewBox="0 0 540 250" role="img" aria-label="LERF-OVS ablation ladder">']
    for tick in (64, 66, 68, 70):
        out.append(f'<line x1="{X(tick)}" y1="26" x2="{X(tick)}" y2="206" class="grid"/>')
        out.append(f'<text x="{X(tick)}" y="226" class="tick" text-anchor="middle">{tick}</text>')
    out.append(f'<line x1="{X(base)}" y1="20" x2="{X(base)}" y2="206" class="baseline"/>')
    out.append(f'<text x="{X(base)}" y="14" class="sub" text-anchor="middle">anchor {n(base)}</text>')
    for i, rid in enumerate(ids):
        row = r[rid]
        v = pct(row)
        y = 44 + i * 36
        is_oracle = rid == "O-16"
        out.append(f'<text x="138" y="{y + 4}" class="lab" text-anchor="end">{names[rid]}</text>')
        out.append(f'<line x1="{X(base)}" y1="{y}" x2="{X(v)}" y2="{y}" class="conn"/>')
        if rid == "O-04":
            lo_ci = base + pct(row, "ci_low")
            hi_ci = base + pct(row, "ci_high")
            out.append(f'<line x1="{X(lo_ci)}" y1="{y}" x2="{X(hi_ci)}" y2="{y}" class="ci"/>')
            out.append(f'<line x1="{X(lo_ci)}" y1="{y - 5}" x2="{X(lo_ci)}" y2="{y + 5}" class="ci"/>')
            out.append(f'<line x1="{X(hi_ci)}" y1="{y - 5}" x2="{X(hi_ci)}" y2="{y + 5}" class="ci"/>')
        cls = "dot-oracle" if is_oracle else "dot-best"
        out.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="7" class="{cls}"><title>{names[rid]} {n(v)} (P1)</title></circle>')
        out.append(f'<text x="{X(v) + 14:.1f}" y="{y + 4}" class="val">{n(v)}</text>')
        out.append(f'<text x="{X(v) + 52:.1f}" y="{y + 4}" class="delta">{d(pct(row, "delta"))}</text>')
    out.append('<text x="150" y="242" class="sub">blue = deployable, hollow = oracle, whisker = 95% CI</text>')
    out.append('</svg>')
    return "".join(out)


def panel_headroom(r: dict[str, dict]) -> str:
    rows = [
        ("LERF-OVS", pct(r["O-16"]) - pct(r["O-04"]), "67.11 → 67.51"),
        ("LERF-MASK", pct(r["M-10"]) - pct(r["M-04"]), "66.60 → 72.99"),
        ("ScanNet-10", pct(r["S-30"]) - pct(r["S-40"]), "51.7 → 76.9 (K10)"),
        ("3D-OVS", pct(r["D-04"]) - pct(r["D-01"]), "70.05 → 95.94"),
    ]
    x0, x1, vmax = 150, 470, 28.0
    X = lambda v: x0 + v / vmax * (x1 - x0)
    out = ['<svg viewBox="0 0 540 230" role="img" aria-label="Remaining headroom per benchmark">']
    for tick in (0, 10, 20, 28):
        out.append(f'<line x1="{X(tick)}" y1="26" x2="{X(tick)}" y2="196" class="grid"/>')
        out.append(f'<text x="{X(tick)}" y="216" class="tick" text-anchor="middle">{tick}</text>')
    for i, (name, room, sub) in enumerate(rows):
        y = 46 + i * 40
        out.append(f'<text x="138" y="{y - 3}" class="lab" text-anchor="end">{name}</text>')
        out.append(f'<text x="138" y="{y + 11}" class="sub" text-anchor="end">{sub}</text>')
        w = max(2.0, X(room) - X(0))
        out.append(f'<rect x="{X(0):.1f}" y="{y - 7}" width="{w:.1f}" height="14" rx="4" class="bar">'
                   f'<title>{name}: {n(room)} mIoU points of headroom</title></rect>')
        out.append(f'<text x="{X(0) + w + 8:.1f}" y="{y + 4}" class="val">{n(room)}</text>')
    out.append('<text x="150" y="228" class="sub">headroom = oracle − current, same readout</text>')
    out.append('</svg>')
    return "".join(out)


def panel_swap(r: dict[str, dict]) -> str:
    rows = [
        ("certificate", [("deep", pct(r["O-10"], "delta")), ("found", pct(r["O-11"], "delta"))]),
        ("Grounded-SAM", [("waldo", pct(r["O-30"], "delta")), ("figurines", pct(r["O-31"], "delta"))]),
    ]
    vmax = 75.0
    X = lambda v: 260 + v / vmax * 190
    out = ['<svg viewBox="0 0 540 200" role="img" aria-label="Rescue-the-hard, hurt-the-easy pattern">']
    out.append(f'<line x1="260" y1="26" x2="260" y2="150" class="baseline"/>')
    for tick in (-60, -30, 0, 30, 60):
        out.append(f'<text x="{X(tick)}" y="172" class="tick" text-anchor="middle">{tick}</text>')
    for i, (name, pts) in enumerate(rows):
        y = 52 + i * 52
        out.append(f'<text x="60" y="{y + 4}" class="lab">{name}</text>')
        xs = [X(v) for _, v in pts]
        out.append(f'<line x1="{min(xs):.1f}" y1="{y}" x2="{max(xs):.1f}" y2="{y}" class="conn"/>')
        for (sub, v), x in zip(pts, xs):
            cls = "dot-best" if v > 0 else "dot-neg"
            out.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" class="{cls}"><title>{name} / {sub}: {d(v, 1)} points</title></circle>')
            anchor = "start" if v > 0 else "end"
            dx = 11 if v > 0 else -11
            out.append(f'<text x="{x + dx:.1f}" y="{y - 10}" class="val" text-anchor="{anchor}">{sub} {d(v, 1)}</text>')
    out.append('<text x="40" y="192" class="sub">rescue the weak subgroup, damage the strong one; the aggregate cancels</text>')
    out.append('</svg>')
    return "".join(out)


def table_view(r: dict[str, dict]) -> str:
    order = sorted(r.values(), key=lambda x: x["id"])
    head = ["ID", "Bench", "Tier", "Module", "P", "Base", "Value", "Delta", "Status", "R"]
    rows = []
    for x in order:
        b, v, dl = x["baseline"], x["value"], x["delta"]
        rows.append(
            "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in
                             [x["id"], x["bench"], x["tier"], x["module"], x["protocol"],
                              b, v, dl, x["status"], x["repro"]]) + "</tr>")
    return ("<details class='tv'><summary>表格视图：全部 " + str(len(order)) + " 条登记</summary>"
            "<div class='scroll'><table><thead><tr>" +
            "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" +
            "".join(rows) + "</tbody></table></div></details>")


CSS = """
:root{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --blue:#2a78d6; --red:#e34948; --ring:rgba(11,11,11,.10);
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --blue:#3987e5; --red:#e66767; --ring:rgba(255,255,255,.10);
  }
}
:root[data-theme="dark"]{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --blue:#3987e5; --red:#e66767; --ring:rgba(255,255,255,.10);
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}
.wrap{max-width:1140px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 4px}
.lede{color:var(--ink2);margin:0 0 22px;max-width:70ch}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.tile{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:26px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .s{font-size:12px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:16px 16px 10px}
.card h2{font-size:14px;margin:0 0 2px}
.card p{margin:0 0 6px;color:var(--ink2);font-size:12.5px}
.card.wide{grid-column:1 / -1}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.baseline{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:4 4}
.conn{stroke:var(--muted);stroke-width:2;stroke-linecap:round}
.ci{stroke:var(--ink2);stroke-width:2}
.dot-base{fill:var(--muted)}
.dot-best{fill:var(--blue)}
.dot-neg{fill:var(--red)}
.dot-oracle{fill:var(--surface-1);stroke:var(--blue);stroke-width:3}
.bar{fill:var(--blue);opacity:.9}
.empty{fill:var(--grid);opacity:.35}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.lab{fill:var(--ink);font-size:12px}
.sub{fill:var(--muted);font-size:11px}
.val{fill:var(--ink);font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.delta{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.cell{font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.na{font-size:11px;fill:var(--muted)}
.note{font-size:11px;fill:var(--muted)}
.tv{margin-top:22px;background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:12px 16px}
.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
.scroll{max-height:420px;overflow:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{position:sticky;top:0;background:var(--surface-1);color:var(--ink2);font-weight:600}
@media (max-width:820px){.grid2{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
"""


def main() -> None:
    r = load()
    tiles = [
        ("LERF-OVS", f'{pct(r["O-01"]):.2f} → {pct(r["O-04"]):.2f}', "+3.13, CI +1.14/+5.48"),
        ("ScanNet-10", f'{pct(r["S-02"]):.2f} → {pct(r["S-06"]):.2f}', "+17.76 pooled, closed-set"),
        ("LERF-MASK", f'{pct(r["M-01"]):.2f} → {pct(r["M-02"]):.2f}', "+3.12, weak (4 queries)"),
        ("3D-OVS", f'{pct(r["D-01"]):.2f}', "no deployable module"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in tiles)

    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Four 3D-OVS benchmarks at a glance</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>四个 3D-OVS 基准：一张图看完全部实验</h1>
<p class="lede">每个基准用自己的协议衡量，绝对数不可跨行比较。整体结论：一条轴边缘化原则、两个可部署实例、
两处精确零对照；同一个算子跨基准符号相反，正是原则的证据。</p>
<div class="tiles">{tile_html}</div>

<div class="grid2">
  <div class="card"><h2>1. 每个基准的基线 → 最好可部署结果</h2>
    <p>灰点基线，蓝点方法；3D-OVS 没有可部署模块，只有一个复现基线。</p>{panel_main(r)}</div>
  <div class="card"><h2>2. 同一算子跨基准：符号翻转</h2>
    <p>列轴算子只救多类读端，集合证书只救等大小替换。</p>{panel_transfer(r)}</div>
  <div class="card"><h2>3. LERF-OVS：从基线到审计器</h2>
    <p>每一步都是同一 208 条查询、同一评测器；审计器已用掉选项 oracle 的 88.6%。</p>{panel_lerf(r)}</div>
  <div class="card"><h2>4. 剩余余量在哪</h2>
    <p>LERF-OVS 只剩 0.4 分，ScanNet 与 3D-OVS 各剩约 25 分。</p>{panel_headroom(r)}</div>
  <div class="card wide"><h2>5. 所有干预的共同形态：救困难、伤容易</h2>
    <p>强子群被拖累的幅度几乎等于弱子群被救回的幅度，聚合起来正好抵消。</p>{panel_swap(r)}</div>
</div>

{table_view(r)}
</div></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()