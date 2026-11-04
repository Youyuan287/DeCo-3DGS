#!/usr/bin/env python3
"""Export each dashboard panel as a standalone SVG and a vector PDF.

SVG is the editable master (Illustrator / Inkscape / Keynote). PDF is produced
by Chrome's print-to-pdf, so text and shapes stay vector.

Usage:  python reports/PAPER_EXPERIMENTS_20261104/export_figures.py
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import build_dashboard as bd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
if not CHROME.exists():
    CHROME = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

STYLE = """
text{font-family:Arial,"Helvetica Neue",Helvetica,sans-serif}
.grid{stroke:#e1e0d9;stroke-width:1}
.baseline{stroke:#c3c2b7;stroke-width:1.5;stroke-dasharray:4 4}
.conn{stroke:#898781;stroke-width:2;stroke-linecap:round}
.ci{stroke:#52514e;stroke-width:2}
.dot-base{fill:#898781}
.dot-best{fill:#2a78d6}
.dot-neg{fill:#e34948}
.dot-oracle{fill:#fcfcfb;stroke:#2a78d6;stroke-width:3}
.bar{fill:#2a78d6;opacity:.9}
.empty{fill:#e1e0d9;opacity:.35}
.tick{fill:#898781;font-size:11px;font-variant-numeric:tabular-nums}
.lab{fill:#0b0b0b;font-size:12px}
.sub{fill:#898781;font-size:11px}
.val{fill:#0b0b0b;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.delta{fill:#52514e;font-size:11px;font-variant-numeric:tabular-nums}
.cell{font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.na{font-size:11px;fill:#898781}
.note{font-size:11px;fill:#898781}
"""


def standalone(svg: str, name: str) -> Path:
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
    w, h = m.group(1), m.group(2)
    svg = svg.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ', 1)
    svg = svg.replace(">", f"><style>{STYLE}</style>"
                           f'<rect width="{w}" height="{h}" fill="#fcfcfb"/>', 1)
    path = FIG / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def to_pdf(svg_path: Path, w_in: float, h_in: float) -> Path:
    body = svg_path.read_text(encoding="utf-8")
    html = (f'<!doctype html><meta charset="utf-8">'
            f'<style>@page{{size:{w_in:.2f}in {h_in:.2f}in;margin:0}}'
            f'html,body{{margin:0;padding:0}}'
            f'svg{{display:block;width:{w_in:.2f}in;height:{h_in:.2f}in}}</style>'
            f'{body}')
    page = svg_path.with_suffix(".html")
    page.write_text(html, encoding="utf-8")
    pdf = svg_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([
            str(CHROME), "--headless=new", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={tmp}", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}", page.resolve().as_uri(),
        ], check=True, capture_output=True)
    page.unlink()
    return pdf


def main() -> None:
    r = bd.load()
    panels = [
        ("F1_benchmarks", bd.panel_main(r), 5.40, 2.50),
        ("F2_operator_transfer", bd.panel_transfer(r), 5.40, 3.00),
        ("F3_lerf_ladder", bd.panel_lerf(r), 5.40, 2.50),
        ("F4_headroom", bd.panel_headroom(r), 5.40, 2.30),
        ("F5_sign_swap", bd.panel_swap(r), 5.40, 2.00),
    ]
    for name, svg, w, h in panels:
        sp = standalone(svg, name)
        pdf = to_pdf(sp, w, h)
        print(f"{name}: {sp.stat().st_size:6} B svg | {pdf.stat().st_size:7} B pdf")


if __name__ == "__main__":
    main()