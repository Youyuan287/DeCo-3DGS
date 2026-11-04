#!/usr/bin/env python3
"""Generate the paper tables (LaTeX + Markdown previews) from registry.csv.

Usage:  python reports/PAPER_EXPERIMENTS_20261104/build_tables.py

Design rules enforced here:
  * a table never mixes protocols within a row;
  * superseded rows are excluded from every table except T7/gaps;
  * Tier-2 (oracle) and Tier-3 (negative) rows never enter T1;
  * every value is rendered as a percentage when its metric is a fraction.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
TABLES.mkdir(exist_ok=True)

FRACTION_METRICS = {"mIoU", "BIoU", "AP", "rate", "median-IoU", "AUC"}
BENCH_ORDER = ["LERF-OVS", "LERF-MASK", "ScanNet", "3D-OVS"]

SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex(s: str) -> str:
    return "".join(SPECIAL.get(ch, ch) for ch in str(s))


def load_registry() -> list[dict]:
    with open(HERE / "registry.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def load_protocols() -> dict[str, dict]:
    with open(HERE / "protocols.csv", newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def scale(row: dict) -> float:
    """LERF rows store fractions, ScanNet/3D-OVS rows store percentages."""
    if row["metric"] not in FRACTION_METRICS:
        return 1.0
    return 100.0 if row["bench"].startswith("LERF") else 1.0


def num(row: dict, key: str = "value") -> str:
    value = row.get(key, "")
    if value in ("", None):
        return "--"
    x = float(value)
    if row["metric"] in FRACTION_METRICS:
        x *= scale(row)
        return f"{x:.2f}" if scale(row) == 1.0 else f"{x:.1f}"
    return f"{x:.2f}"


def delta(row: dict) -> str:
    value = row.get("delta", "")
    if value in ("", None):
        return "--"
    x = float(value) * scale(row)
    return f"{x:+.2f}".replace("-", "$-$")


def ci(row: dict) -> str:
    lo, hi = row.get("ci_low", ""), row.get("ci_high", "")
    if lo in ("", None) or hi in ("", None):
        return ""
    f = scale(row)
    return f"[{float(lo) * f:+.2f}, {float(hi) * f:+.2f}]".replace("-", "$-$")


def status_cell(row: dict) -> str:
    s = row["status"]
    return {"2d-only": "2D", "identity": "null", "weak": "weak"}.get(s, s)


def emit(name: str, caption: str, label: str, colspec: str,
         header: list[str], rows: list[list[str]], note: str = "") -> None:
    if not rows:
        raise SystemExit(f"{name}: no rows")
    width = len(header)
    for r in rows:
        if len(r) != width:
            raise SystemExit(f"{name}: row has {len(r)} cells, expected {width}: {r}")

    body = "\n".join(" & ".join(c for c in r) + r" \\" for r in rows)
    head = " & ".join(header) + r" \\"
    note_tex = f"\n\\par\\smallskip\\footnotesize {note}\n" if note else "\n"
    out = (
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{head}\n\\midrule\n{body}\n\\bottomrule\n"
        f"\\end{{tabular}}{note_tex}\\end{{table*}}\n"
    )
    (TABLES / f"{name}.tex").write_text(out, encoding="utf-8")

    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * width) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            c.replace("$-$", "-").replace("\\_", "_").replace("\\%", "%") for c in r) + " |")
    if note:
        md += ["", note.replace("\\", "")]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def t1_main(reg: list[dict]) -> None:
    rows = []
    for bench in BENCH_ORDER:
        sub = [r for r in reg if r["bench"] == bench and r["tier"] == "T1"
               and r["status"] != "superseded"]
        sub.sort(key=lambda r: (r["status"] != "anchor", r["id"]))
        first = True
        for r in sub:
            rows.append([
                bench if first else "",
                r["protocol"], tex(r["module"]), r["metric"],
                num(r, "baseline"), num(r), delta(r), ci(r),
                r["n"], r["repro"],
            ])
            first = False
    emit("T1_main_results",
         "Deployable modules per benchmark. Each row is measured under its own protocol; "
         "absolute values are not comparable across benchmarks.",
         "tab:main", "llllrrrccl",
         ["Bench.", "P", "Module", "Metric", "Base", "Value", "$\\Delta$", "95\\% CI", "n", "R"],
         rows,
         "P = protocol card in Table~\\ref{tab:prot}. $\\Delta$ and CI are in the metric's own units "
         "(mIoU points). R = reproduction level: R0 recomputable from this repo, R1 rerun on an "
         "existing cache, R2 needs re-encoding, R3 artefact missing.")


def t2_transfer(reg: list[dict]) -> None:
    by_id = {r["id"]: r for r in reg}
    ops = [
        ("C1 region-axis centring", {"LERF-OVS": "O-21", "LERF-MASK": "M-02", "ScanNet": "S-03"}),
        ("C2 text lexicon, $\\tau$-LSE", {"LERF-OVS": "O-23", "ScanNet": "S-04"}),
        ("score-side $zR$", {"ScanNet": "S-13"}),
        ("per-view evidence E0", {"LERF-OVS": "O-26", "ScanNet": "S-14"}),
        ("decision certificate", {"LERF-OVS": "O-04", "LERF-MASK": "M-09",
                                  "ScanNet": "S-35", "3D-OVS": "D-11"}),
    ]
    rows = []
    for name, cells in ops:
        out = [name]
        for bench in BENCH_ORDER:
            rid = cells.get(bench)
            if rid is None:
                out.append("--")
            else:
                r = by_id[rid]
                out.append(delta(r))
        rows.append(out)
    emit("T2_operator_transfer",
         "The same operator under each benchmark's own protocol. Sign flips are the point: "
         "a column-axis operator helps only multi-class readouts, a set-level certificate helps "
         "only equal-size replacement on an open-vocabulary query axis.",
         "tab:transfer", "lcccc",
         ["Operator", "LERF-OVS", "LERF-MASK", "ScanNet-10", "3D-OVS"], rows,
         "Cells are $\\Delta$ in mIoU points. Columns: O-16/P1, M-04/P4, S-13/S-35/P5, D-11/P8. "
         "The certificate is evaluated on nesting in LERF-MASK, on the class axis in ScanNet and on "
         "the superpoint set in 3D-OVS; all three violate its equal-size, same-query premise. "
         "Grounded-SAM (O-28) is a 2D observation path with no 3D lift and is reported in Table~\\ref{tab:ceil}.")


def t3_lerf(reg: list[dict]) -> None:
    ids = ["O-01", "O-02", "O-03", "O-04", "O-05", "O-06", "O-07", "O-08", "O-09",
           "O-12", "O-13", "O-14", "O-15", "O-16", "O-21", "O-22", "O-23", "O-24", "O-26", "O-27",
           "O-32", "O-33", "O-34"]
    by_id = {r["id"]: r for r in reg}
    rows = []
    for rid in ids:
        r = by_id[rid]
        rows.append([r["id"], tex(r["module"]), num(r, "baseline"),
                     num(r), delta(r), ci(r), status_cell(r), r["repro"], tex(r["note"])])
    emit("T3_ablation_lerf",
         "LERF-OVS ablation on the official readout (P1). Baseline is 0.6401 unless stated.",
         "tab:abl-lerf", "llrrrcclp{5.2cm}",
         ["ID", "Ablation", "Base", "Value", "$\\Delta$", "95\\% CI", "Status", "R", "Note"],
         rows,
         "All rows use the same 208 queries and the same evaluator. O-24 (0.6760) is a diagnostic, "
         "not a method: its module-C branch reads a frozen LOSO-GBM predicted IoU.")


def t4_scannet(reg: list[dict]) -> None:
    sub = [r for r in reg if r["bench"] == "ScanNet" and r["tier"] in ("T1", "T3")]
    order = {"T1": 0, "T3": 1}
    sub.sort(key=lambda r: (r["id"] != "S-02", order[r["tier"]], r["id"]))
    rows = []
    for r in sub:
        if r["status"] == "superseded" and r["id"] not in ("S-13", "S-15", "S-17", "S-32"):
            continue
        rows.append([r["id"], tex(r["module"]), r["protocol"],
                     num(r, "baseline"), num(r), delta(r), status_cell(r), r["repro"]])
    emit("T4_ablation_scannet",
         "ScanNet-10 ablation. Pooled (P5) and macro (P6) rows are marked and must not be mixed.",
         "tab:abl-scannet", "lllrrrcl",
         ["ID", "Ablation", "P", "Base", "Value", "$\\Delta$", "Status", "R"], rows,
         "S-02/S-03--S-08 are the shipped-field anchor (R3: the 366-cluster field is missing). "
         "S-09--S-12 are the rebuilt 386-cluster field (R1) and sit 1.3--3.4 points below the anchor. "
         "S-13/S-15/S-17 are superseded: score-side $zR$, wrong text space, and the early "
         "per-view-mean implementation of C1. S-37/S-38/S-39 are summarised from the ledger.")


def t5_ceilings(reg: list[dict]) -> None:
    ids = ["O-16", "O-17", "O-18", "O-19", "O-20", "O-28", "O-29", "O-30", "O-31",
           "M-10", "M-11", "M-12", "M-13", "S-01", "S-30", "S-31", "S-33",
           "D-02", "D-03", "D-04", "D-05", "D-06", "D-07", "D-09", "D-12"]
    by_id = {r["id"]: r for r in reg}
    rows = []
    for rid in ids:
        r = by_id[rid]
        value = num(r)
        rows.append([r["id"], r["bench"], tex(r["module"]), r["metric"], value, tex(r["note"])])
    emit("T5_ceilings",
         "Diagnostic ceilings and reference points. These are not deployable methods.",
         "tab:ceil", "llllrp{5.6cm}",
         ["ID", "Bench.", "Diagnostic", "Metric", "Value", "What it bounds"], rows,
         "An oracle may select with ground truth or may pick the best option post hoc; neither is a "
         "method. O-28 is the only row measured on 2D masks without a 3D lift.")


def t6_negatives(reg: list[dict]) -> None:
    sub = [r for r in reg if r["status"] in ("negative", "superseded")]
    sub.sort(key=lambda r: (BENCH_ORDER.index(r["bench"]), r["id"]))
    rows = []
    for r in sub:
        rows.append([r["id"], r["bench"], r["protocol"], tex(r["module"]),
                     delta(r), status_cell(r), tex(r["note"])])
    emit("T6_negatives",
         "Falsified directions and superseded numbers, kept because they constrain the claim.",
         "tab:neg", "llllrlp{6.0cm}",
         ["ID", "Bench.", "P", "Attempt", "$\\Delta$", "Status", "Why it failed / why it is superseded"],
         rows,
         "Negative results are reported under the protocol they were run on; a negative on one "
         "protocol is not evidence against the same operator under another.")


def t7_protocols(prot: dict[str, dict]) -> None:
    rows = []
    for pid in sorted(prot):
        p = prot[pid]
        rows.append([pid, p["benchmark"], tex(p["name"]), tex(p["readout"]),
                     tex(p["aggregation"]), tex(p["anchor"]), tex(p["notes"])])
    emit("T7_protocol_genealogy",
         "Protocol cards. A number is comparable only to another number with the same P.",
         "tab:prot", "llp{2.4cm}p{3.4cm}p{3.2cm}pp{4.0cm}",
         ["P", "Bench.", "Name", "Readout", "Aggregation", "Anchor", "Comparability rule"],
         rows)


def main() -> None:
    reg = load_registry()
    prot = load_protocols()
    t1_main(reg)
    t2_transfer(reg)
    t3_lerf(reg)
    t4_scannet(reg)
    t5_ceilings(reg)
    t6_negatives(reg)
    t7_protocols(prot)
    for f in sorted(TABLES.iterdir()):
        print(f"{f.name:32} {f.stat().st_size:6} B")


if __name__ == "__main__":
    main()