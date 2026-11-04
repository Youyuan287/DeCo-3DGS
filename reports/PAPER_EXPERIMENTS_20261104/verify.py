#!/usr/bin/env python3
"""Recompute every R0 number in registry.csv from local artefacts.

The point of this script is to make the tables auditable: if a number in the
registry cannot be regenerated from a file in this repository, it is not R0
and must say so.

Run from anywhere:  python reports/PAPER_EXPERIMENTS_20261104/verify.py
Exit code 0 iff every assertion passes.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.csv"
PROTOCOLS = HERE / "protocols.csv"

RESULTS = REPO / "analysis" / "results"
GATE_TF = RESULTS / "20261101" / "cnone_gate_tf"
EVIDENCE = RESULTS / "20261101" / "cnone_ev"
WITNESS = RESULTS / "20261101" / "witness"
BASELINE = RESULTS / "20261031" / "baselines" / "relags"

failures: list[str] = []
passes: list[str] = []


def check(name: str, expected: float, actual: float, tol: float) -> None:
    ok = abs(expected - actual) <= tol
    (passes if ok else failures).append(
        f"{'PASS' if ok else 'FAIL'}  {name:<46} expected {expected:.4f}  got {actual:.4f}"
    )


def read_rows(paths) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def nested_macro(rows: list[dict], key: str) -> float:
    """query mean within image -> image mean within scene -> mean over scenes.

    This is the P1 aggregation. A flat mean over rows is a different (and for
    LERF-OVS, lower) number; see protocols.csv P1.
    """
    per_image: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_image[(r["scene"], r["image"])].append(float(r[key]))
    per_scene: dict[str, list[float]] = defaultdict(list)
    for (scene, _image), values in per_image.items():
        per_scene[scene].append(sum(values) / len(values))
    return sum(sum(v) / len(v) for v in per_scene.values()) / len(per_scene)


# --------------------------------------------------------------------------
# 1. registry / protocols structural checks
# --------------------------------------------------------------------------
def structural_checks() -> None:
    with open(REGISTRY, newline="") as fh:
        reg = list(csv.DictReader(fh))
    with open(PROTOCOLS, newline="") as fh:
        prot = {r["id"] for r in csv.DictReader(fh)}

    ids = [r["id"] for r in reg]
    assert len(ids) == len(set(ids)), "duplicate ids in registry.csv"

    allowed_tier = {"T1", "T2", "T3"}
    allowed_repro = {"R0", "R1", "R2", "R3"}
    for r in reg:
        assert r["tier"] in allowed_tier, f"{r['id']}: bad tier {r['tier']}"
        assert r["repro"] in allowed_repro, f"{r['id']}: bad repro {r['repro']}"
        assert r["protocol"] in prot, f"{r['id']}: unknown protocol {r['protocol']}"
        assert r["bench"].startswith(("LERF", "ScanNet", "3D-OVS")), r["id"]
    passes.append(f"PASS  registry/protocols structure ({len(reg)} rows, {len(prot)} protocols)")

    assessed = {r["id"] for r in reg if r["repro"] == "R0"}
    covered = {"O-01", "O-02", "O-03", "O-04", "O-10", "O-11", "O-16",
               "O-28", "O-29", "O-30", "O-31", "O-32", "O-33", "O-34",
               "M-13", "S-33"}
    missing = assessed - covered
    assert not missing, f"R0 rows without an assertion: {sorted(missing)}"
    passes.append(f"PASS  every R0 row has an assertion ({len(assessed)} rows)")


# --------------------------------------------------------------------------
# 2. LERF-OVS: released readout and the three options
# --------------------------------------------------------------------------
def lerf_readouts() -> list[dict]:
    rows = read_rows(sorted(GATE_TF.glob("*_cmaskgate_none.csv")))
    check("O-01 released main_iou (P1 nested)", 0.6401, nested_macro(rows, "main_iou"), 5e-5)
    check("O-02 C_tf c_iou", 0.6572, nested_macro(rows, "c_iou"), 5e-5)
    check("O-03 mask rerank mask_iou", 0.6582, nested_macro(rows, "mask_iou"), 5e-5)
    for r in rows:
        r["_opt"] = max(float(r["main_iou"]), float(r["c_iou"]), float(r["mask_iou"]))
    check("O-16 option oracle", 0.6751, nested_macro(rows, "_opt"), 5e-5)
    return rows


def lerf_baseline_anchor() -> None:
    rows = read_rows(sorted(BASELINE.glob("*.csv")))
    check("O-01 independent baseline CSV (P1 nested)", 0.6401, nested_macro(rows, "iou"), 5e-5)
    # The failure mode this guards against: a flat mean is NOT the protocol.
    flat = sum(float(r["iou"]) for r in rows) / len(rows)
    if abs(flat - 0.6401) < 1e-4:
        failures.append("FAIL  aggregation guard: flat mean equalled the nested macro")
    else:
        passes.append(f"PASS  aggregation guard: flat mean {flat:.4f} != nested 0.6401")


# --------------------------------------------------------------------------
# 3. LERF-OVS: certificate (re-run the frozen CPU script)
# --------------------------------------------------------------------------
def certificate() -> None:
    cmd = [
        sys.executable,
        str(REPO / "analysis" / "lerf_witness_certificate.py"),
        str(GATE_TF), str(GATE_TF), str(EVIDENCE), str(WITNESS),
        "none", "0.5", "sc",
    ]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout

    def macro(tag: str) -> float:
        m = re.search(rf"MACRO {tag}\s+([0-9.]+)", out)
        assert m, f"cannot parse MACRO {tag}:\n{out}"
        return float(m.group(1))

    check("O-04 auditor pi=0.5", 0.6711, macro("cert"), 5e-5)
    check("O-04 auditor vs released", 0.6401, macro("main"), 5e-5)

    fam = {}
    for line in out.splitlines():
        m = re.match(r"(found|rank2|deep|recall)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
        if m:
            fam[m.group(1)] = (float(m.group(2)), float(m.group(5)))
    assert "deep" in fam and "found" in fam, f"cannot parse family table:\n{out}"
    check("O-10 deep family (released -> auditor)", 0.6515, fam["deep"][1], 5e-5)
    check("O-10 deep family baseline", 0.2914, fam["deep"][0], 5e-5)
    check("O-11 found family (released -> auditor)", 0.8275, fam["found"][1], 5e-5)
    check("O-11 found family baseline", 0.8316, fam["found"][0], 5e-5)

    m = re.search(r"cert-main mean ([+-][0-9.]+) CI95 \[([+-][0-9.]+), ([+-][0-9.]+)\]", out)
    assert m, f"cannot parse bootstrap CI:\n{out}"
    check("O-04 bootstrap mean delta", 0.0313, float(m.group(1)), 5e-5)
    check("O-04 CI low", 0.0114, float(m.group(2)), 5e-4)
    check("O-04 CI high", 0.0548, float(m.group(3)), 5e-4)


# --------------------------------------------------------------------------
# 4. LERF-OVS 2D grounding probe
# --------------------------------------------------------------------------
def grounding() -> None:
    g = json.load(open(REPO / "results" / "lerfovs_gsam_20261104" / "lerfovs_gsam.json"))
    check("O-28 CLIP median IoU (208)", 0.803, g["clip_med"], 5e-5)
    check("O-28 Grounded-SAM median IoU", 0.722, g["gsam_med"], 5e-5)
    check("O-29 CLIP >=0.5", 0.577, g["clip_ge50"], 5e-5)
    check("O-29 Grounded-SAM >=0.5", 0.572, g["gsam_ge50"], 5e-5)

    rows = read_rows([REPO / "results" / "lerfovs_gsam_20261104" / "lerfovs_gsam.csv"])
    for scene, clip, gsam, n in (("waldo_kitchen", 0.158, 0.788, 22),
                                 ("figurines", 0.916, 0.333, 56)):
        sub = [r for r in rows if r["scene"] == scene]
        assert len(sub) == n, f"{scene}: {len(sub)} rows, expected {n}"
        med = lambda k: statistics.median(float(r[k]) for r in sub)
        check(f"O-30/31 {scene} CLIP median", clip, med("clip_iou"), 5e-4)
        check(f"O-30/31 {scene} GSAM median", gsam, med("gsam_iou"), 5e-4)

    m = json.load(open(REPO / "results" / "grounded_sam_20261104" / "grounded_sam.json"))
    check("M-13 LERF-MASK probe CLIP-bad median", 0.925, m["clip_bad_med_gsam_samsel"], 5e-5)


def visibility() -> None:
    rows = read_rows(sorted((REPO / "results" / "lerf_visibility_20261104").glob("*_official.csv")))
    check("O-01/32 base official readout", 0.6401, nested_macro(rows, "base_iou"), 5e-5)
    check("O-32 V1 visible-prefix prune (identity)", 0.6401, nested_macro(rows, "V1_iou"), 5e-5)
    check("O-33 V2 visibility reorder", 0.6368, nested_macro(rows, "V2_iou"), 5e-5)
    check("O-34 V3 empty-render guard", 0.6524, nested_macro(rows, "V3_iou"), 5e-5)
    changed = [r for r in rows if abs(float(r["V3_iou"]) - float(r["base_iou"])) > 1e-12]
    assert len(changed) == 5, f"V3 changed {len(changed)} rows, expected 5"
    assert {r["scene"] for r in changed} == {"ramen"}, "V3 must only touch ramen"
    passes.append("PASS  O-34 V3 touches exactly 5 rows, all ramen")


# --------------------------------------------------------------------------
# 5. ScanNet rebuilt-field ablation + single-query AP
# --------------------------------------------------------------------------
def scannet() -> None:
    d = json.load(open(REPO / "results" / "scannet_c1c2_ablation_20261103" / "c1c2_ablation.json"))
    s = d["summary"]
    check("S-09 rebuilt baseline pooled", 41.05, s["C1=off_C2=off_r0"]["pooled_miou"], 5e-3)
    check("S-09 rebuilt baseline macro", 38.87, s["C1=off_C2=off_r0"]["scene_mean"], 5e-3)
    check("S-10 rebuilt C1+C2 pooled", 56.05, s["C1=on_C2=on_r0"]["pooled_miou"], 5e-3)
    check("S-11 rebuilt C1+C2 macro", 49.89, s["C1=on_C2=on_r0"]["scene_mean"], 5e-3)
    check("S-12 rebuilt C1+C2 renorm1 macro", 51.10, s["C1=on_C2=on_r1"]["scene_mean"], 5e-3)

    a = json.load(open(REPO / "results" / "scannet_singlequery_20261104" / "scannet_singlequery_ap.json"))
    check("S-33 single-query AP raw", 0.5797, a["ap_raw"], 5e-5)
    check("S-33 single-query AP minus a_g", 0.5568, a["ap_minus_a"], 5e-5)
    check("S-33 single-query AP minus b_c (exact null)", 0.5797, a["ap_minus_b"], 5e-5)


def main() -> int:
    structural_checks()
    lerf_baseline_anchor()
    lerf_readouts()
    certificate()
    grounding()
    visibility()
    scannet()

    print("\n".join(passes))
    if failures:
        print()
        print("\n".join(failures))
        print(f"\n{len(failures)} assertion(s) FAILED")
        return 1
    print(f"\nAll {len(passes)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())