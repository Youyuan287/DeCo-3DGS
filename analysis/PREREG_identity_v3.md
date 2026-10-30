# PREREG — LERF-OVS identity pilot v3 (frozen before computing v3 metrics)

v2 result (already known): R_F = argmax z(A)+z(Vc) passed hard rescue (0.636), cross-root
rescue (0.674), residualized AUC (0.852), but FAILED found retention (0.742 < 0.85): the
additive fusion is too eager to switch away from a correct top-1.

v3 hypothesis (fixed now): require BOTH independent evidence streams to agree before
switching. Two zero-parameter forms, both evaluated once:
  G_min = min(z(A), z(Vc))            (conservative AND-like fusion; pick argmax(G_min))
  R_cons = switch to c* only if argmax(A) == argmax(Vc) == c* != top-1 (else keep top-1)
Signals identical to v2 (A = z(area)+z(mean_vis); Vc = z(id_mean_c)+z(agree_c)); same families,
bootstrap (B=2000, seed 20261030, scene-stratified) and residualization protocol.

## v3 gate (strict; PASS requires ALL), primary rule R_cons
1. hard (55): rescue(R_cons) >= 0.50 AND CI lower > 0.30
2. found (124): retention(R_cons) >= 0.85 AND CI lower > 0.75
3. cross_root_hard (46): rescue(R_cons) >= 0.45 AND CI lower > 0.25
4. hard: AUC(G_min) residualized on log n_points, CI lower > 0.60
Secondary reports only: deep rescue(R_cons), hard rescue(G_min), retention(G_min).

## End-to-end (only if PASS)
- If R_cons switches, union the released gap prefix with the chosen challenger; render exact set.
- No parameters. Ship bar: scene-mean mIoU delta > +0.01 with paired stratified bootstrap CI lower > 0
  and no scene worse than -0.01, vs main 0.6396 and R1 rule 0.6525.

## Discipline / caveat
- This is the third decision rule examined on the same 4 scenes; each additional rule raises the
  chance of fitting noise. Any v3 PASS is provisional and must be validated on new scenes or Phase B.
- No threshold tuning after the fact.