# PREREG — Query-only template reliability gate (frozen before computing)

Constraint: no scene vocabulary, no other query categories. Only the current query and four fixed
generic templates: `{}`, `a photo of a {}`, `the {}`, `a {} in the scene`.

Signal: on the released top-5 pool, each template gives a top-1 candidate; reliability = max
agreement count / 4. Gate: `qgate = gateE if reliability < 0.75 else module C` (use mask evidence
only when the text is uncertain; otherwise conservative).

Primary metric: B0 four-scene macro; report per-family reliability and gateE/qgate. Success:
found not worse than gateE by >0.005 and macro >= gateE (descriptive; this is a small test).
No tuning; sensitivity 0.5/1.0 reported only.

Rationale: if template agreement identifies deep/uncertain queries, gating can protect found and
let mask evidence fix deep. If reliability is uniformly high, the signal is vacuous and the
text-side line closes.