# Early root top-3 retrieval — frozen before execution

2026-11-01. Question: Does moving existing root multi-view evidence before the root shortlist improve selection of existing masks?

A: exact existing raw-root top10, >50-point filtering, root top5, original leaf promotion, raw candidate ordering, original maximum-gap multi-candidate selection.
B: top3 observation/text cosine averaged per root over the existing fixed observations_level3 bank; choose five >50-point roots globally using this score BEFORE leaf promotion. Then exactly the same leaf promotion, raw candidate ordering, gap selection, and exact renderer as A. A root needs at least three original observations; if fewer than five roots qualify globally, fill unused slots in raw-root score order. Report all fallback slots.

No new encoding, masks, geometric repair, feature centering, consistency score, learned selector, λ/μ, threshold sweep or prompt expansion. The legacy NAG preprocessing is inherited unchanged to reproduce A. The top3 operation is the original root id_top3; it does not acquire new views or claim independence of observations.

208 existing query-view tasks from four scenes. Current released ReLaGS/DeCo Gaussian model and m1_sas_20261029 observation bank. THGS .81 oracle and its Gaussian IDs/cache are not mixed into this run. Frozen gateE .6760107 is an external same-checkpoint anchor; A is raw-gap main, not gateE.

Primary: B-A IoU using image mean within scene then four-scene mean. Mechanism secondary: candidate-pool single-mask oracle B-A, root shortlist changes, losses of original good candidates. Also report query mean, scene query mean, family effects, and scene-stratified paired image-cluster bootstrap B2000 seed20261101. A must match archived gap5 per query to 1e-6. GT read only after A/B selections, used only for evaluation and candidate-pool oracle.

Continue this specific early-evidence policy only if B-A is positive with bootstrap lower >0 and no scene decreases by >.01. A positive candidate-pool oracle without a final IoU gain supports improved retrieval but does not establish a working selector. Failure does not prove information is missing or that all mask selectors fail. No tuning after viewing results.

All new server writes, logs, temporary and cache directories must be under /root/rivermind-data/repro/deco-b123/early_root_top3_20261101. Existing model caches are read-only/offline.
