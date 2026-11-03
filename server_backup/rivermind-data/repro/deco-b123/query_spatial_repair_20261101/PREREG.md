# Query-conditioned local spatial repair pilot

Frozen 2026-11-01 before results. Four LERF-OVS scenes, 208 queries. No tuning after results.

All remote writes under /root/rivermind-data/repro/deco-b123/query_spatial_repair_20261101 including temp/cache/screen sockets.

A: reproduce existing gateE from official released NAG, same preds/evidence/segidentity. Per-query rendered IoU must match reference within 1e-6 or abort.

Evidence: at most24 training cameras, sorted image_name then equally spaced, excluding every annotated evaluation camera. Existing semantic levels1,2,3 and original CLIP features. Gaussian support uses existing validated projection xyz@R+T, render_point weight>1e-4, z>1e-4. SAM masks at least50 pixels. No query-view evidence in repair.

For each source view, masks eligible for local editing must share at least20 visible Gaussians with baseline G and cover at least10% of both baseline-visible support and their own visible Gaussian support. These thresholds are fixed feasibility choices, not validated optimal settings. Select one mask per view across levels:
B geometry: maximum visible Gaussian-support IoU with G.
C text: maximum original CLIP cosine to query among eligible masks.
D shuffled: same G/eligible masks but text from next distinct prompt in sorted scene query list (wrap around). No GT mask or IoU enters selection.

Each selected source view casts a positive membership vote inside selected mask and a negative vote on other visible Gaussians. For Gaussians observed in at least2 selected views, assign membership by vote fraction>=0.5; preserve original membership for fewer observations. This is a minimal voting pilot, not a claimed novel final algorithm. C can both remove and add support; positive votes and denominators logged. No new foundation model or feature extraction.

Report rendered macro (query mean within image, image mean within scene, four-scene mean), scene/family deltas, wins/losses, changed Gaussians, number selected views, and 2000 scene-stratified image-cluster paired bootstrap CI (seed20261030). Family frozen from old candidate CSV (best IoU<.3 recall, otherwise rank1 found/rank2/deep). Report diagnostic max(A,B,C) oracle separately from deployable outputs; it is not a general spatial upper bound.

Go criterion for this pilot: C improves macro vs A by>=.01 with bootstrap lower>0, no scene drops>.01, found drop<=.005, and C exceeds B and D on macro. Failure reported without threshold or lambda sweep. Success motivates further novelty/generalization evaluation, not proof of CVPR novelty.
