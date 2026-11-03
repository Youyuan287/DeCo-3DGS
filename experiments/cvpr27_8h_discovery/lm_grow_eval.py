"""Non-monotone multi-view rule for set SIZE on LERF-Mask.

The max-support certificate fails on nested sets because E_v(B) >= E_v(A) always.
Test the increment instead: candidate k+1 enters only if its per-view evidence is
not worse than the current support in a majority of views (paired test on
delta_v = a_v(k+1) - E_v(S)). This is non-monotone and uses the same testimony.
"""
import csv, re, sys
import numpy as np

wpath, logpath = sys.argv[1], sys.argv[2]
PI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
NB = 300
rng = np.random.default_rng(20261103)

wit = {}
for r in csv.DictReader(open(wpath)):
    wit.setdefault((r["prompt"], int(r["rank"])), {})[r["view"]] = float(r["sim"])

pairs = []
for line in open(logpath):
    m = re.match(r"\s*CURVE (\S+) (\S+) (\S+) K=(\d+) GAPK=(\d+) (.*)", line)
    if m:
        pairs.append(dict(view=m.group(2), prompt=m.group(3).replace("_", " "),
                          K=int(m.group(4)), gapk=int(m.group(5)),
                          curve=[float(x) for x in m.group(6).split()]))

def ev_of(ranks, prompt):
    out = {}
    for r in ranks:
        for v, s in wit.get((prompt, r), {}).items():
            if s > out.get(v, -1e9):
                out[v] = s
    return out

def grow(prompt, K, pi):
    S = [1]
    for k in range(2, K + 1):
        e = ev_of(S, prompt)
        a = wit.get((prompt, k), {})
        vs = sorted(set(e) & set(a))
        if len(vs) < 5:
            break
        d = np.array([a[v] - e[v] for v in vs])
        kk = max(2, len(vs) // 2)
        means = np.array([d[rng.choice(len(vs), kk, replace=False)].mean() for _ in range(NB)])
        if (means > 0).mean() < pi:
            break
        S.append(k)
    return S

chosen = {}
for p in sorted({x["prompt"] for x in pairs}):
    K = max(x["K"] for x in pairs if x["prompt"] == p)
    s = grow(p, K, PI)
    chosen[p] = len(s)
    print("  %-26s K=%d -> k=%d" % (p, K, len(s)))

def macro(idx_key):
    return float(np.mean([x["curve"][min(x[idx_key], x["K"]) - 1] for x in pairs]))

base = macro("gapk"); growm = np.mean([x["curve"][min(chosen[x["prompt"]], x["K"]) - 1] for x in pairs])
oracle = float(np.mean([max(x["curve"]) for x in pairs]))
print("\ngap %.4f | grow-rule %.4f | all %.4f | k=1 %.4f | oracle %.4f | n=%d" % (
    base, growm, float(np.mean([x["curve"][-1] for x in pairs])),
    float(np.mean([x["curve"][0] for x in pairs])), oracle, len(pairs)))
