"""Apply the paired certificate to LERF-Mask: should we add the remaining candidates?

A = gap prefix, B = full candidate list. Testimonies come from held-out training
views (rendered candidate mask -> SAM superpixels -> CLIP feature -> text sim).
Half-sample majority over views decides whether B replaces A.
IoU is read from the precomputed prefix curves, so nothing is re-rendered.
"""
import csv, json, re, sys
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

def setev(ranks):
    out = {}
    for r in ranks:
        for v, s in wit.get((p, r), {}).items():
            if s > out.get(v, -1e9):
                out[v] = s
    return out

decisions = {}
for prompt in sorted({x["prompt"] for x in pairs}):
    K = max(x["K"] for x in pairs if x["prompt"] == prompt)
    gapk = next(x["gapk"] for x in pairs if x["prompt"] == prompt)
    A, B = list(range(1, gapk + 1)), list(range(1, K + 1))
    p = prompt
    ea, eb = setev(A), setev(B)
    vs = sorted(set(ea) & set(eb))
    if len(vs) < 5:
        decisions[prompt] = (False, 0.0, len(vs))
        continue
    d = np.array([eb[v] - ea[v] for v in vs])
    k = max(2, len(vs) // 2)
    means = np.array([d[rng.choice(len(vs), k, replace=False)].mean() for _ in range(NB)])
    prob = float((means > 0).mean())
    decisions[prompt] = (prob >= PI, prob, len(vs))

base = np.mean([x["curve"][x["gapk"] - 1] for x in pairs])
cert = np.mean([x["curve"][x["K"] - 1] if decisions[x["prompt"]][0] else x["curve"][x["gapk"] - 1]
                for x in pairs])
oracle = np.mean([max(x["curve"]) for x in pairs])
print("%-6s %-5s %-6s %-6s %s" % ("prompt", "gapk", "p", "fires", "curve"))
for p in sorted(decisions):
    f, prob, nv = decisions[p]
    print("  %-26s gapk=%d p=%.2f fire=%s views=%d" % (p, next(x["gapk"] for x in pairs if x["prompt"] == p),
                                                       prob, f, nv))
print("\ngap baseline %.4f | certificate %.4f | oracle %.4f | n=%d" % (base, cert, oracle, len(pairs)))
