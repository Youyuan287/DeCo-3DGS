import json, os
B = '/root/rivermind-data/deco/results'
ARMS = ['base', 'c1', 'c2', 'c1c2']
LIT = {'bed': 94.4, 'bench': 91.2, 'lawn': 94.8, 'room': 86.9, 'sofa': 92.5, 'mean': 95.4}
def get(scene, arm):
    p = os.path.join(B, '%s_%s_summary.json' % (scene, arm))
    if not os.path.exists(p):
        return None
    return json.load(open(p))['miou'] * 100
hdr = '%-7s %8s %8s %8s %8s %10s' % ('scene', 'base', 'c1', 'c2', 'c1c2', 'ReLaGS')
print(hdr); print('-' * len(hdr))
cols = {a: [] for a in ARMS}
for sc in ['bed', 'bench', 'lawn', 'room', 'sofa']:
    vals = {a: get(sc, a) for a in ARMS}
    cells = []
    for a in ARMS:
        v = vals[a]
        cells.append('%8s' % ('--' if v is None else '%.2f' % v))
        if v is not None:
            cols[a].append(v)
    print('%-7s %s %10.1f' % (sc, ' '.join(cells), LIT[sc]))
print('-' * len(hdr))
cells = []
for a in ARMS:
    cells.append('%8s' % ('--' if not cols[a] else '%.2f' % (sum(cols[a]) / len(cols[a]))))
print('%-7s %s %10.1f' % ('mean', ' '.join(cells), LIT['mean']))
print()
for a in ARMS:
    print('%-5s n_scenes=%d' % (a, len(cols[a])))
