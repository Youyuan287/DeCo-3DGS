"""Read-only CPU verification of the FJ HF-only output; no model imports."""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics

ROOT=Path('/root/rivermind-data')
RUN=ROOT/'ReLaGS_run_20261023'
REF=ROOT/'ReLaGS_Reproduction'
EXPECTED={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22}

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

current=rows(RUN/'lerf_query_metrics.csv')
old=rows(REF/'results/lerf_query_metrics.csv')
key=lambda r:(r['scene'],r['image'],r['prompt'])
assert len(current)==208 and Counter(r['scene'] for r in current)==EXPECTED
assert len({key(r) for r in current})==208
old_map={key(r):r for r in old}
assert set(map(key,current))==set(old_map)
assert all(float(r['iou'])==float(old_map[key(r)]['iou']) for r in current)
summaries={}
for scene in EXPECTED:
    summary=json.loads((RUN/(scene+'_summary.json')).read_text())
    assert summary['n_queries']==EXPECTED[scene]
    assert summary['protocol']==dict(topk=5,level_until=3,filter_small=50,remove_small=True)
    frames=defaultdict(list)
    for r in current:
        if r['scene']==scene:frames[r['image']].append(float(r['iou']))
    rounded=statistics.mean(statistics.mean(v) for v in frames.values())
    assert abs(rounded-summary['miou'])<1e-6
    summaries[scene]=dict(miou=summary['miou'],n_queries=summary['n_queries'],n_images=len(frames))
files=[RUN/'lerf_query_metrics.csv',REF/'scripts/eval_lerf_relags.py',REF/'scripts/run_hf_only.sh',REF/'ReLaGS/nag_data.py',REF/'ReLaGS/merge_proj.py',ROOT/'ReLaGS-1023/reproduction/scripts/eval_lerf_relags.py']
files += [REF/'ref/relags_hf'/s/'sai_nag.pt' for s in EXPECTED]
result=dict(status='HF_OUTPUT_VERIFIED',verified_date='2026-10-24',query_count=208,
    historical_query_iou_values_equal=True,scene_metrics=summaries,
    macro_miou=statistics.mean(s['miou'] for s in summaries.values()),
    source_binding='HF model/evaluator reference paths verified now; original launch command not present in run directory, so current hash is not proof of which source was executed.',
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
    environment_dependency=str((REF/'venv/bin/python').resolve()),
    evaluator_executable_ast_equal=ast.dump(ast.parse((REF/'scripts/eval_lerf_relags.py').read_text()))==ast.dump(ast.parse((ROOT/'ReLaGS-1023/reproduction/scripts/eval_lerf_relags.py').read_text())),
    no_gpu_or_renderer_executed=True)
print(json.dumps(result,ensure_ascii=False,indent=2))
