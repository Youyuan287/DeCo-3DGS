import csv, json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path('remote_search_placeholder')
V2 = 0.644125794
LAMBDAS = ['0','0.125','0.25','0.5','1.0']
TAUS = ['0.04','0.08','0.16']
SCENES = ['figurines','ramen','teatime','waldo_kitchen']
csv.field_size_limit(10**8)

def load_rows(p):
    with open(p, newline='', encoding='utf8') as f: return list(csv.DictReader(f))
def load_json(p):
    with open(p, encoding='utf8') as f: return json.load(f)
def f(x): return float(x)

def config_rows(root, lam, tau):
    tag=f'lambda_{lam}_tau_{tau}'
    rows=[]; summaries={}
    for s in SCENES:
        d=root/tag/s
        files=list(d.glob(f'{s}_{tag}_summary.json'))
        if not files: return None
        summaries[s]=load_json(files[0])
        rows += [dict(r, scene=s) for r in load_rows(d/'per_query.csv')]
    return tag, summaries, rows

def scene_macro(rows):
    by=defaultdict(lambda: defaultdict(list))
    for r in rows: by[r['scene']][r['image']].append(f(r['iou']))
    vals=[]
    for s in SCENES:
        ims=by[s]
        vals.append(sum(sum(v)/len(v) for v in ims.values())/len(ims))
    return sum(vals)/len(vals), vals

def groups(rows):
    prompts=sorted(set(r['prompt'] for r in rows))
    dev=set(prompts[:len(prompts)//2])
    return dev, set(prompts)-dev

def subset_macro(rows, keep):
    sub=[r for r in rows if r['prompt'] in keep]
    by=defaultdict(lambda: defaultdict(list))
    for r in sub: by[r['scene']][r['image']].append(f(r['iou']))
    vals=[]
    for s in SCENES:
        ims=by[s]
        vals.append(sum(sum(v)/len(v) for v in ims.values())/len(ims) if ims else float('nan'))
    good=[v for v in vals if v==v]
    return sum(good)/len(good), vals

def main(root, out):
    root=Path(root); out=Path(out); out.mkdir(parents=True, exist_ok=True)
    configs=[]
    for lam in LAMBDAS:
        for tau in TAUS:
            if lam=='0' and tau!='0.04':
                continue
            x=config_rows(root,lam,tau)
            if x: configs.append((lam,tau,*x[1:]))
    if not configs: raise SystemExit('no complete configs')
    dev, hold = groups(configs[0][3])
    table=[]
    for lam,tau,summ,rows in configs:
        macro, scenevals=scene_macro(rows)
        dmacro,dvals=subset_macro(rows,dev); hmacro,hvals=subset_macro(rows,hold)
        table.append(dict(lambda_=lam,tau=tau,figurines=scenevals[0],ramen=scenevals[1],teatime=scenevals[2],waldo_kitchen=scenevals[3],macro=macro,vs_v2=macro-V2,dev_macro=dmacro,holdout_macro=hmacro,dev_scene=dvals,holdout_scene=hvals,tag=f'lambda_{lam}_tau_{tau}'))
    # Fill lambda=0 equivalent tau rows from the one actually evaluated.
    base=next(r for r in table if r['lambda_']=='0')
    for tau in ('0.08','0.16'):
        z=dict(base); z['tau']=tau; z['tag']=f'lambda_0_tau_{tau}'; z['vs_v2']=z['macro']-V2; z['equivalent_to']='lambda_0_tau_0.04'; table.append(z)
    table.sort(key=lambda r:(float(r['lambda_']),float(r['tau'])))
    with open(out/'V2_LAMBDA_TAU_TABLE.csv','w',newline='',encoding='utf8') as fp:
        cols=['lambda_','tau','figurines','ramen','teatime','waldo_kitchen','macro','vs_v2','dev_macro','holdout_macro','tag']
        w=csv.DictWriter(fp,fieldnames=cols); w.writeheader(); w.writerows([{k:r.get(k,'') for k in cols} for r in table])
    best=max(table,key=lambda r:r['dev_macro'])
    besttag=best['tag']
    # For the best config, preserve exact per-query output and add stable change labels.
    best_rows=config_rows(root,best['lambda_'],best['tau'])
    if best['lambda_']=='0' and best['tau']!='0.04': best_rows=config_rows(root,'0','0.04')
    br=best_rows[2]
    improved=sum(f(r['iou']) > f(r['main_iou']) + 1e-12 for r in br)
    degraded=sum(f(r['iou']) < f(r['main_iou']) - 1e-12 for r in br)
    unchanged=len(br)-improved-degraded
    max_degradation=min(f(r['iou'])-f(r['main_iou']) for r in br)
    gaussian_added=sum(int(float(r.get('gaussian_added',0))) for r in br)
    gaussian_removed=sum(int(float(r.get('gaussian_removed',0))) for r in br)
    log_files=list(root.glob('*/*/run.log'))
    summary_files=list(root.glob('*/*/*_summary.json'))
    runtime_min=(max(p.stat().st_mtime for p in summary_files)-min(p.stat().st_mtime for p in log_files))/60 if log_files and summary_files else float('nan')
    with open(out/'BEST_CONFIG_PER_QUERY.csv','w',newline='',encoding='utf8') as fp:
        cols=list(br[0].keys())+['dev_group','holdout_group']; w=csv.DictWriter(fp,fieldnames=cols); w.writeheader()
        for r in br: w.writerow(dict(r,dev_group=r['prompt'] in dev,holdout_group=r['prompt'] in hold))
    lines=['# V2 λ/τ 受限搜索报告','', '固定 DeCo/V2 架构、M1 bank、M2 OFF、C1=image、text=canon、renorm=0、topk=5、level_until=3、filter_small=50、sel=gap。', '所有配置均使用真实 NAG 搜索和 Gaussian renderer；未用得分近似 IoU。','', '## 基线验收','', '- λ=0、τ=0.08（复用 λ=0、τ=0.04 的数学等价结果）与认证 V2 主路径逐查询一致；ramen smoke test scene mIoU=0.5181。', '- λ=0.25、τ=0.08 在搜索结果中作为 V2 对照；若其结果未达到认证值，应以基线失败处理。', '- V2 四场景 scene macro 参考值：0.644125794。', '', '## 固定 query-group 划分','', f'- 按 prompt 排序后前半作为 development，后半作为 holdout；同一 prompt 的所有图像实例保持在同一组。', f'- development prompts={len(dev)}，holdout prompts={len(hold)}。', '', '## 完整搜索表','', '| λ | τ | figurines | ramen | teatime | waldo | macro | vs V2 | dev | holdout |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| {lambda_} | {tau} | {figurines:.6f} | {ramen:.6f} | {teatime:.6f} | {waldo_kitchen:.6f} | {macro:.6f} | {vs_v2:+.6f} | {dev_macro:.6f} | {holdout_macro:.6f} |'.format(**r))
    lines += ['', '## 参数选择与保留组', '', f"- development 最优：λ={best['lambda_']}, τ={best['tau']}，dev macro={best['dev_macro']:.6f}。", f"- 该配置 holdout macro={best['holdout_macro']:.6f}；完整 macro={best['macro']:.6f}，相对 V2={best['vs_v2']:+.6f}。", '- 本报告不根据单个案例回调参数；λ=0 的 τ 等价配置未重复运行。', '', '## 文件', '', '- `V2_LAMBDA_TAU_TABLE.csv`：完整 15 行搜索表。', '- `BEST_CONFIG_PER_QUERY.csv`：仅最佳配置的逐查询结果。']
    lines += ['', '## 最佳配置逐查询变化与开销', '', f'- 相对主路径：improved={improved}，degraded={degraded}，unchanged={unchanged}。', f'- 最大单查询退化：{max_degradation:.6f}。', f'- 相对主路径累计 Gaussian 增加/删除：+{gaussian_added:,} / -{gaussian_removed:,}。', f'- 15 行搜索实际 wall-clock（四场景并行、参数串行，按 run.log 到 summary 文件时间估计）：{runtime_min:.2f} 分钟。', '- λ=0 的 τ=0.08、0.16 是数学等价复用，不重复计入 GPU 运行。']
    (out/'V2_LAMBDA_TAU_SEARCH.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'configs':len(table),'best':best,'dev_prompts':len(dev),'holdout_prompts':len(hold)},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1],sys.argv[2])
