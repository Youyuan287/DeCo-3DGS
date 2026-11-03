import csv, json, hashlib, sys
from pathlib import Path
from collections import defaultdict
csv.field_size_limit(10**8)
SCENES=['figurines','ramen','teatime','waldo_kitchen']
ARMS=['E0','E1','E2','E3']
def rows(p):
    with open(p,newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def summary(root,arm,s): return json.load(open(root/arm/s/f'{s}_{arm}_summary.json',encoding='utf8'))
def f(x): return float(x)
def macro(root,arm): return sum(summary(root,arm,s)['miou'] for s in SCENES)/4
def split(rs):
    ps=sorted(set(r['prompt'] for r in rs)); n=len(ps)//2
    return set(ps[:n]),set(ps[n:])
def subset(rs,keep):
    by=defaultdict(lambda:defaultdict(list))
    for r in rs:
        if r['prompt'] in keep: by[r['scene']][r['image']].append(f(r['iou']))
    vals=[]
    for s in SCENES:
        ims=by[s]; vals.append(sum(sum(v)/len(v) for v in ims.values())/len(ims) if ims else float('nan'))
    z=[v for v in vals if v==v]; return sum(z)/len(z),vals
def main(root,out):
    root=Path(root); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    allrows={a:[] for a in ARMS}
    for a in ARMS:
        for s in SCENES: allrows[a]+= [dict(r,scene=s) for r in rows(root/a/s/'per_query.csv')]
    rmap={a:{(r['scene'],r['image'],r['prompt']):r for r in rs} for a,rs in allrows.items()}
    dev,hold=split(allrows['E1'])
    table=[]
    for a in ARMS:
        mm=macro(root,a); dm,_=subset(allrows[a],dev); hm,_=subset(allrows[a],hold)
        table.append({'arm':a,'figurines':summary(root,a,'figurines')['miou'],'ramen':summary(root,a,'ramen')['miou'],'teatime':summary(root,a,'teatime')['miou'],'waldo_kitchen':summary(root,a,'waldo_kitchen')['miou'],'macro':mm,'vs_e1':mm-macro(root,'E1'),'dev':dm,'holdout':hm})
    with open(out/'DECO_LI_ABLATION.csv','w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    per=[]
    for key,e1 in rmap['E1'].items():
        d=dict(scene=key[0],image=key[1],prompt=key[2],e1_iou=f(e1['iou']))
        for a in ('E0','E2','E3'):
            r=rmap[a][key]; d[a+'_iou']=f(r['iou']); d[a+'_delta_vs_e1']=f(r['iou'])-f(e1['iou']); d[a+'_candidate_changed_vs_e1']=r.get('new_candidate_regions','')!=e1.get('new_candidate_regions',''); d[a+'_gaussian_added_vs_e0']=int(float(r.get('gaussian_added',0))); d[a+'_gaussian_removed_vs_e0']=int(float(r.get('gaussian_removed',0))); d[a+'_mask_xor_vs_e0']=int(float(r.get('mask_xor_pixels',0))); d[a+'_pred_px']=int(float(r.get('pred_px',0)))
        d['dev_group']=key[2] in dev; d['holdout_group']=key[2] in hold; per.append(d)
    with open(out/'DECO_LI_PER_QUERY.csv','w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    def stats(a):
        rs=per; ds=[r[a+'_delta_vs_e1'] for r in rs]; return dict(improved=sum(x>1e-12 for x in ds),degraded=sum(x<-1e-12 for x in ds),unchanged=sum(abs(x)<=1e-12 for x in ds),candidate_changed=sum(r[a+'_candidate_changed_vs_e1'] for r in rs),max_drop=min(ds),gaussian_added=sum(r[a+'_gaussian_added_vs_e0'] for r in rs),gaussian_removed=sum(r[a+'_gaussian_removed_vs_e0'] for r in rs))
    lines=['# DeCo-LI 结果','', '固定 M1 bank、M2 OFF、C1=image、canon、renorm=0、topk=5、level_until=3、filter_small=50、sel=gap、coefficient=0.25。E0/E1/E2/E3 均使用真实 NAG 搜索和 Gaussian renderer。','', '## 结果表','', '| 组别 | figurines | ramen | teatime | waldo_kitchen | macro | 相对 E1 | dev | holdout |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in table: lines.append('| {arm} | {figurines:.6f} | {ramen:.6f} | {teatime:.6f} | {waldo_kitchen:.6f} | {macro:.6f} | {vs_e1:+.6f} | {dev:.6f} | {holdout:.6f} |'.format(**t))
    lines += ['', '## 逐查询与 Gaussian 变化']
    for a in ('E2','E3'):
        s=stats(a); lines += [f'- {a}: improved={s["improved"]}, degraded={s["degraded"]}, unchanged={s["unchanged"]}; candidate changed vs E1={s["candidate_changed"]}; max degradation={s["max_drop"]:.6f}; Gaussian +{s["gaussian_added"]:,}/-{s["gaussian_removed"]:,} relative to E0.']
    lines += ['', '## 重点查询', '- `DECO_LI_PER_QUERY.csv` 保留 nori、corn、tesla door handle、ottolenghi 的逐查询 IoU、候选变化、Gaussian 增删和 mask XOR。', '', '## 判断', '- E2/E3 是否值得保留取决于其相对 E1 的 scene macro、开发/保留组和退化数量；本轮不依据单一案例调参。', '- 若新算子没有稳定净收益，则不进行 coefficient 第二阶段。']
    (out/'DECO_LI_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    method=['# DeCo-LI 方法说明','', '本轮仅替换 V2 的模式优势聚合器，保留原始 canon 评分、有效模式过滤、正向截断、coefficient/cap、NAG 得分写入和 gap 搜索。', '', '## 聚合定义', '', '- Softmax：冻结 V2 公式 `softmax(mode_score / 0.08 + log(prior))` 后对 positive advantage 加权。', '- MaxSim：有效模式中 `max(clamp(mode_score-main, min=0))`。', '- Prior-MaxSim：`max((prior/max(prior))*positive_advantage)`，非正先验不参与。', '- M2 关闭；没有新增特征、观测、Gaussian 级缓存或 GT 规则。', '', '## 验证', '- 单元测试覆盖无有效模式、单模式、多模式边界、先验抑制、置换不变性、无效模式、非有限值和 Softmax 冻结公式一致性。', '- E0/E1 先验验收通过后才运行 E2/E3。']
    (out/'DECO_LI_METHOD.md').write_text('\n'.join(method),encoding='utf8')
    print(json.dumps({'macro':{x['arm']:x['macro'] for x in table},'stats':{a:stats(a) for a in ('E2','E3')}},ensure_ascii=False,indent=2))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
