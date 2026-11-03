"""Frozen local support editing pilot; initialization reused from existing validated worker."""
from pathlib import Path
import hashlib
import time

# The supplied base initializes the exact released NAG, cameras, CLIP, and frozen inputs.
base = Path(__file__).with_name('base_mask_geom.py').read_text()
exec(compile(base.split('gt_dir = ')[0], 'base_mask_geom.py', 'exec'), globals())
torch.manual_seed(0)
np.random.seed(0)
ROOT = Path('/root/rivermind-data/repro/deco-b123')
ref_path = ROOT/'gatee_20261031'/f'{args.scene}_cmaskgate_image.csv'
references = list(csv.DictReader(ref_path.open()))
eval_names = {p.stem for p in (Path(args.gt_root)/args.scene).glob('*.jpg')}
all_cams = sorted(scene.getTrainCameras(), key=lambda c:c.image_name)
cams = {c.image_name:c for c in all_cams}
available = [c for c in all_cams if c.image_name not in eval_names]
selected = [available[i] for i in np.linspace(0,len(available)-1,min(24,len(available))).astype(int)]
N = len(xyz)
print('SOURCE_VIEWS', [c.image_name for c in selected], 'N_GAUSS',N,flush=True)
cache=[]
with torch.no_grad():
    for cam in selected:
        h,w=cam.image_height,cam.image_width
        wt=render_point(cam,gaussians,pipe,bg)['weight'].detach().flatten()
        pc=xyz@torch.as_tensor(np.asarray(cam.R),device='cuda',dtype=torch.float32)+torch.as_tensor(np.asarray(cam.T),device='cuda',dtype=torch.float32)
        z=pc[:,2].clamp(min=1e-6)
        ix=((pc[:,0]/z)*(w/(2*np.tan(cam.FoVx*.5)))+w/2).round().long()
        iy=((pc[:,1]/z)*(h/(2*np.tan(cam.FoVy*.5)))+h/2).round().long()
        good=(wt>1e-4)&(pc[:,2]>1e-4)&(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        ids=torch.where(good)[0].cpu().numpy()
        xx,yy=ix[good].cpu().numpy(),iy[good].cpu().numpy()
        sem=cam.semantic['sem'].detach().cpu().numpy().astype(np.float32)
        sem/=np.maximum(np.linalg.norm(sem,axis=1,keepdims=True),1e-8)
        maps=[]; features=[]; node_info=[]; offset=0
        for lev in (1,2,3):
            seg=cam.semantic['seg_map'][lev].detach().cpu().numpy().astype(np.int64)
            fg=cam.semantic['fg_mask'][lev].detach().cpu().numpy().astype(bool)
            labs,pixels=np.unique(seg[fg],return_counts=True)
            labs=labs[(pixels>=50)&(labs>=0)&(labs<len(sem))]
            lab_to_node=np.full(len(sem),-1,dtype=np.int32)
            lab_to_node[labs]=np.arange(len(labs))+offset
            pixel_lab=seg[yy,xx]
            valid=fg[yy,xx]&(pixel_lab>=0)&(pixel_lab<len(sem))
            mp=np.full(len(ids),-1,dtype=np.int32)
            mp[valid]=lab_to_node[pixel_lab[valid]]
            maps.append(mp)
            features.extend(sem[labs]);node_info.extend((lev,int(L)) for L in labs)
            offset+=len(labs)
        maps=np.stack(maps)
        totals=np.bincount(maps[maps>=0],minlength=offset)
        cache.append(dict(name=cam.image_name,ids=ids,maps=maps,totals=totals,
                          features=np.asarray(features,dtype=np.float32),nodes=node_info))
        print('CACHE_VIEW',cam.image_name,len(ids),offset,flush=True)

def baseline(prompt,image):
    sims=sims_for(prompt); p5=pool(sims,5); n=len(p5)
    pv=[preds[(image,prompt,i+1)] for i in range(n)]
    vv=[ev[(image,prompt,i+1)] for i in range(n)]
    mm=np.array([msig.get((image,prompt,i+1),np.nan) for i in range(n)],float)
    mm=np.where(np.isfinite(mm),mm,np.nanmean(mm) if np.isfinite(mm).any() else 0.)
    scores=np.array([c[3] for c in p5],float);g=gap_len(scores);S0=list(range(g))
    area=np.array([v[0] for v in vv]);vis=np.array([v[1] for v in vv])
    mp=max(pv);prune=[i for i in S0 if i==0 or pv[i]>=.5*mp] or ([0] if S0 else [])
    sc=prune if mp>=.5 else S0
    champ=int(np.argmax(zs(area)+zs(vis)))
    if area[0]<.001 and area.max()>2*area[0] and champ!=0:sc=sorted(set(sc)|{champ})
    sm=list(np.argsort(-(zs(scores)+zs(mm)),kind='stable'))[:g]
    ec=max((mm[i] for i in sc),default=-1e9);em=max((mm[i] for i in sm),default=-1e9)
    sel=sm if set(sm)!=set(S0) and em>ec else sc
    rel=snag._get_related_gaussians([p5[i] for i in sel]) if sel else torch.zeros(N,1)
    return rel.detach().cpu().numpy().ravel()>0,sel

def text_vector(prompt):
    vlm.encode_text(prompt)
    t=vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
    return t/max(np.linalg.norm(t),1e-8)

def repair(G,t,twrong):
    den=np.zeros((3,N),dtype=np.uint16);pos=den.copy();used=np.zeros(3,int);choices=[[],[],[]]
    for v in cache:
        ids,mp=v['ids'],v['maps'];gb=G[ids]; ng=int(gb.sum())
        if ng<20:continue
        picked=mp[:,gb];inter=np.bincount(picked[picked>=0],minlength=len(v['totals']))
        counts=v['totals'];eligible=(inter>=20)&(inter>=.1*ng)&(inter>=.1*counts)
        nodes=np.flatnonzero(eligible)
        if not len(nodes):continue
        ious=inter[nodes]/np.maximum(ng+counts[nodes]-inter[nodes],1)
        picks=[nodes[np.argmax(ious)],nodes[np.argmax(v['features'][nodes]@t)],nodes[np.argmax(v['features'][nodes]@twrong)]]
        for j,node in enumerate(picks):
            den[j,ids]+=1;inside=(mp==node).any(axis=0);pos[j,ids[inside]]+=1
            used[j]+=1;choices[j].append((v['name'],*v['nodes'][node]))
    outputs=[]
    for j in range(3):
        result=G.copy();valid=den[j]>=2;result[valid]=2*pos[j,valid]>=den[j,valid]
        outputs.append(result)
    return outputs,used,choices

family={}
with (ROOT/'lerf_segidentity_20261030'/f'{args.scene}_segidentity.csv').open() as f:
    for r in csv.DictReader(f):family.setdefault((r['image'],r['prompt']),[]).append(r)
prompts=sorted({r['prompt'] for r in references});wrong={p:prompts[(i+1)%len(prompts)] for i,p in enumerate(prompts)}
os.makedirs(args.out,exist_ok=True)
Path(args.out,f'{args.scene}_sources.json').write_text(json.dumps([v['name'] for v in cache]))
out_path=Path(args.out,f'{args.scene}_spatial.csv')
fieldnames=['scene','image','prompt','family','wrong_prompt','n_baseline','A_iou','B_iou','C_iou','D_iou','oracle_ABC_iou','baseline_error']
for tag in 'BCD':fieldnames.extend([f'{tag}_views',f'{tag}_added',f'{tag}_removed'])
started=time.time()
with out_path.open('w',newline='') as handle,Path(args.out,f'{args.scene}_choices.jsonl').open('w') as details,torch.no_grad():
    writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader()
    for qi,r in enumerate(references):
        if args.limit_queries and qi>=args.limit_queries:break
        image,prompt=r['image'],r['prompt'];cam=cams[image];h,w=cam.image_height,cam.image_width
        G,sel=baseline(prompt,image);variants,used,choices=repair(G,text_vector(prompt),text_vector(wrong[prompt]))
        # Ground truth is read only after every selection has been finalized.
        anno=json.load(open(Path(args.gt_root)/args.scene/(image+'.json')))
        gt=np.zeros((h,w),np.uint8)
        for obj in anno['objects']:
            if obj['category']==prompt:gt=np.maximum(gt,polygon_to_mask((h,w),obj['segmentation']))
        gt=gt.astype(bool)
        vals=[iou(render_bool(torch.from_numpy(bm),cam,h,w),gt) if bm.any() else 0. for bm in [G]+variants]
        error=abs(vals[0]-float(r['gatee_iou']))
        if error>1e-6:raise RuntimeError(f'BASELINE_MISMATCH {image} {prompt} {vals[0]} {r["gatee_iou"]}')
        candidates=sorted(family[(image,prompt)],key=lambda x:int(x['rank']))
        best=max(candidates,key=lambda x:float(x['iou']))
        fam='recall' if float(best['iou'])<.3 else ('found' if int(best['rank'])==1 else 'rank2' if int(best['rank'])==2 else 'deep')
        row=dict(scene=args.scene,image=image,prompt=prompt,family=fam,wrong_prompt=wrong[prompt],n_baseline=int(G.sum()),baseline_error=error,oracle_ABC_iou=max(vals[:3]))
        for tag,val in zip('ABCD',vals):row[f'{tag}_iou']=val
        for j,tag in enumerate('BCD'):
            row[f'{tag}_views']=int(used[j]);row[f'{tag}_added']=int((variants[j]&~G).sum());row[f'{tag}_removed']=int((G&~variants[j]).sum())
        writer.writerow(row);handle.flush();details.write(json.dumps(dict(image=image,prompt=prompt,sel=sel,choices=choices),default=int)+'\n');details.flush()
        print('QUERY',args.scene,qi+1,image,prompt,fam,dict(zip('ABCD',vals)),'views',used.tolist(),flush=True)
print('SCENE_DONE',args.scene,'seconds',time.time()-started,'peak_vram',torch.cuda.max_memory_allocated(),flush=True)
