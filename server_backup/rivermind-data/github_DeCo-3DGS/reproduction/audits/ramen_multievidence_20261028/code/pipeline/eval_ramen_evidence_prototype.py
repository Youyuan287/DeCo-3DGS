"""Same-scene prototype only: final-region top-5 readout for K1/B2/B3."""
import csv, json, os, sys
import cv2, numpy as np, torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure
from multi_evidence import EvidenceConfig, build_bank, l2n, load_npz, score_bank_certified_style


def polygon(shape, points):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
    return mask


def main(obs, b2, b3, source, model_path, nag_path, gt_root, out):
    p = ArgumentParser(); lp = ModelParams(p); pp = PipelineParams(p)
    args, _ = p.parse_known_args(["-s", source, "-m", model_path])
    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    nag = torch.load(nag_path, map_location="cpu")
    labels = nag["nag"][-1].detach().cpu().numpy().astype(np.int64)
    fallback = l2n(nag["nag_feat"][-1].detach().float().cpu().numpy())
    with np.load(obs, allow_pickle=False) as d:
        f, w, r = d["features"], d["weights"], d["region_ids"]
    u = l2n(l2n(f).mean(0))
    k1 = build_bank(f, w, r, u, EvidenceConfig(max_evidence=1), mode="raw")
    banks = {"k1": k1, "b2_raw": load_npz(b2), "b3_same_scene": load_npz(b3)}
    if fallback.shape[0] != labels.max() + 1: raise ValueError("fallback/label mismatch")
    vlm = ClipSimMeasure(); vlm.load_model()
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    rows=[]; per_arm={key:[] for key in banks}
    for image_file in sorted(x for x in os.listdir(gt_root) if x.endswith('.jpg')):
        name=image_file[:-4]; anno=json.load(open(os.path.join(gt_root,name+'.json'))); cam=cams[name]
        prompts=[]
        for obj in anno['objects']:
            if obj['category'] not in prompts: prompts.append(obj['category'])
        arm_vals={key:[] for key in banks}
        for prompt in prompts:
            text=vlm.get_text_encoding([prompt]).detach().float().cpu().numpy()
            for arm, bank in banks.items():
                score,_=score_bank_certified_style(bank,text,alpha=0.0)
                full=(fallback @ l2n(text)[0]).astype(np.float32)
                for i, rid in enumerate(bank.region_ids): full[int(rid)]=score[i,0]
                chosen=np.argpartition(full,-5)[-5:]
                rel=torch.from_numpy(np.isin(labels,chosen).astype(np.float32)).reshape(-1,1).cuda()
                gaussians._semantics=rel.expand(-1,20)
                sem=render(cam,gaussians,pipe,bg)['semantics']
                pred=(sem.reshape(sem.shape[0],-1)[0]>.5).reshape(cam.image_height,cam.image_width).cpu().numpy().astype(np.uint8)
                gt=np.zeros((cam.image_height,cam.image_width),np.uint8)
                for obj in anno['objects']:
                    if obj['category']==prompt: gt=np.maximum(gt,polygon(gt.shape,obj['segmentation']))
                union=np.logical_or(pred,gt).sum(); iou=float(np.logical_and(pred,gt).sum()/union) if union else float('nan')
                arm_vals[arm].append(iou); rows.append(dict(arm=arm,image=name,prompt=prompt,iou=iou,topk=5))
        for arm, vals in arm_vals.items(): per_arm[arm].append(float(np.mean(vals)))
    summary={"status":"PROTOTYPE_UNVERIFIED_SAME_SCENE_DIRECTION_FINAL_REGION_TOP5","arms":{a:{"scene_miou":float(np.mean(v)),"images":len(v)} for a,v in per_arm.items()}}
    os.makedirs(out,exist_ok=True); json.dump(summary,open(os.path.join(out,'summary.json'),'w'),indent=2)
    with open(os.path.join(out,'per_query.csv'),'w',newline='') as h: csv.DictWriter(h,fieldnames=rows[0]).writeheader(); csv.DictWriter(h,fieldnames=rows[0]).writerows(rows)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__': main(*sys.argv[1:])
