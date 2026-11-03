"""Generate an independent DeCo-derived evaluator, preserving all existing scoring paths.
Refuse if anchors change; never patch original files.
"""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()
src=root/'deco/lerf_ovs/eval_lerf_ovs.py'
out=root/'deco/lerf_ovs/eval_lerf_ovs_b123.py'
s=src.read_text()
def sub(old,new):
 global s
 if s.count(old)!=1:raise SystemExit('DECO SOURCE ANCHOR FAILURE: '+old[:110])
 s=s.replace(old,new,1)
sub('from deco_ops import center_torch  # noqa: E402','from deco_ops import center_torch  # noqa: E402\nfrom lerf_ovs.evidence_readout import load_bank, replace_root, descriptors, sha256')
sub('    ap.add_argument("--tag", default="deco")','''    ap.add_argument("--tag", default="deco")
    ap.add_argument("--b1",type=int,choices=[0,1],default=0)
    ap.add_argument("--control-k1",type=int,choices=[0,1],default=0)
    ap.add_argument("--b2",type=int,choices=[0,1],default=0)
    ap.add_argument("--b3",type=int,choices=[0,1],default=0)
    ap.add_argument("--evidence-bank",default="")
    ap.add_argument("--evidence-temperature",type=float,default=0.08)''')
sub('    args = ap.parse_args()','''    args = ap.parse_args()
    if (args.b2 or args.b3) and not args.b1: raise ValueError('B2/B3 require B1')
    if (args.b1 or args.control_k1) and not args.evidence_bank: raise ValueError('Evidence bank required')
    if args.control_k1 and (args.b1 or args.b2 or args.b3): raise ValueError('K0 must not enable B1/B2/B3')
    if not args.b1 and not args.control_k1 and args.evidence_bank: raise ValueError('B0 cannot load bank')
    os.makedirs(args.out,exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)),exist_ok=True)''')
sub('    snag.feat = [f.cuda().float() for f in snag.feat]', '''    snag.feat = [f.cuda().float() for f in snag.feat]\n\n    raw_root_mean = snag.feat[-1].mean(dim=0).clone()\n    raw_root_count = snag.feat[-1].shape[0]\n    raw_root_dim = snag.feat[-1].shape[1]''')
sub('    vlm = ClipSimMeasure()\n    vlm.load_model()','''    vlm = ClipSimMeasure()
    vlm.load_model()


    evidence = None
    if args.b1 or args.control_k1:
        evidence,meta=load_bank(args.evidence_bank,args.nag,raw_root_count,raw_root_dim)
        if bool(meta.get('b1_enabled')) != bool(args.b1) or bool(meta.get('b2_enabled')) != bool(args.b2):
            raise ValueError('bank build B1/B2 flags mismatch requested ablation')
        print('B123_BANK_META',meta,flush=True)''')
start=s.index('    def text_sims(prompt):');end=s.index('\n    gt_dir =',start)
chunk=s[start:end]
chunk=chunk.replace('    def text_sims(prompt):','    def text_sims(prompt, feats=None):\n        if feats is None: feats=snag.feat',1)
chunk=chunk.replace('for f in snag.feat','for f in feats')
s=s[:start]+chunk+s[end:]
sub('            for prompt in prompts:\n                sims = scale_norm(text_sims(prompt))','''            for prompt in prompts:
                sims=text_sims(prompt)
                if evidence is not None:
                    ev=descriptors(evidence,snag.feat[-1].device,center=raw_root_mean if args.center=='image' else None,renorm=bool(args.renorm))
                    slot_scores=text_sims(prompt,feats=[ev])[0].reshape(evidence['priors'].shape)
                    sims=replace_root(sims,evidence,slot_scores,b3=bool(args.b3),temperature=args.evidence_temperature)
                sims=scale_norm(sims)''')
# This source includes alternative closed-set "--arbitrate" path; it cannot honor B1 as-is.
sub('    if args.control_k1 and (args.b1 or args.b2 or args.b3): raise ValueError(\'K0 must not enable B1/B2/B3\')', '    if args.control_k1 and (args.b1 or args.b2 or args.b3): raise ValueError(\'K0 must not enable B1/B2/B3\')\n    if (args.b1 or args.control_k1) and args.arbitrate: raise ValueError(\'B123 only supports open-vocabulary retrieval, not --arbitrate\')')
# append provenance to saved JSON
sub('filter_small=args.filter_small, center=args.center,','filter_small=args.filter_small, control_k1=args.control_k1, b1=args.b1, b2=args.b2, b3=args.b3, evidence_bank=args.evidence_bank, nag_sha256=sha256(args.nag), center=args.center,')
if out.exists():raise SystemExit('Existing generated evaluator. Do not overwrite; inspect/remove explicitly')
out.write_text(s)
print('GENERATED',out)
