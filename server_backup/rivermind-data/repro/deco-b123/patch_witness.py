"""One-off: create lerf_segment_identity_dump_cnone_witness.py that also dumps
per-view witness sims (image,prompt,rank,view_id,sim) for each candidate."""
import sys

src = sys.argv[1]
s = open(src).read()
out = src.replace(".py", "_witness.py")
assert "wrows" not in s, "already patched"

s = s.replace("rows = []\n", "rows = []\nwrows = []\n", 1)
s = s.replace(
    """                    mask = oRegion == int(root_id)
                    FVs = oF[mask]
                    ws = oW[mask].tolist()""",
    """                    mask = oRegion == int(root_id)
                    FVs = oF[mask]
                    ws = oW[mask].tolist()
                    vw = oView[mask].astype(np.int64).tolist()""", 1)
s = s.replace(
    """                    FVs, ws, mfs, scs = [], [], [], []""",
    """                    FVs, ws, mfs, scs = [], [], [], []
                    vw = []""", 1)
s = s.replace(
    """                        FVs.append(fv / nrm)
                        ws.append(wv)""",
    """                        FVs.append(fv / nrm)
                        ws.append(wv)
                        vw.append(int(v))""", 1)
s = s.replace(
    """                mask_r = render_mask([c], cam, h, w)""",
    """                if FVs is not None and len(FVs) > 0:
                    for _vid, _fv in zip(vw, FVs):
                        wrows.append((name, prompt, i + 1, int(_vid), float(np.dot(_fv, t))))
                mask_r = render_mask([c], cam, h, w)""", 1)
s = s.replace(
    """print("SCENE_DONE", args.scene, len(rows))""",
    """with open(os.path.join(args.out, "%s_witness.csv" % args.scene), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", "prompt", "rank", "view_id", "sim"])
    wr.writerows(wrows)
print("SCENE_DONE", args.scene, len(rows), "witness_rows", len(wrows))""", 1)
open(out, "w").write(s)
print("patched ->", out)