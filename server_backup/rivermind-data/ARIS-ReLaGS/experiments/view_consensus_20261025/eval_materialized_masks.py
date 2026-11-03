"""Post-materialization-only evaluator. It never selects candidates or changes masks."""
import argparse, json, numbers
from pathlib import Path
import numpy as np
import cv2


def gt_mask(path, prompt, shape):
    objects = json.loads(Path(path).read_text(encoding='utf-8'))['objects']
    out = np.zeros(shape[:2], dtype=np.uint8)
    for obj in objects:
        if obj.get('category') != prompt:
            continue
        seg = obj.get('segmentation')
        if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) \
                and len(seg[0]) == 2 and all(isinstance(v, numbers.Number) for v in seg[0]):
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(out, [pts], 1)
        elif isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
            for poly in seg:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(out, [pts], 1)
        elif isinstance(seg, list):
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(out, [pts], 1)
    return out.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='completed real-mode output directory')
    ap.add_argument('--gt-root', required=True, help='GT JSON directory; read only here')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('gt_opened') is not False:
        raise SystemExit('FAIL_CLOSED: prediction manifest does not prove GT was unopened')
    rows = []
    for item in manifest['render_manifest']['records']:
        pred_path = Path(item['path'])
        pred = np.load(pred_path).astype(bool)
        gt = gt_mask(Path(args.gt_root) / (Path(item['image']).stem + '.json'),
                     next(r['prompt'] for r in json.loads((run / 'per_query.json').read_text())
                          if r['query_index'] == item['query_index']), pred.shape)
        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        rows.append({'query_index': item['query_index'], 'arm': item['arm'],
                     'image': item['image'], 'iou': inter / union if union else 1.0,
                     'pred_px': int(pred.sum()), 'gt_px': int(gt.sum())})
    summary = {arm: float(np.mean([r['iou'] for r in rows if r['arm'] == arm]))
               for arm in ('A', 'B')}
    out = {'status': 'POST_MATERIALIZATION_EVAL_ONLY', 'gt_used': True,
           'selection_or_masks_modified': False, 'summary': summary, 'rows': rows}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
