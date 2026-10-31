"""Build a 1024x768 (r=4) image mirror of a 3D-OVS scene so that image_encoding
runs at exactly the resolution the 2DGS model was trained at."""
import os, sys, cv2, glob, shutil
B = "/root/rivermind-data/deco"
sc = sys.argv[1]
src = f"{B}/data/3DOVS/{sc}/images"
dst = f"{B}/data/3DOVS_r4/{sc}/images"
os.makedirs(dst, exist_ok=True)
n = 0
for p in sorted(glob.glob(src + "/*.jpg")):
    im = cv2.imread(p)
    h, w = im.shape[:2]
    out = cv2.resize(im, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst, os.path.basename(p)), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1
print(f"{sc}: {n} images -> {dst} {out.shape[1]}x{out.shape[0]}")
