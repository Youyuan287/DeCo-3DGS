"""Convert 3D-OVS COLMAP cameras.bin (SIMPLE_RADIAL/OPENCV) to PINHOLE.

The GT masks live in the original (distorted) pixel grid, so we keep the images
and masks untouched and only re-declare the camera as a pinhole with
fx=fy=f, cx, cy.  k1=-0.0175 moves the corner pixels by ~34 px at 4096px,
which is the standard approximation used by 3DGS-family pipelines on 3D-OVS.
"""
import shutil
import struct
import sys

MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
          3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
          6: ("FULL_OPENCV", 12), 7: ("FOV", 5)}


def convert(path):
    data = open(path, "rb").read()
    off = 0
    n = struct.unpack_from("<Q", data, off)[0]
    off += 8
    out = bytearray(struct.pack("<Q", n))
    changed = False
    for _ in range(n):
        cid, mid, w, h = struct.unpack_from("<iiQQ", data, off)
        off += 24
        name, npar = MODELS[mid]
        params = list(struct.unpack_from("<" + "d" * npar, data, off))
        off += 8 * npar
        if mid == 0:      # SIMPLE_PINHOLE -> PINHOLE
            f, cx, cy = params
            new = (1, [f, f, cx, cy])
        elif mid == 2:    # SIMPLE_RADIAL
            f, cx, cy, k1 = params
            new = (1, [f, f, cx, cy])
        elif mid == 4:    # OPENCV
            fx, fy, cx, cy = params[:4]
            new = (1, [fx, fy, cx, cy])
        elif mid == 1:
            new = (1, params)
        else:
            raise SystemExit(f"unhandled camera model {name}")
        changed = True
        out += struct.pack("<iiQQ", cid, new[0], w, h)
        out += struct.pack("<" + "d" * len(new[1]), *new[1])
        print(f"  cam {cid}: {name} {w}x{h} {[round(p,3) for p in params]} -> "
              f"PINHOLE {[round(p,3) for p in new[1]]}")
    if changed:
        shutil.copyfile(path, path + ".orig")
        open(path, "wb").write(bytes(out))
        print("  written", path)
    return changed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        convert(p)