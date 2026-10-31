import os, json, time, sys
import gdown
from gdown.download_folder import download_folder

B = "/root/rivermind-data/deco"
# eval-5, ordered smallest-first so one scene can be validated end-to-end early
IDS = [
 ("sofa",  "10sikG9KwXO-RjWN3z4sRUl26r10qMPSY"),
 ("room",  "1g9AV7ADGKkHTaOdfYs3f_OaoVXINAKsf"),
 ("bench", "1qRJZ3hfBV0imG_Ftt0C3OyEa_KWFY3JI"),
 ("lawn",  "1oKynDeyyj_KcpkOAnmsz8TO8F8eFvRoT"),
 ("bed",   "1POSuYN2y5q_6lrUXjij7Q7tQ9JDJ_0pZ"),
]
SKIP = ("database.db", "colmap_output.txt", "project.ini")
dest_root = os.path.join(B, "data", "3dovs")
os.makedirs(dest_root, exist_ok=True)
manifest = {}

for scene, fid in IDS:
    dest = os.path.join(dest_root, scene)
    os.makedirs(dest, exist_ok=True)
    try:
        fs = download_folder(id=fid, output=dest, quiet=True, skip_download=True, use_cookies=True)
    except Exception as e:
        print(f"[{scene}] LIST FAIL {type(e).__name__}", flush=True); continue
    todo = [f for f in fs if not any(s in str(f.path) for s in SKIP)]
    need = [f for f in todo if not (os.path.exists(os.path.join(dest, str(f.path))) and
            os.path.getsize(os.path.join(dest, str(f.path))) > 0)]
    print(f"[{scene}] {len(fs)} files, {len(todo)} wanted, {len(need)} missing", flush=True)
    fail = 0
    for f in need:
        out = os.path.join(dest, str(f.path))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            gdown.download(id=f.id, output=out, quiet=True, resume=True, use_cookies=True)
            fail = 0
        except Exception as e:
            fail += 1
            print(f"  ERR {type(e).__name__} {f.path}", flush=True)
            if fail >= 3:
                print(f"[{scene}] too many consecutive errors, moving on", flush=True)
                break
            time.sleep(2)
        if os.path.exists(out):
            manifest[os.path.relpath(out, dest_root)] = os.path.getsize(out)
    json.dump(manifest, open(os.path.join(B, "data", "manifest.json"), "w"), indent=1)
    print(f"=== [{scene}] pass done ===", flush=True)

print("ALL_SCENES_DONE", flush=True)