"""Deliverable checks against the Track 1 evaluation slide. Usage: python tools/qa_check.py <package dir>"""
import glob
import json
import os
import subprocess
import sys
from PIL import Image

root = sys.argv[1]
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok &= bool(cond)


sets = sorted(glob.glob(os.path.join(root, "textures", "set*_*")))
check(5 <= len(sets) <= 10, f"texture sets: {len(sets)} (need 5-10)")
for d in sets:
    sid = os.path.basename(d)[3:5]
    tiles = {t: os.path.join(d, f"set{sid}_albedo.{t}.png") for t in (1001, 1002, 1004, 1006, 1007)}
    miss = [t for t, p in tiles.items() if not os.path.exists(p)]
    sizes = {t: Image.open(p).size for t, p in tiles.items() if os.path.exists(p)}
    extra = [os.path.exists(os.path.join(d, f"set{sid}_{k}.1001.png")) for k in ("height", "roughness", "hairmask")]
    check(not miss and sizes[1001] == (4096, 4096) and sizes[1002] == sizes[1004] == (2048, 2048) and all(extra),
          f"set {sid}: UDIM tiles 1001/1002/1004/1006/1007 present, skin 4096^2, eyes 2048^2, height/roughness/hairmask")
    a, b = Image.open(tiles[1002]).tobytes(), Image.open(tiles[1004]).tobytes()
    check(a != b, f"set {sid}: left and right iris textures differ")
renders = sorted(glob.glob(os.path.join(root, "renders", "*.png")))
check(5 <= len(renders) <= 10, f"rendered images: {len(renders)} (need 5-10)")
for v in ("left", "right", "generate_motion"):
    p = os.path.join(root, "videos", f"{v}.mp4")
    if not os.path.exists(p):
        check(False, f"video {v}.mp4 missing"); continue
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate,nb_frames",
                          "-of", "json", p], capture_output=True, text=True).stdout
    info = json.loads(out); dur = float(info["format"]["duration"]); st = info["streams"][0]
    check(dur <= 15.0, f"video {v}: {dur:.2f} s <= 15 s, {st['width']}x{st['height']}, {st['r_frame_rate']} fps, {st.get('nb_frames')} frames")
for b in ("GazeForge_GNM.blend", "GazeForge_GNM_Timeline.blend"):
    check(os.path.exists(os.path.join(root, b)), f"{b} present")
check(os.path.exists(os.path.join(root, "gnm_eyes.py")), "gnm_eyes.py present")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
