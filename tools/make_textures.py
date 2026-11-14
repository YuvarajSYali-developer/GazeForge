"""Generate the texture sets (UDIM tiles 1001, 1002, 1004, 1006, 1007) + hair strands for every set.
Usage: python tools/make_textures.py GNM_NPZ LANDMARKS_TXT OUT_DIR [--res 4096] [--sets 01,02]"""
import argparse
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gazeforge_gnm.gnm_model import GNMHead
from gazeforge_gnm.textures.uvraster import TileRaster
from gazeforge_gnm.textures.skin import SkinContext, make_skin
from gazeforge_gnm.textures.eyes import make_eye, make_teeth, make_tongue
from gazeforge_gnm.textures.sets import SETS, IRIS

ap = argparse.ArgumentParser()
ap.add_argument("gnm"); ap.add_argument("landmarks"); ap.add_argument("out")
ap.add_argument("--res", type=int, default=4096); ap.add_argument("--eye-res", type=int, default=2048)
ap.add_argument("--sets", default="")
a = ap.parse_args()
g = GNMHead(a.gnm)
Lt = np.loadtxt(a.landmarks)
LM = (g.template[Lt[:, 0::2].astype(int)] * Lt[:, 1::2][..., None]).sum(1)
t0 = time.time()
R = TileRaster(g, 1001, a.res)
ctx = SkinContext(g, R, LM)
print(f"context {time.time() - t0:.1f}s", flush=True)
want = set(a.sets.split(",")) if a.sets else None
for spec in SETS:
    if want and spec["id"] not in want:
        continue
    t = time.time()
    d = os.path.join(a.out, f"set{spec['id']}_{spec['name']}")
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(d, f"set{spec['id']}")
    brows, lashes, kinds = make_skin(ctx, spec, pre)
    info = {}
    info["L"] = make_eye(g, 1002, 2, IRIS[spec["eye"]], spec["seed"] * 7 + 1, a.eye_res, pre)
    info["R"] = make_eye(g, 1004, 3, IRIS[spec["eye"]], spec["seed"] * 7 + 2, a.eye_res, pre)
    make_teeth(g, 1024, spec["seed"], pre); make_tongue(g, 1024, spec["seed"], pre)
    np.savez_compressed(os.path.join(d, "hair_strands.npz"),
                        brows=np.array(brows, dtype=object), lashes=np.array(lashes, dtype=object),
                        lash_upper=np.array(kinds))
    json.dump({**spec, "eye_geometry_mm": info, "skin_res": a.res, "eye_res": a.eye_res,
               "udim_tiles": {"1001": "skin", "1002": "left eye interior (iris/sclera/pupil)", "1003": "left cornea (no texture: refracting cornea shader)",
                              "1004": "right eye interior", "1005": "right cornea (no texture: refracting cornea shader)", "1006": "teeth+gums", "1007": "tongue"}},
              open(os.path.join(d, "texture_set.json"), "w"), indent=1)
    print(f"set {spec['id']} {spec['name']}: {time.time() - t:.1f}s", flush=True)
import resource
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
