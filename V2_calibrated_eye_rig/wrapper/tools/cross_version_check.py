"""
cross_version_check.py -- portability evidence: replay metadata.json in another Blender version.

    <blender-or-python> wrapper/tools/cross_version_check.py [--samples 8] [--out DIR]

Opens the shipped scene.blend, re-poses all 10 sequence frames, compares the ground truth with
metadata.json (max |diff| of 2-D pupils, rotations, camera matrix) and runs the 5-direction test
(renders go to --out, default a temp dir; nothing in the package is overwritten).
Prints one JSON line prefixed with CROSSVER.
"""
import json
import math
import os
import sys
import tempfile

WRAPPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WRAPPER)

import bpy  # noqa: E402

import eyes  # noqa: E402
import validate_directions  # noqa: E402


def main(argv):
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 8
    out = argv[argv.index("--out") + 1] if "--out" in argv else tempfile.mkdtemp(prefix="xver_")
    with open(os.path.join(eyes.ROOT_DIR, "metadata.json")) as fh:
        meta = json.load(fh)
    e = eyes.get()
    e.scene.cycles.samples = samples
    d2, dr, dp = 0.0, 0.0, 0.0
    for m in meta:
        e.move(m["input"]["h"], m["input"]["v"])
        r = e.metadata()
        for s in ("L", "R"):
            d2 = max(d2, math.dist(r[f"pupil_center_2D_{s}"], m[f"pupil_center_2D_{s}"]))
            dr = max(dr, max(abs(a - b) for a, b in zip(r[f"eye_rotation_{s}"], m[f"eye_rotation_{s}"])))
        dp = max(dp, max(abs(a - b) for ra, rb in zip(r["camera_matrix"], m["camera_matrix"]) for a, b in zip(ra, rb)))
    v = validate_directions.run(e, out, verbose=False)
    res = {"blender": bpy.app.version_string, "python": sys.version.split()[0],
           "replay_max_pupil2D_diff_px": d2, "replay_max_rotation_diff_rad": dr, "replay_max_camera_matrix_diff": dp,
           "direction_validation": f"{v['passed']}/{v['total']}", "samples": samples,
           "tests": {t["test"]: t["status"] for t in v["tests"]}}
    print("CROSSVER " + json.dumps(res))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
