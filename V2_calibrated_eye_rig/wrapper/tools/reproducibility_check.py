"""
reproducibility_check.py -- re-render official sequence frames and compare them with renders/.

    <python-with-bpy> wrapper/tools/reproducibility_check.py [--frames 0 5] [--out DIR]

Uses the scene's quality settings (seed 0, no adaptive sampling) and writes the re-renders to a
temporary directory (never into renders/). Reports per-frame max / mean absolute 8-bit difference
and whether the metadata of the re-rendered pose equals metadata.json. Prints REPRO <json>.
"""
import json
import math
import os
import sys
import tempfile

WRAPPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WRAPPER)

import numpy as np  # noqa: E402

import eyes  # noqa: E402
import imgutil  # noqa: E402


def main(argv):
    frames = [0, 5]
    if "--frames" in argv:
        i = argv.index("--frames") + 1
        frames = []
        while i < len(argv) and argv[i].lstrip("-").isdigit():
            frames.append(int(argv[i]))
            i += 1
    out = argv[argv.index("--out") + 1] if "--out" in argv else tempfile.mkdtemp(prefix="repro_")
    with open(os.path.join(eyes.ROOT_DIR, "metadata.json")) as fh:
        meta = json.load(fh)
    e = eyes.get()
    res = {"samples": e.scene.cycles.samples, "frames": {}}
    for k in frames:
        m = meta[k]
        e.move(m["input"]["h"], m["input"]["v"])
        p = e.render(os.path.join(out, f"frame_{k:02d}.png"))
        a = imgutil.load_rgb(p)
        b = imgutil.load_rgb(os.path.join(eyes.ROOT_DIR, "renders", f"frame_{k:02d}.png"))
        d = np.abs(a - b) * 255.0
        md = e.last_metadata
        gt = max(math.dist(md[f"pupil_center_2D_{s}"], m[f"pupil_center_2D_{s}"]) for s in ("L", "R"))
        res["frames"][k] = {"max_abs_diff_8bit": round(float(d.max()), 3), "mean_abs_diff_8bit": round(float(d.mean()), 5),
                            "identical_pixels": bool(d.max() == 0), "metadata_pupil_diff_px": gt,
                            "render_seconds": round(e.last_render_seconds, 1)}
    print("REPRO " + json.dumps(res))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
