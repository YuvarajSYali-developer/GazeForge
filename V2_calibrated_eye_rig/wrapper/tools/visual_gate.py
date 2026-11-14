"""
visual_gate.py -- regression gate for visual (look-dev) changes to scene.blend.

    python wrapper/tools/visual_gate.py <candidate.blend> <reference.blend> <out_dir> [samples] [--full]

1. Ground truth must be untouched: calibration JSON, and for every pose of a grid
   (the 10 sequence inputs + the 4 extremes + centre) the eye rotations, 3-D/2-D pupil
   centres and camera matrix of the candidate must equal the reference exactly.
2. Renders NEUTRAL/LEFT/RIGHT/UP/DOWN (eye region only, render border, unless --full) and
   requires: pupil detected in the image within 1.0 px of the projected ground truth,
   detected pupil area >= 85 % of the neutral reference area of the reference scene's
   geometry (i.e. no eyelid / lash occlusion of the pupil), and 5-direction signs correct.
Prints GATE PASS / GATE FAIL and writes <out_dir>/gate.json.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

POSES = [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] + \
    [(math.sin(2 * math.pi * k / 10), math.cos(2 * math.pi * k / 10)) for k in range(10)]


def ground_truth(blend):
    os.environ["EYES_SCENE"] = blend
    bpy.ops.wm.open_mainfile(filepath=blend)
    import eyes
    eyes._INSTANCE = None
    e = eyes.get()
    out = {"cal": json.loads(e.rig["calibration_json"]), "poses": []}
    for h, v in POSES:
        e.move(h, v)
        m = e.metadata()
        out["poses"].append({k: m[k] for k in ("eye_rotation_L", "eye_rotation_R", "pupil_center_2D_L",
                                                "pupil_center_2D_R", "camera_matrix")} |
                            {"p3L": m["extra"]["pupil_center_3D_L"], "p3R": m["extra"]["pupil_center_3D_R"]})
    return out


def flat(x):
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in flat(x[k])]
    if isinstance(x, (list, tuple)):
        return [v for i in x for v in flat(i)]
    return [x] if isinstance(x, (int, float)) else []


def main(argv):
    cand, ref, out_dir = (os.path.abspath(a) for a in argv[:3])
    samples = int(argv[3]) if len(argv) > 3 and argv[3].isdigit() else 16
    full = "--full" in argv
    os.makedirs(out_dir, exist_ok=True)
    gt_ref = ground_truth(ref)
    gt_new = ground_truth(cand)
    ign = {"blender_version_built", "asset_source"}
    cal_diff = sorted(k for k in set(gt_ref["cal"]) | set(gt_new["cal"])
                      if k not in ign and gt_ref["cal"].get(k) != gt_new["cal"].get(k))
    gt_max = max(abs(a - b) for pa, pb in zip(gt_ref["poses"], gt_new["poses"])
                 for a, b in zip(flat(pa), flat(pb)))
    res = {"calibration_keys_changed": cal_diff, "ground_truth_max_abs_diff": gt_max}

    import eyes
    e = eyes.get()
    sc = e.scene
    sc.cycles.samples = samples
    if not full:
        r = sc.render
        r.use_border = True
        r.use_crop_to_border = False
        r.border_min_x, r.border_max_x = 0.28, 0.72
        r.border_min_y, r.border_max_y = 0.43, 0.57
    rows = {}
    for name, (h, v) in zip(["NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN"], POSES[:5]):
        e.move(h, v)
        p = e.render(os.path.join(out_dir, f"{name.lower()}.png"))
        md = e.last_metadata
        cam = e.camera.matrix_world.to_translation()
        fx = md["extra"]["camera_intrinsics_K"][0][0]
        row = {}
        for s in ("L", "R"):
            depth = math.dist(md["extra"][f"pupil_center_3D_{s}"], cam)
            roi = 0.95 * e.cal["iris_radius_m"] * fx / depth
            det, area = eyes.detect_pupil_in_image(p, md["extra"][f"eye_center_2D_{s}"], roi,
                                                   pupil_radius_px=e.cal["pupil_radius_m"] * fx / depth)
            row[s] = {"proj": md[f"pupil_center_2D_{s}"], "det": det, "area": area,
                      "err": math.dist(det, md[f"pupil_center_2D_{s}"]) if det else float("inf")}
        rows[name] = row
        print(f"[gate] {name:<7} L err {row['L']['err']:.2f}px area {row['L']['area']}  "
              f"R err {row['R']['err']:.2f}px area {row['R']['area']}  ({e.last_render_seconds:.1f}s)")
    exp = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}
    a0 = {s: rows["NEUTRAL"][s]["area"] for s in ("L", "R")}
    ok_dir = all(((rows[n][s]["det"][0] - rows["NEUTRAL"][s]["det"][0]) * ex +
                  (rows[n][s]["det"][1] - rows["NEUTRAL"][s]["det"][1]) * ey) >= 2.0
                 for n, (ex, ey) in exp.items() for s in ("L", "R"))
    max_err = max(rows[n][s]["err"] for n in rows for s in ("L", "R"))
    min_area_ratio = min(rows[n][s]["area"] / a0[s] for n in rows for s in ("L", "R"))
    res.update({"rows": rows, "max_err_px": max_err, "min_area_ratio_vs_neutral": min_area_ratio,
                "directions_ok": ok_dir, "samples": samples})
    passed = (not cal_diff) and gt_max == 0.0 and max_err <= 1.0 and min_area_ratio >= 0.85 and ok_dir
    res["pass"] = passed
    with open(os.path.join(out_dir, "gate.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"[gate] calibration changed: {cal_diff or 'none'}; ground-truth max diff vs reference: {gt_max:.1e}")
    print(f"[gate] max |proj-det| {max_err:.3f}px; min pupil area ratio {min_area_ratio:.3f}; directions ok {ok_dir}")
    print("GATE PASS" if passed else "GATE FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
