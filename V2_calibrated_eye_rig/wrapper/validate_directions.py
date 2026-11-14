"""
validate_directions.py -- the organisers' 5-direction pass/fail check, automated.

For LEFT / RIGHT / UP / DOWN the wrapper is driven with (h, v) = (-1,0) / (1,0) / (0,1) / (0,-1),
a frame is rendered, and each pupil is located TWO independent ways:
  (a) projection of the 3-D pupil centre through the camera (ground truth used in metadata.json)
  (b) detection of the dark pupil blob in the rendered PNG (ROI centred on the static eyeball
      centre, so the detector is not told where the pupil should be)
Displacements are measured against the neutral (reset) render. An eye passes only if BOTH
measurements move in the expected image direction by >= MIN_SHIFT_PX and the off-axis drift is
small; a test passes only if both eyes pass and the camera/lighting/head locks hold.
A movement in the wrong direction (sign error) therefore fails.

RESET: after the four moves, reset() is called repeatedly; the pose must equal the neutral pose
exactly (quaternions), projected pupils must agree to 1e-6 px, and the re-rendered image must
match the neutral render (deterministic renderer).

Run standalone:
    blender -b scene.blend --python wrapper/validate_directions.py [-- --demo | -- --samples N]
    python wrapper/validate_directions.py [--demo]            (pip bpy)
Quality runs write reports/direction_validation.{json,md} + reports/direction_test/;
runs below the scene's quality sample count write under demo_run/ instead.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import eyes  # noqa: E402
import imgutil  # noqa: E402

MIN_SHIFT_PX = 2.0           # minimum visible displacement along the expected axis (pixels @1024^2)
MAX_CROSS_RATIO = 0.5        # off-axis drift must be < 50 % of the on-axis displacement
TESTS = [                    # name, (h, v), expected image direction (x, y) with y pointing DOWN
    ("LEFT", (-1.0, 0.0), (-1, 0)),
    ("RIGHT", (1.0, 0.0), (1, 0)),
    ("UP", (0.0, 1.0), (0, -1)),
    ("DOWN", (0.0, -1.0), (0, 1)),
]


def _measure(e, path):
    md = e.last_metadata
    luma = imgutil.load_rgb(path) @ np.array([0.2126, 0.7152, 0.0722])
    out = {}
    for s in ("L", "R"):
        c2 = md["extra"][f"eye_center_2D_{s}"]
        # ROI radius = 0.95 x projected iris radius (pupil always inside, eyelids excluded)
        depth = math.dist(md["extra"][f"pupil_center_3D_{s}"], e.camera.matrix_world.to_translation())
        fx = md["extra"]["camera_intrinsics_K"][0][0]
        roi = 0.95 * e.cal["iris_radius_m"] * fx / depth
        rp = e.cal["pupil_radius_m"] * fx / depth
        det, area = eyes.detect_pupil_in_image(luma, c2, roi, pupil_radius_px=rp)
        out[s] = {"projected": md[f"pupil_center_2D_{s}"], "detected": list(det) if det else None,
                  "detected_area_px": area, "roi_radius_px": roi, "pupil_radius_px": rp}
    return out


def run(e=None, out_dir=None, verbose=True):
    e = e or eyes.get()
    out_dir = os.path.abspath(out_dir or os.path.join(eyes.ROOT_DIR, "reports", "direction_test"))
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # ---- neutral reference ----
    e.reset()
    n_path = e.render(os.path.join(out_dir, "neutral.png"))
    n_md = e.last_metadata
    n_meas = _measure(e, n_path)
    n_quat = {s: list(e.rig.pose.bones[f"eye.{s}"].rotation_quaternion) for s in ("L", "R")}
    frames = {"NEUTRAL": (n_path, n_meas)}

    for name, (h, v), (ex, ey) in TESTS:
        e.reset()
        e.move(h, v)
        p = e.render(os.path.join(out_dir, f"{name.lower()}.png"))
        meas = _measure(e, p)
        frames[name] = (p, meas)
        lock = e.static_state_unchanged()
        eye_res = {}
        for s in ("L", "R"):
            row = {}
            ok = True
            for kind in ("projected", "detected"):
                a, b = n_meas[s][kind], meas[s][kind]
                if a is None or b is None:
                    row[kind] = {"dx": None, "dy": None, "pass": False}
                    ok = False
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                along = dx * ex + dy * ey                      # signed displacement along expected axis
                cross = abs(dx * ey - dy * ex)                 # perpendicular drift
                k_ok = along >= MIN_SHIFT_PX and cross <= MAX_CROSS_RATIO * along
                row[kind] = {"dx": round(dx, 4), "dy": round(dy, 4), "along_px": round(along, 4),
                             "cross_px": round(cross, 4), "pass": bool(k_ok)}
                ok &= k_ok
            row["pass"] = bool(ok)
            eye_res[s] = row
        passed = eye_res["L"]["pass"] and eye_res["R"]["pass"] and lock["all"]
        results.append({"test": name, "input": {"h": h, "v": v},
                        "expected_image_direction": {"x": ex, "y": ey, "note": "image y axis points down"},
                        "left_eye": eye_res["L"], "right_eye": eye_res["R"],
                        "static_scene_locked": lock, "render": eyes.relpath_safe(p, eyes.ROOT_DIR),
                        "status": "PASS" if passed else "FAIL"})

    # ---- RESET: repeated resets must restore exactly the neutral baseline ----
    e.move(0.6, -0.8)
    pose_diffs = []
    for _ in range(3):
        e.reset()
        pose_diffs.append(max(abs(a - b) for s in ("L", "R")
                              for a, b in zip(e.rig.pose.bones[f"eye.{s}"].rotation_quaternion, n_quat[s])))
    r_path = e.render(os.path.join(out_dir, "reset.png"))
    r_md = e.last_metadata
    r_meas = _measure(e, r_path)
    frames["RESET"] = (r_path, r_meas)
    img_a, img_b = imgutil.load_rgb(n_path), imgutil.load_rgb(r_path)
    img_diff_max = float(np.abs(img_a - img_b).max() * 255.0)
    img_diff_mean = float(np.abs(img_a - img_b).mean() * 255.0)
    eye_res = {}
    for s in ("L", "R"):
        pj = math.dist(n_md[f"pupil_center_2D_{s}"], r_md[f"pupil_center_2D_{s}"])
        dt = math.dist(n_meas[s]["detected"], r_meas[s]["detected"]) if n_meas[s]["detected"] and r_meas[s]["detected"] else float("inf")
        rot = max(abs(a - b) for a, b in zip(n_md[f"eye_rotation_{s}"], r_md[f"eye_rotation_{s}"]))
        ok = pj <= 1e-6 and dt <= 0.25 and rot <= 1e-9
        eye_res[s] = {"projected_shift_px": pj, "detected_shift_px": round(dt, 4), "rotation_diff_rad": rot, "pass": bool(ok)}
    lock = e.static_state_unchanged()
    reset_ok = (eye_res["L"]["pass"] and eye_res["R"]["pass"] and max(pose_diffs) <= 1e-12
                and img_diff_mean <= 0.5 and lock["all"])
    results.append({"test": "RESET", "input": "reset() x3 after move(0.6, -0.8)",
                    "expected_image_direction": "pupils return exactly to the neutral baseline",
                    "left_eye": eye_res["L"], "right_eye": eye_res["R"],
                    "pose_quaternion_max_diff_over_3_resets": max(pose_diffs),
                    "image_diff_vs_neutral_8bit": {"max": round(img_diff_max, 3), "mean": round(img_diff_mean, 4)},
                    "static_scene_locked": lock, "render": eyes.relpath_safe(r_path, eyes.ROOT_DIR),
                    "status": "PASS" if reset_ok else "FAIL"})

    summary = {"passed": sum(r["status"] == "PASS" for r in results), "total": len(results),
               "criteria": {"min_shift_px": MIN_SHIFT_PX, "max_cross_ratio": MAX_CROSS_RATIO,
                            "measurements": "projected ground truth AND independent image detection must both pass"},
               "neutral_pupils": {s: n_meas[s] for s in ("L", "R")}, "tests": results}
    with open(os.path.join(out_dir, "..", "direction_validation.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    _write_markdown(summary, os.path.join(out_dir, "..", "direction_validation.md"))
    _sheet(frames, n_meas, os.path.join(out_dir, "direction_sheet.png"))
    if verbose:
        for r in results:
            if r["test"] == "RESET":
                print(f"[5-dir] RESET  {r['status']}  pose diff {r['pose_quaternion_max_diff_over_3_resets']:.1e}"
                      f"  image diff mean {r['image_diff_vs_neutral_8bit']['mean']}")
            else:
                L, R = r["left_eye"], r["right_eye"]
                print(f"[5-dir] {r['test']:<5}  {r['status']}  L proj ({L['projected']['dx']:+.2f},{L['projected']['dy']:+.2f}) "
                      f"det ({L['detected']['dx']:+.2f},{L['detected']['dy']:+.2f})   R proj ({R['projected']['dx']:+.2f},"
                      f"{R['projected']['dy']:+.2f}) det ({R['detected']['dx']:+.2f},{R['detected']['dy']:+.2f})")
        print(f"[5-dir] {summary['passed']}/{summary['total']} passed")
    return summary


def _write_markdown(summary, path):
    L = ["# Five-direction validation report", "",
         f"Result: **{summary['passed']}/{summary['total']} PASS**", "",
         "Pupil displacement relative to the neutral (reset) render, in pixels of the 1024x1024 image "
         "(x right, y down). Each eye is measured by projection of the 3-D pupil centre (`proj`) and by "
         "independent dark-blob detection in the PNG (`det`); both must move the expected way by "
         f">= {summary['criteria']['min_shift_px']} px with off-axis drift < "
         f"{int(summary['criteria']['max_cross_ratio']*100)} % of the on-axis shift.", "",
         "| Test | Input (h, v) | Expected | Left eye proj dx, dy | Left eye det dx, dy | Right eye proj dx, dy | Right eye det dx, dy | Left | Right | Camera locked | Status |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    exp = {"LEFT": "x < 0", "RIGHT": "x > 0", "UP": "y < 0", "DOWN": "y > 0"}
    for r in summary["tests"]:
        if r["test"] == "RESET":
            continue
        le, re_ = r["left_eye"], r["right_eye"]
        L.append(f"| {r['test']} | ({r['input']['h']:+.0f}, {r['input']['v']:+.0f}) | {exp[r['test']]} | "
                 f"{le['projected']['dx']:+.2f}, {le['projected']['dy']:+.2f} | {le['detected']['dx']:+.2f}, {le['detected']['dy']:+.2f} | "
                 f"{re_['projected']['dx']:+.2f}, {re_['projected']['dy']:+.2f} | {re_['detected']['dx']:+.2f}, {re_['detected']['dy']:+.2f} | "
                 f"{'PASS' if le['pass'] else 'FAIL'} | {'PASS' if re_['pass'] else 'FAIL'} | "
                 f"{r['static_scene_locked']['all']} | **{r['status']}** |")
    r = summary["tests"][-1]
    L += ["", "## Reset", "",
          f"- `reset()` called 3x after `move(0.6, -0.8)`; max eye-bone quaternion difference vs neutral: "
          f"{r['pose_quaternion_max_diff_over_3_resets']:.1e}",
          f"- projected pupil shift vs neutral: L {r['left_eye']['projected_shift_px']:.1e} px, "
          f"R {r['right_eye']['projected_shift_px']:.1e} px",
          f"- detected pupil shift vs neutral: L {r['left_eye']['detected_shift_px']} px, R {r['right_eye']['detected_shift_px']} px",
          f"- re-rendered image vs neutral render (8-bit): max abs diff {r['image_diff_vs_neutral_8bit']['max']}, "
          f"mean {r['image_diff_vs_neutral_8bit']['mean']}",
          f"- Status: **{r['status']}**", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def _sheet(frames, n_meas, path):
    """Eye-region strips (2x zoom) for NEUTRAL, LEFT, RIGHT, UP, DOWN, RESET.
    Red cross = neutral pupil position, green circle = pupil detected in this frame."""
    order = ["NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN", "RESET"]
    x0, x1, y0, y1 = 320, 704, 478, 546
    z = 2
    sw, sh = (x1 - x0) * z, (y1 - y0) * z
    pad, lab = 12, 26
    canvas = np.full((len(order) * (sh + lab + pad) + pad, sw + 2 * pad, 3), 0.95, dtype=np.float32)
    for i, name in enumerate(order):
        p, meas = frames[name]
        img = imgutil.load_rgb(p)[y0:y1, x0:x1]
        strip = imgutil.upscale_nearest(img, z)
        oy = pad + i * (sh + lab + pad)
        imgutil.draw_text(canvas, name, pad, oy + 4, scale=2)
        canvas[oy + lab:oy + lab + sh, pad:pad + sw] = strip
        for s in ("L", "R"):
            nx, ny = n_meas[s]["projected"]
            imgutil.draw_cross(canvas, pad + (nx - x0) * z, oy + lab + (ny - y0) * z, size=7, color=(0.9, 0.1, 0.1))
            d = meas[s]["detected"]
            if d:
                imgutil.draw_circle(canvas, pad + (d[0] - x0) * z, oy + lab + (d[1] - y0) * z, 5, color=(0.1, 0.75, 0.2), thick=2)
    imgutil.save_rgb(path, canvas)


if __name__ == "__main__":
    # Same protection as generate_sequence.py: draft-quality runs go to demo_run/, never reports/.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    e = eyes.get()
    quality = int(e.scene.cycles.samples)
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else (8 if "--demo" in argv else quality)
    e.scene.cycles.samples = samples
    out = os.path.join(eyes.ROOT_DIR, "reports", "direction_test") if samples >= quality else \
        os.path.join(eyes.ROOT_DIR, "demo_run", "reports", "direction_test")
    print(f"[5-dir] samples={samples} -> {eyes.relpath_safe(out, eyes.ROOT_DIR)}")
    s = run(e, out)
    sys.exit(0 if s["passed"] == s["total"] else 1)
