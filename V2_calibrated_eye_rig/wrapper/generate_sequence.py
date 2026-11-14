"""
generate_sequence.py -- one-command pipeline for the Track 1 deliverables.

    blender -b scene.blend --python wrapper/generate_sequence.py            # Blender executable
    python  wrapper/generate_sequence.py                                     # pip "bpy" module (Py 3.11)

Profiles (after "--" when run through Blender):
    (default) quality   Cycles samples stored in scene.blend (32) -> OFFICIAL outputs:
                        renders/, metadata.json, reports/
    --demo              8 samples (fast live demo) -> demo_run/{renders, metadata.json, reports}
    --samples N         custom sample count; only allowed with --demo or --out
    --out DIR           write renders/, metadata.json, reports/ under DIR instead
    --no-validation     skip the 5-direction test

A run whose sample count is below the quality setting can NEVER write into the official
renders/ + metadata.json: the script refuses (exit 2). Regenerating the official outputs
always re-runs the 5-direction validation unless --no-validation is given explicitly.

Loads the scene ONCE, then:
  1. renders the 10 clockwise frames   theta_k = 2*pi*k/10,  h = sin(theta_k),  v = cos(theta_k)
     -> renders/frame_00.png ... frame_09.png
  2. writes metadata.json (one ground-truth record per frame, values read from the live scene)
  3. builds renders/contact_sheet.png (clock layout: frame k sits at angle theta_k, top = k 0)
     and reports/contact_sheet_annotated.png (projected / detected pupils overlaid)
  4. runs the 5-direction validation (reports/direction_validation.{json,md})
  5. writes reports/run_log.json (versions, render config, measured timings, results)
"""
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

T_START = time.perf_counter()

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import eyes  # noqa: E402
import imgutil  # noqa: E402
import validate_directions  # noqa: E402

ROOT = eyes.ROOT_DIR
N_FRAMES = 10
DEMO_SAMPLES = 8


def _peak_rss_mb():
    """Peak resident memory of this process in MB (None where not measurable)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0, 1)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1e6, 1)   # Windows
        except Exception:
            return None


_STATS = {"peak_mb": 0.0}


def _render_stats_handler(stats):
    """Cycles reports 'Mem:...M, Peak:...M' in its status line; keep the maximum."""
    import re
    m = re.search(r"Peak[: ]+([0-9.]+)\s*([MG])", str(stats))
    if m:
        v = float(m.group(1)) * (1024.0 if m.group(2) == "G" else 1.0)
        _STATS["peak_mb"] = max(_STATS["peak_mb"], v)


def sequence_inputs(n=N_FRAMES):
    """Exact challenge sampling: clockwise from the top."""
    out = []
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        out.append((k, theta, math.sin(theta), math.cos(theta)))
    return out


def detect_pupils(e, path, md):
    luma = imgutil.load_rgb(path) @ np.array([0.2126, 0.7152, 0.0722])
    res = {}
    cam = e.camera.matrix_world.to_translation()
    fx = md["extra"]["camera_intrinsics_K"][0][0]
    for s in ("L", "R"):
        depth = math.dist(md["extra"][f"pupil_center_3D_{s}"], cam)
        roi = 0.95 * e.cal["iris_radius_m"] * fx / depth
        det, area = eyes.detect_pupil_in_image(luma, md["extra"][f"eye_center_2D_{s}"], roi,
                                               pupil_radius_px=e.cal["pupil_radius_m"] * fx / depth)
        res[s] = (det, area)
    return res


def contact_sheet(records, out_path, annotated_path=None):
    """Clock-face layout: tile k at angle theta_k measured clockwise from 12 o'clock."""
    tile = 512                         # full frame, 2x box-downscaled
    ex0, ex1, ey0, ey1 = 256, 768, 478, 546   # native-resolution eye strip (512 x 68)
    lab = 34
    tw, th = tile, lab + tile + (ey1 - ey0)
    R = 1290                           # >= 1260 keeps adjacent 512x614 tiles from overlapping
    size = 2 * R + th + 120
    cx = cy = size // 2
    sheets = []
    for annotate in (False, True):
        cv = np.full((size, size, 3), 0.96, dtype=np.float32)
        # clockwise guide circle + arrowheads
        for a in np.linspace(0, 2 * np.pi, 2000):
            x, y = cx + 300 * math.sin(a), cy - 300 * math.cos(a)
            cv[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2] = (0.65, 0.65, 0.65)
        for k in range(10):
            a = 2 * math.pi * (k + 0.5) / 10
            x, y = cx + 300 * math.sin(a), cy - 300 * math.cos(a)
            tx, ty = math.cos(a), math.sin(a)              # clockwise tangent (screen coords)
            for side in (+1, -1):
                bx = x - 18 * tx + side * 10 * ty
                by = y - 18 * ty - side * 10 * tx
                imgutil.draw_line(cv, x, y, bx, by, color=(0.45, 0.45, 0.45), thick=3)
        for i, line in enumerate(["CLOCKWISE GAZE SEQUENCE", "ΘK = 2πK/10", "H = SIN ΘK   V = COS ΘK",
                                  "K=0 AT THE TOP"]):
            w = imgutil.text_width(line, 3)
            imgutil.draw_text(cv, line, cx - w // 2, cy - 70 + i * 36, scale=3, color=(0.15, 0.15, 0.15))
        for rec in records:
            k, theta = rec["frame"], rec["theta"]
            img = imgutil.load_rgb(rec["_path"])
            x = int(cx + R * math.sin(theta) - tw / 2)
            y = int(cy - R * math.cos(theta) - th / 2)
            label = f"K={k}  H={rec['input']['h']:+.3f}  V={rec['input']['v']:+.3f}"
            imgutil.draw_text(cv, label, x, y + 6, scale=3, color=(0.1, 0.1, 0.1))
            cv[y + lab:y + lab + tile, x:x + tw] = imgutil.downscale(img, 2)
            strip = img[ey0:ey1, ex0:ex1].copy()
            if annotate:
                for s in ("L", "R"):
                    px, py = rec[f"pupil_center_2D_{s}"]
                    imgutil.draw_cross(strip, px - ex0, py - ey0, size=5, color=(0.95, 0.1, 0.1))
                    d = rec["extra"]["qa"][f"pupil_center_2D_detected_{s}"]
                    if d:
                        imgutil.draw_circle(strip, d[0] - ex0, d[1] - ey0, 4, color=(0.1, 0.8, 0.2), thick=1.5)
            cv[y + lab + tile:y + th, x:x + tw] = strip
            imgutil.draw_rect(cv, x - 2, y + lab - 2, x + tw + 2, y + th + 2, color=(0.55, 0.55, 0.55), thick=2)
        if annotate:
            imgutil.draw_text(cv, "RED CROSS = METADATA PUPIL (PROJECTED)   GREEN = DETECTED IN PNG",
                              40, size - 40, scale=2, color=(0.2, 0.2, 0.2))
        sheets.append(cv)
    imgutil.save_rgb(out_path, sheets[0])
    if annotated_path:
        imgutil.save_rgb(annotated_path, sheets[1])


def resolve_profile(argv, quality_samples):
    """-> (profile name, samples, output root). Enforces the official-output protection."""
    demo = "--demo" in argv

    def _val(flag):
        i = argv.index(flag)
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]
    samples = None
    if "--samples" in argv:
        try:
            samples = int(_val("--samples"))
        except ValueError:
            raise SystemExit(f"--samples must be a positive integer, got {_val('--samples')!r}") from None
        if samples < 1:
            raise SystemExit(f"--samples must be a positive integer, got {samples}")
    out = os.path.abspath(_val("--out")) if "--out" in argv else None
    if samples is not None and not (demo or out):
        raise SystemExit("--samples is only allowed together with --demo or --out "
                         "(the official outputs always use the quality setting stored in scene.blend)")
    if demo:
        samples = samples or DEMO_SAMPLES
        out = out or os.path.join(ROOT, "demo_run")
    samples = samples or quality_samples
    out = out or ROOT
    if os.path.normcase(os.path.realpath(out)) == os.path.normcase(os.path.realpath(ROOT)) and samples < quality_samples:
        raise SystemExit(f"refusing to write {samples}-sample renders over the official outputs "
                         f"(quality = {quality_samples} samples); use --demo or --out DIR")
    profile = "quality" if samples >= quality_samples else "demo"
    return profile, samples, out


def main(argv):
    do_validation = "--no-validation" not in argv
    t_load = time.perf_counter()
    e = eyes.get()
    quality_samples = int(e.scene.cycles.samples)
    try:
        profile, samples, out_root = resolve_profile(argv, quality_samples)
    except SystemExit as ex:
        print(f"[seq] ERROR: {ex}")
        return 2
    RENDER_DIR = os.path.join(out_root, "renders")
    REPORT_DIR = os.path.join(out_root, "reports")
    os.makedirs(RENDER_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    e.render_dir = RENDER_DIR
    # only a quality run into the package root may (re)write renders/frame_NN.png
    e.allow_official_outputs = (profile == "quality")
    e.scene.cycles.samples = samples
    if _render_stats_handler not in bpy.app.handlers.render_stats:
        bpy.app.handlers.render_stats.append(_render_stats_handler)
    t_loaded = time.perf_counter()
    print(f"[seq] scene ready in {t_loaded - t_load:.2f}s  (Blender {bpy.app.version_string})  "
          f"profile={profile} samples={samples} -> {eyes.relpath_safe(out_root, ROOT) if out_root != ROOT else '.'}")

    records, timings = [], []
    for k, theta, h, v in sequence_inputs():
        e.move(h, v)                                   # synchronous pose update
        path = e.render(os.path.join(RENDER_DIR, f"frame_{k:02d}.png"))
        md = e.last_metadata                           # captured for exactly this pose
        det = detect_pupils(e, path, md)
        rec = {"frame": k, "theta": theta, "input": {"h": h, "v": v}}
        for key in ("eye_rotation_L", "eye_rotation_R", "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix"):
            rec[key] = md[key]
        rec["extra"] = md["extra"]
        rec["extra"]["image"] = f"renders/frame_{k:02d}.png"
        rec["extra"]["render_profile"] = {"profile": profile, "samples": samples}
        qa = {}
        for s in ("L", "R"):
            d, area = det[s]
            qa[f"pupil_center_2D_detected_{s}"] = list(d) if d else None
            qa[f"detected_area_px_{s}"] = area
            qa[f"projection_vs_detection_px_{s}"] = math.dist(d, md[f"pupil_center_2D_{s}"]) if d else None
        rec["extra"]["qa"] = qa
        rec["_path"] = path
        records.append(rec)
        timings.append(e.last_render_seconds)
        print(f"[seq] frame {k:02d}  theta={theta:.4f}  h={h:+.3f} v={v:+.3f}  "
              f"pupil L {md['pupil_center_2D_L'][0]:.2f},{md['pupil_center_2D_L'][1]:.2f}  "
              f"R {md['pupil_center_2D_R'][0]:.2f},{md['pupil_center_2D_R'][1]:.2f}  "
              f"det-err L {qa['projection_vs_detection_px_L']:.2f}px R {qa['projection_vs_detection_px_R']:.2f}px  "
              f"render {e.last_render_seconds:.1f}s")

    with open(os.path.join(out_root, "metadata.json"), "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_path"} for r in records], fh, indent=1)
    t_seq = time.perf_counter()

    contact_sheet(records, os.path.join(RENDER_DIR, "contact_sheet.png"),
                  os.path.join(REPORT_DIR, "contact_sheet_annotated.png"))
    t_sheet = time.perf_counter()

    val = None
    if do_validation:
        val = validate_directions.run(e, os.path.join(REPORT_DIR, "direction_test"))
    t_end = time.perf_counter()

    errs = [r["extra"]["qa"][f"projection_vs_detection_px_{s}"] for r in records for s in ("L", "R")]
    sc = e.scene
    log = {
        "profile": profile,
        "output_root": "." if out_root == ROOT else eyes.relpath_safe(out_root, ROOT).replace(os.sep, "/"),
        "blender_version": bpy.app.version_string,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "render": {"engine": sc.render.engine, "device": sc.cycles.device, "samples": sc.cycles.samples,
                   "denoiser": getattr(sc.cycles, "denoiser", None), "seed": sc.cycles.seed,
                   "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                   "persistent_data": sc.render.use_persistent_data,
                   "view_transform": sc.view_settings.view_transform},
        "frames_generated": len(records),
        "timing_seconds": {
            "interpreter_start_to_scene_ready": round(t_loaded - T_START, 3),
            "scene_load_and_wrapper_init": round(t_loaded - t_load, 3),
            "per_frame_render": [round(t, 3) for t in timings],
            "mean_frame_render": round(sum(timings) / len(timings), 3),
            "sequence_total_incl_metadata": round(t_seq - t_loaded, 3),
            "contact_sheet": round(t_sheet - t_seq, 3),
            "direction_validation": round(t_end - t_sheet, 3) if do_validation else None,
            "total": round(t_end - T_START, 3),
        },
        "memory_mb": {"process_peak_rss": _peak_rss_mb(),
                      "cycles_peak_reported": round(_STATS["peak_mb"], 1) or None},
        "sequence_projection_vs_detection_px": {"max": max(errs), "mean": sum(errs) / len(errs)},
        "direction_validation": f"{val['passed']}/{val['total']} PASS" if val else "skipped",
    }
    with open(os.path.join(REPORT_DIR, "run_log.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    # unregister before interpreter shutdown: a Python handler left in bpy.app.handlers can crash
    # the embedded interpreter at exit (observed as SIGSEGV with the bpy module), losing the exit code
    while _render_stats_handler in bpy.app.handlers.render_stats:
        bpy.app.handlers.render_stats.remove(_render_stats_handler)
    print(f"[seq] done: {len(records)} frames, total {t_end - T_START:.1f}s, "
          f"projection-vs-detection max {max(errs):.2f}px; validation {log['direction_validation']}")
    return 0 if (val is None or val["passed"] == val["total"]) else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
