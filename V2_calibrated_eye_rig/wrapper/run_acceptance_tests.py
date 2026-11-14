"""
run_acceptance_tests.py -- acceptance tests A..I against the SAVED deliverables.

    python wrapper/run_acceptance_tests.py [--zip path/to/team.zip]
    blender -b --python wrapper/run_acceptance_tests.py -- [--zip ...]

Prerequisite: generate_sequence.py has been run (renders/, metadata.json, reports/).
Writes reports/acceptance_report.{json,md}. Exit code 0 only if every test passes.

No test is "passed because something ran": each check compares a measured quantity with
the specification, and test H re-derives every rotation / pupil centre independently in numpy
from the calibration constants (it does not reuse Blender's math).
"""
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import eyes  # noqa: E402
import imgutil  # noqa: E402

ROOT = eyes.ROOT_DIR
REQUIRED_ZIP_PATHS = ["scene.blend", "wrapper/eyes.py", "wrapper/generate_sequence.py", "metadata.json", "README.md"] + \
    [f"renders/frame_{k:02d}.png" for k in range(10)] + ["renders/contact_sheet.png"]
# Blender stores object/bone transforms in float32 (~1e-7 relative), so geometric equalities are
# checked at 1 micrometre / 1e-6 relative -- four orders of magnitude below one rendered pixel
# (1 px ~ 0.22 mm at the eye), but far above float32 round-off.
TOL_M = 1e-6
TOL_REL = 1e-6
REQUIRED_FIELDS = ["frame", "theta", "input", "eye_rotation_L", "eye_rotation_R",
                   "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix"]


class Suite:
    def __init__(self):
        self.tests = []

    def test(self, tid, title):
        t = {"id": tid, "title": title, "checks": []}
        self.tests.append(t)
        return t

    @staticmethod
    def _portable(text):
        """Strip machine-specific absolute paths so reports are identical on any machine."""
        text = str(text)
        for base, tag in ((ROOT, "<pkg>"), (tempfile.gettempdir(), "<tmp>"), (os.path.expanduser("~"), "<home>")):
            for b in {base, base.replace(os.sep, "/")}:
                if b and len(b) > 1:
                    text = text.replace(b, tag)
        return text

    def check(self, t, name, ok, detail=""):
        detail = self._portable(detail)
        t["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {t['id']}: {name}  {detail}")
        return ok



_TMP_DIRS = []


def _tmpdir(prefix="gf_accept_"):
    """Temp folder removed when the suite exits."""
    d = tempfile.mkdtemp(prefix=prefix)
    _TMP_DIRS.append(d)
    return d


def _cleanup_tmp():
    import shutil
    for d in _TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


import atexit  # noqa: E402
atexit.register(_cleanup_tmp)

def euler_xyz_to_matrix(e):
    x, y, z = e
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                     # Blender 'XYZ' Euler: X applied first


def shortest_arc(a, d):
    a, d = a / np.linalg.norm(a), d / np.linalg.norm(d)
    ax = np.cross(a, d)
    s, c = np.linalg.norm(ax), float(a @ d)
    if s < 1e-15:
        return np.eye(3)
    k = ax / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def main(argv):
    zip_path = argv[argv.index("--zip") + 1] if "--zip" in argv else None
    ref_meta = argv[argv.index("--reference-metadata") + 1] if "--reference-metadata" in argv else None
    S = Suite()
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ A. scene integrity
    t = S.test("A", "Scene integrity (saved scene.blend)")
    scene_path = os.path.join(ROOT, "scene.blend")
    bpy.ops.wm.open_mainfile(filepath=scene_path)
    S.check(t, "scene.blend opens", bpy.data.filepath == scene_path, os.path.relpath(scene_path, ROOT))
    need = ["Face", "EyeRig", "RenderCamera", "FixationTarget", "KeyLight", "FillLight", "RimLight"] + \
        [f"{n}.{s}" for n in ("Eyeball", "Iris", "Pupil", "Cornea") for s in ("L", "R")]
    missing = [n for n in need if n not in bpy.data.objects]
    S.check(t, "required objects exist", not missing, f"missing={missing}")
    rig = bpy.data.objects.get("EyeRig")
    bones = sorted(b.name for b in rig.data.bones) if rig else []
    S.check(t, "eye bones eye.L / eye.R (+ head) exist", {"head", "eye.L", "eye.R"} <= set(bones), bones)
    S.check(t, "calibration stored in scene", rig is not None and "calibration_json" in rig.keys())
    cam = bpy.data.objects.get("RenderCamera")
    S.check(t, "scene camera is RenderCamera", bpy.context.scene.camera == cam)
    imgs = list(bpy.data.images)
    unpacked = [i.name for i in imgs if i.source == "FILE" and i.packed_file is None]
    S.check(t, "all textures packed into .blend", imgs and not unpacked, f"{len(imgs)} images, unpacked={unpacked}")
    stats = {}
    for im in imgs:
        px = np.empty(im.size[0] * im.size[1] * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        stats[im.name] = round(float(px.reshape(-1, 4)[:, :3].std()), 4)
    S.check(t, "textures load with real content (non-constant)", all(v > 0.01 for v in stats.values()), stats)
    face = bpy.data.objects.get("Face")
    has_tex = face and any(n.type == "TEX_IMAGE" and n.image for m in face.data.materials for n in m.node_tree.nodes)
    S.check(t, "face mesh is UV-mapped and image-textured", has_tex and len(face.data.uv_layers) > 0,
            f"{len(face.data.vertices)} verts")
    S.check(t, "no external library links", len(bpy.data.libraries) == 0)

    # ------------------------------------------------------------------ B. API integrity
    t = S.test("B", "API integrity")
    e = eyes.get()
    st = e.move(0.3, -0.4)
    S.check(t, "eyes.move(h, v) works and updates pose", abs(st["input"]["h"] - 0.3) < 1e-12 and
            e.rig.pose.bones["eye.L"].rotation_quaternion.angle > 0)
    bad_ok = []
    for bad in [(1.5, 0), (0, -1.01), (float("nan"), 0), (float("inf"), 0), ("x", 0), (None, 0), (True, 0)]:
        try:
            e.move(*bad)
            bad_ok.append((bad, "accepted"))
        except (ValueError, TypeError) as ex:
            bad_ok.append((bad, type(ex).__name__))
    S.check(t, "invalid inputs rejected (out of range, NaN, inf, non-numeric, bool)",
            all(r != "accepted" for _, r in bad_ok), bad_ok)
    S.check(t, "boundary inputs accepted (+-1)", all(e.move(h, v) for h, v in [(1, 1), (-1, -1), (1, -1)]))
    qs = []
    for _ in range(5):
        e.move(np.random.default_rng(len(qs)).uniform(-1, 1), 0.7)
        e.reset()
        qs.append([tuple(e.rig.pose.bones[f"eye.{s}"].rotation_quaternion) for s in ("L", "R")])
    S.check(t, "reset() repeatable (5x identical quaternions)", all(q == qs[0] for q in qs))
    e.scene.cycles.samples, old = 2, e.scene.cycles.samples
    tmp_png = os.path.join(_tmpdir(), "api_probe.png")
    p = e.render(tmp_png)
    e.scene.cycles.samples = old
    probe = imgutil.load_rgb(p)
    S.check(t, "eyes.render() writes a 1024x1024 PNG and returns its path",
            p == tmp_png and os.path.isfile(p) and probe.shape[:2] == (1024, 1024), f"<tmp>/{os.path.basename(p)} {probe.shape}")
    S.check(t, "module-level API eyes.move/reset/render/metadata exists",
            all(callable(getattr(eyes, n, None)) for n in ("move", "reset", "render", "metadata")))
    S.check(t, "no GUI needed (running in background mode)", bpy.app.background)

    # ------------------------------------------------------------------ C. anatomical articulation
    t = S.test("C", "Anatomical articulation")
    e.reset()
    base = {s: (e.eyeballs[s].matrix_world.to_translation().copy(),
                {c.name: c.matrix_local.copy() for c in e.eyeballs[s].children_recursive}) for s in ("L", "R")}
    face_mw = face.matrix_world.copy()
    face_co = np.empty(len(face.data.vertices) * 3)
    face.data.vertices.foreach_get("co", face_co)
    worst_t, worst_s, worst_child, worst_pd = 0.0, 0.0, 0.0, 0.0
    d_pupil = e.cal["pupil_offset_local_m"]
    grid = [(h, v) for h in (-1, -0.5, 0, 0.5, 1) for v in (-1, -0.5, 0, 0.5, 1)]
    for h, v in grid:
        e.move(h, v)
        for s in ("L", "R"):
            mw = e.eyeballs[s].matrix_world
            worst_t = max(worst_t, (mw.to_translation() - base[s][0]).length)
            worst_s = max(worst_s, max(abs(x - 1) for x in mw.to_scale()))
            for c in e.eyeballs[s].children_recursive:
                worst_child = max(worst_child, max(abs(a - b) for ra, rb in zip(c.matrix_local, base[s][1][c.name]) for a, b in zip(ra, rb)))
            worst_pd = max(worst_pd, abs((e.pupils[s].matrix_world.to_translation() - mw.to_translation()).length - d_pupil))
    S.check(t, "eyeball centres never translate (25-pose grid)", worst_t < TOL_M, f"max {worst_t:.2e} m")
    S.check(t, "eyeball scale preserved", worst_s < TOL_REL, f"max |s-1| {worst_s:.2e}")
    S.check(t, "iris/pupil/cornea rigidly attached (local transforms unchanged)", worst_child < TOL_REL, f"{worst_child:.1e}")
    S.check(t, "pupil stays on the eyeball radius (rotation about own centre)", worst_pd < TOL_M, f"{worst_pd:.2e} m")
    co2 = np.empty_like(face_co)
    face.data.vertices.foreach_get("co", co2)
    S.check(t, "head/face does not move or deform", np.array_equal(face_co, co2) and
            all(abs(a - b) < 1e-12 for ra, rb in zip(face.matrix_world, face_mw) for a, b in zip(ra, rb)) and
            e.rig.pose.bones["head"].matrix_basis == Matrix.Identity(4))
    S.check(t, "eye assemblies parented to eye bones",
            all(e.eyeballs[s].parent == e.rig and e.eyeballs[s].parent_bone == f"eye.{s}" for s in ("L", "R")) and
            all(e.pupils[s].parent.name == f"Iris.{s}" and e.pupils[s].parent.parent.name == f"Eyeball.{s}" for s in ("L", "R")))

    # ------------------------------------------------------------------ D. fixation convergence
    t = S.test("D", "Binocular fixation convergence")
    e.reset()
    M = e.eye_midpoint()
    F0 = e.nominal_fixation()
    fwd, rgt, up = e.head_axes()
    S.check(t, "nominal fixation 35 cm forward of the eye midpoint", abs((F0 - M).length - 0.35) < TOL_M and
            (F0 - M).normalized().dot(fwd) > 1 - TOL_REL, f"{(F0 - M).length:.9f} m")
    worst_miss, worst_bound, worst_fwd = 0.0, 0.0, 0.0
    for h, v in grid + [(math.sin(2 * math.pi * k / 10), math.cos(2 * math.pi * k / 10)) for k in range(10)]:
        stt = e.move(h, v)
        T = e.target_for(h, v)
        worst_miss = max(worst_miss, stt["gaze_ray_miss_m_L"], stt["gaze_ray_miss_m_R"])
        off = T - F0
        worst_bound = max(worst_bound, abs(off.dot(rgt)), abs(off.dot(up)))
        worst_fwd = max(worst_fwd, abs(off.dot(fwd)))
    S.check(t, "both gaze rays pass through the common target (35 poses)", worst_miss < TOL_M, f"max miss {worst_miss:.2e} m")
    S.check(t, "target displacement within +-5 cm bounds", worst_bound <= 0.05 + TOL_M, f"max {worst_bound:.7f} m")
    S.check(t, "target stays in the 35 cm fixation plane", worst_fwd < TOL_M, f"max off-plane {worst_fwd:.1e} m")
    e.reset()
    q_reset = [tuple(e.rig.pose.bones[f"eye.{s}"].rotation_quaternion) for s in ("L", "R")]
    e.move(0, 0)
    q_centre = [tuple(e.rig.pose.bones[f"eye.{s}"].rotation_quaternion) for s in ("L", "R")]
    S.check(t, "centre gaze move(0,0) == calibrated neutral (reset)", q_reset == q_centre)
    stt = e.reset()
    S.check(t, "neutral pose converges (symmetric inward rotation)", abs(stt["yaw_deg_L"] + stt["yaw_deg_R"]) < 1e-4 and stt["vergence_deg"] > 1,
            f"yaw L {stt['yaw_deg_L']:.3f} deg, R {stt['yaw_deg_R']:.3f} deg, vergence {stt['vergence_deg']:.3f} deg")

    # ------------------------------------------------------------------ E. camera calibration
    t = S.test("E", "Camera calibration")
    cpos = e.camera.matrix_world.to_translation()
    S.check(t, "camera 40 cm forward of the eye midpoint", abs((cpos - M).length - 0.40) < TOL_M and
            (cpos - M).normalized().dot(fwd) > 1 - TOL_REL, f"{(cpos - M).length:.7f} m")
    view = -(e.camera.matrix_world.to_3x3().normalized() @ Vector((0.0, 0.0, 1.0)))   # camera looks along local -Z
    S.check(t, "camera faces the face (optical axis = -forward)", view.normalized().dot(-fwd) > 1 - TOL_REL,
            f"cos = {view.normalized().dot(-fwd):.12f}")
    pm = e.project(M)
    S.check(t, "optical axis passes through the eye midpoint (projects to image centre)",
            math.dist(pm, (512, 512)) < 1e-3, pm)
    r = e.scene.render
    S.check(t, "resolution 1024x1024 @100 %", (r.resolution_x, r.resolution_y, r.resolution_percentage) == (1024, 1024, 100))
    with open(os.path.join(ROOT, "metadata.json")) as fh:
        meta = json.load(fh)
    cams = {json.dumps(m["camera_matrix"]) for m in meta}
    S.check(t, "camera matrix identical in all 10 frames (camera locked)", len(cams) == 1)
    S.check(t, "lighting/camera/head locks hold", e.static_state_unchanged()["all"])
    det_err = [m["extra"]["qa"][f"projection_vs_detection_px_{s}"] for m in meta for s in ("L", "R")]
    S.check(t, "projected pupils agree with pupils detected in the PNGs", max(det_err) < 1.5,
            f"max {max(det_err):.3f} px, mean {sum(det_err)/len(det_err):.3f} px over 20 pupils")

    # ------------------------------------------------------------------ F. direction validation
    t = S.test("F", "Five-direction validation")
    with open(os.path.join(ROOT, "reports", "direction_validation.json")) as fh:
        dv = json.load(fh)
    for r_ in dv["tests"]:
        S.check(t, f"{r_['test']} passes", r_["status"] == "PASS",
                "" if r_["test"] == "RESET" else
                f"L proj ({r_['left_eye']['projected']['dx']:+.2f},{r_['left_eye']['projected']['dy']:+.2f}) "
                f"R proj ({r_['right_eye']['projected']['dx']:+.2f},{r_['right_eye']['projected']['dy']:+.2f})")
    # independent re-measurement of the saved validation PNGs (no re-render)
    dd = os.path.join(ROOT, "reports", "direction_test")
    n_meas = dv["neutral_pupils"]
    expect = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
    ok_all, det = True, {}
    for name, (ex, ey) in expect.items():
        luma = imgutil.load_rgb(os.path.join(dd, f"{name}.png")) @ np.array([0.2126, 0.7152, 0.0722])
        for s in ("L", "R"):
            c, _ = eyes.detect_pupil_in_image(luma, meta[0]["extra"][f"eye_center_2D_{s}"], n_meas[s]["roi_radius_px"],
                                              pupil_radius_px=n_meas[s].get("pupil_radius_px"))
            dx, dy = c[0] - n_meas[s]["detected"][0], c[1] - n_meas[s]["detected"][1]
            ok_all &= (dx * ex + dy * ey) >= 2.0
            det[f"{name}_{s}"] = (round(dx, 2), round(dy, 2))
    S.check(t, "re-measured from saved PNGs: all 8 pupil shifts have the expected sign and >= 2 px", ok_all, det)

    # ------------------------------------------------------------------ G. sequence
    t = S.test("G", "Ten-frame clockwise sequence")
    files = [os.path.join(ROOT, "renders", f"frame_{k:02d}.png") for k in range(10)]
    S.check(t, "frame_00..frame_09.png exist", all(os.path.isfile(f) for f in files))
    frames = [imgutil.load_rgb(f) for f in files]
    S.check(t, "all frames 1024x1024", all(f.shape[:2] == (1024, 1024) for f in frames))
    S.check(t, "inputs follow theta_k = 2 pi k/10, h = sin, v = cos",
            all(abs(m["theta"] - 2 * math.pi * k / 10) < 1e-12 and abs(m["input"]["h"] - math.sin(m["theta"])) < 1e-12
                and abs(m["input"]["v"] - math.cos(m["theta"])) < 1e-12 for k, m in enumerate(meta)))
    S.check(t, "frame 0 is the top position (h=0, v=+1)", abs(meta[0]["input"]["h"]) < 1e-12 and meta[0]["input"]["v"] == 1.0)
    diffs = [float(np.abs(frames[i] - frames[j]).max()) for i in range(10) for j in range(i + 1, 10)]
    S.check(t, "no two frames identical", min(diffs) > 0.02, f"min pairwise max-diff {min(diffs)*255:.1f}/255")
    # rendered pupils trace the clock: direction of the detected displacement vs the sequence centroid
    cen = {s: np.mean([m["extra"]["qa"][f"pupil_center_2D_detected_{s}"] for m in meta], axis=0) for s in ("L", "R")}
    ang_err = []
    for m in meta:
        for s in ("L", "R"):
            d = np.array(m["extra"]["qa"][f"pupil_center_2D_detected_{s}"]) - cen[s]
            a = math.atan2(d[0], -d[1]) % (2 * math.pi)          # clockwise from 12 o'clock
            ang_err.append(abs((a - m["theta"] + math.pi) % (2 * math.pi) - math.pi))
    S.check(t, "pupils in the RENDERED images move clockwise from the top (image-measured angle vs theta_k)",
            max(ang_err) < math.radians(20), f"max angular error {math.degrees(max(ang_err)):.1f} deg")
    S.check(t, "contact sheet exists", os.path.isfile(os.path.join(ROOT, "renders", "contact_sheet.png")))

    # ------------------------------------------------------------------ H. metadata
    t = S.test("H", "Metadata ground truth")
    S.check(t, "valid JSON list with 10 records, frames 0..9", isinstance(meta, list) and [m["frame"] for m in meta] == list(range(10)))
    S.check(t, "required field names present", all(all(f in m for f in REQUIRED_FIELDS) and set(m["input"]) == {"h", "v"} for m in meta))

    def finite(x):
        if isinstance(x, (list, tuple)):
            return all(finite(i) for i in x)
        if isinstance(x, dict):
            return all(finite(i) for i in x.values())
        if isinstance(x, (int, float)):
            return math.isfinite(x)
        return True
    S.check(t, "all numeric values finite", all(finite(m) for m in meta))
    cal = e.cal
    Mn = np.array(cal["eye_midpoint"])
    F0n = Mn + cal["fixation_distance_m"] * np.array(cal["head_forward_axis"])
    rot_err, pup3_err, pup2_err, P_err = 0.0, 0.0, 0.0, 0.0
    Pcur = np.array(e.camera_matrices()["P"])
    for m in meta:
        T = F0n + m["input"]["h"] * cal["max_offset_h_m"] * np.array(cal["head_right_axis_image"]) + \
            m["input"]["v"] * cal["max_offset_v_m"] * np.array(cal["head_up_axis"])
        P = np.array(m["camera_matrix"])
        P_err = max(P_err, float(np.abs(P - Pcur).max()))
        for s in ("L", "R"):
            C = np.array(cal[f"eye_center_{s}"])
            Rexp = shortest_arc(np.array([0, -1.0, 0]), T - C)
            rot_err = max(rot_err, float(np.abs(euler_xyz_to_matrix(m[f"eye_rotation_{s}"]) - Rexp).max()))
            p3 = C + Rexp @ np.array([0, -cal["pupil_offset_local_m"], 0])
            pup3_err = max(pup3_err, float(np.linalg.norm(p3 - np.array(m["extra"][f"pupil_center_3D_{s}"]))))
            x = P @ np.append(p3, 1.0)
            pup2_err = max(pup2_err, math.dist(x[:2] / x[2], m[f"pupil_center_2D_{s}"]))
    S.check(t, "eye rotations match independent shortest-arc re-derivation", rot_err < 1e-6, f"max matrix diff {rot_err:.1e}")
    S.check(t, "3-D pupil centres match independent re-derivation", pup3_err < TOL_M, f"{pup3_err:.1e} m")
    S.check(t, "2-D pupil centres = camera_matrix x independent 3-D pupil", pup2_err < 1e-3, f"{pup2_err:.1e} px")
    S.check(t, "camera_matrix equals the live camera projection", P_err < 1e-6, f"{P_err:.1e}")
    if ref_meta:
        with open(ref_meta) as fh:
            ref = json.load(fh)
        keys = ["theta", "eye_rotation_L", "eye_rotation_R", "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix"]

        def fl(x):
            return [v for i in x for v in fl(i)] if isinstance(x, list) else [x]
        dmax = max(abs(a - b) for m, r in zip(meta, ref) for k in keys for a, b in zip(fl(m[k]), fl(r[k])))
        S.check(t, "ground truth identical to the reference release (preservation check)",
                len(ref) == len(meta) and dmax == 0.0, f"max |diff| over all frames/fields = {dmax:.1e}")
    S.check(t, "2-D pupils inside the image", all(0 <= c < 1024 for m in meta for s in ("L", "R") for c in m[f"pupil_center_2D_{s}"]))

    # ------------------------------------------------------------------ I. packaging
    t = S.test("I", "Packaging")
    if zip_path is None:
        cands = sorted(glob.glob(os.path.join(os.path.dirname(ROOT), "*.zip")))
        zip_path = cands[-1] if cands else None
    S.check(t, "submission ZIP found", zip_path and os.path.isfile(zip_path), os.path.basename(zip_path or "") or None)
    if zip_path and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            names = set(z.namelist())
            S.check(t, "ZIP opens and CRCs verify", bad is None)
            miss = [p for p in REQUIRED_ZIP_PATHS if p not in names]
            S.check(t, "required paths at ZIP root (no enclosing folder)", not miss, f"missing={miss}")
            tmpd = _tmpdir(prefix="unzipped_")
            z.extractall(tmpd)
        with open(os.path.join(tmpd, "metadata.json")) as fh:
            zmeta = json.load(fh)
        # run the PACKAGED wrapper in a fresh interpreter against the PACKAGED scene
        k = 3
        h, v = zmeta[k]["input"]["h"], zmeta[k]["input"]["v"]
        out_png = os.path.join(tmpd, "probe.png")
        cmd = [sys.executable, os.path.join(tmpd, "wrapper", "eyes.py"), "--move", repr(h), repr(v),
               "--samples", "2", "--out", out_png]
        if not bpy.app.binary_path.endswith(("python", "python3", "python3.11")) and "blender" in os.path.basename(bpy.app.binary_path).lower():
            cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python", os.path.join(tmpd, "wrapper", "eyes.py"), "--",
                   "--move", repr(h), repr(v), "--samples", "2", "--out", out_png]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        txt = res.stdout
        js = None
        if "{\n" in txt:
            js, _ = json.JSONDecoder().raw_decode(txt[txt.index("{\n"):])
        S.check(t, "packaged wrapper runs headless against packaged scene", res.returncode == 0 and js and os.path.isfile(out_png),
                " ".join(os.path.basename(c) for c in cmd[:2]) + f" ... rc={res.returncode}")
        if js:
            dmax = max(math.dist(js["metadata"][f"pupil_center_2D_{s}"], zmeta[k][f"pupil_center_2D_{s}"]) for s in ("L", "R"))
            S.check(t, "packaged run reproduces metadata.json frame 3 pupils", dmax < 1e-6, f"{dmax:.1e} px")
        bpy.ops.wm.open_mainfile(filepath=os.path.join(tmpd, "scene.blend"))
        unp = [i.name for i in bpy.data.images if i.packed_file is None and i.source == "FILE"]
        S.check(t, "packaged scene self-contained (all images packed, no libraries)", not unp and not bpy.data.libraries)

    # ------------------------------------------------------------------ report
    for tt in S.tests:
        tt["status"] = "PASS" if all(c["pass"] for c in tt["checks"]) else "FAIL"
    summary = {"passed": sum(tt["status"] == "PASS" for tt in S.tests), "total": len(S.tests),
               "checks_passed": sum(c["pass"] for tt in S.tests for c in tt["checks"]),
               "checks_total": sum(len(tt["checks"]) for tt in S.tests),
               "runtime_s": round(time.perf_counter() - t0, 2), "blender_version": bpy.app.version_string,
               "tests": S.tests}
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "acceptance_report.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    L = ["# Acceptance test report", "",
         f"**{summary['passed']}/{summary['total']} tests PASS** ({summary['checks_passed']}/{summary['checks_total']} checks). "
         f"Blender {summary['blender_version']}, runtime {summary['runtime_s']} s. "
         "Generated by `wrapper/run_acceptance_tests.py`.", "",
         "Tolerances: Blender stores transforms in float32 (~1e-7 relative), so geometric equalities are checked "
         "at 1 µm / 1e-6 relative — four orders of magnitude below one rendered pixel (≈0.22 mm at the eye).", ""]
    for tt in S.tests:
        L += [f"## Test {tt['id']} — {tt['title']}: **{tt['status']}**", "", "| Check | Result | Measured |", "|---|---|---|"]
        for c in tt["checks"]:
            d = c["detail"].replace("|", "/")
            L.append(f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} | {d[:220]} |")
        L.append("")
    with open(os.path.join(ROOT, "reports", "acceptance_report.md"), "w") as fh:
        fh.write("\n".join(L))
    print(f"[acceptance] {summary['passed']}/{summary['total']} tests, {summary['checks_passed']}/{summary['checks_total']} checks")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
