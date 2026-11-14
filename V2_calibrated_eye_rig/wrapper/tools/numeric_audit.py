#!/usr/bin/env python
"""
numeric_audit.py -- numerical audit of the GazeForge Track-1 package.

Usage:
    /path/to/python_with_bpy numeric_audit.py <package_dir> [--json out.json] [--no-wrapper] [--no-images]

* All gaze / projection maths is done here in numpy (no mathutils).
* bpy is used only to READ the scene (calibration JSON, bone rest matrices, object transforms,
  camera data) and, for item 4, to evaluate the depsgraph after this script writes a pose that
  it computed itself; bpy_extras.world_to_camera_view is used only as a cross-check.
* The package's own eyes.py is imported only (a) to compare its move() with this script and
  (b) to unit-test its image detector convention. --no-wrapper skips that.
* PNGs are decoded by a small built-in zlib/numpy decoder (no PIL, no Blender colour management).
* Nothing is written into <package_dir> (bytecode writing is disabled; the .blend is never saved).

Prints PASS/FAIL lines and a final JSON summary. Exit code 0 iff all checks pass.
"""
import json
import math
import os
import sys
import zlib

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ARGS = [a for a in sys.argv[1:]]
if "--" in sys.argv:
    ARGS = sys.argv[sys.argv.index("--") + 1:]
if not ARGS or ARGS[0].startswith("--"):
    print(__doc__)
    sys.exit(2)
PKG = os.path.abspath(ARGS[0])
JSON_OUT = ARGS[ARGS.index("--json") + 1] if "--json" in ARGS else None
USE_WRAPPER = "--no-wrapper" not in ARGS
USE_IMAGES = "--no-images" not in ARGS

import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402

RESULTS = []          # (name, ok, detail)
SUMMARY = {}


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def deg(x):
    return float(np.degrees(x))


# ======================================================================================
# numpy rotation helpers (quaternions are w, x, y, z; active rotations, column vectors)
# ======================================================================================
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_to_R(q):
    w, x, y, z = unit(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_q(R):
    """Shepperd's method, w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def axis_angle_q(axis, ang):
    a = unit(axis)
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * a])


def shortest_arc(a, d):
    """Rotation (quaternion) taking unit a onto unit d with axis a x d (Rodrigues)."""
    a, d = unit(a), unit(d)
    c = np.cross(a, d)
    s = np.linalg.norm(c)
    ang = math.atan2(s, float(np.dot(a, d)))
    if s < 1e-15:
        return np.array([1.0, 0, 0, 0])
    return axis_angle_q(c / s, ang)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_xyz_to_R(e):
    """Blender Euler 'XYZ': R = Rz(ez) @ Ry(ey) @ Rx(ex) (extrinsic X, then Y, then Z)."""
    return Rz(e[2]) @ Ry(e[1]) @ Rx(e[0])


def R_to_euler_xyz(R):
    return np.array([math.atan2(R[2, 1], R[2, 2]),
                     math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])),
                     math.atan2(R[1, 0], R[0, 0])])


def rot_angle_between(Ra, Rb):
    """Angle of Ra^T Rb. Uses ||Ra - Rb||_F = 2*sqrt(2)*sin(angle/2), which (unlike acos of the
    trace) stays accurate for tiny angles between float32-derived matrices."""
    x = np.linalg.norm(np.asarray(Ra, float) - np.asarray(Rb, float)) / (2 * math.sqrt(2))
    return 2 * math.asin(min(1.0, x))


def fick(R):
    """R = Rz(theta) Rx(phi) Ry(psi): theta = horizontal (about head-fixed up Z), phi = vertical
    about the once-rotated interaural axis (+ = DOWN), psi = torsion about the line of sight
    (rotation about eye +Y; + moves the 12 o'clock meridian toward world +X)."""
    phi = math.asin(max(-1, min(1, R[2, 1])))
    psi = math.atan2(-R[2, 0], R[2, 2])
    theta = math.atan2(-R[0, 1], R[1, 1])
    return theta, phi, psi


def helmholtz(R):
    """R = Rx(phi) Rz(theta) Ry(psi): vertical (head-fixed interaural axis) outermost."""
    theta = math.asin(max(-1, min(1, -R[0, 1])))
    psi = math.atan2(R[0, 2], R[0, 0])
    phi = math.atan2(R[2, 1], R[1, 1])
    return theta, phi, psi


def M4(m):
    return np.array([[float(m[i][j]) for j in range(4)] for i in range(4)])


# ======================================================================================
# minimal PNG decoder (8/16-bit, grey/RGB/RGBA, non-interlaced)
# ======================================================================================
def load_png_luma(path, max_rows=None):
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, W = 8, [], None
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            W, H = int.from_bytes(body[0:4], "big"), int.from_bytes(body[4:8], "big")
            bitd, ctype, interlace = body[8], body[9], body[12]
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"IEND":
            break
        pos += 12 + n
    assert interlace == 0 and bitd in (8, 16) and ctype in (0, 2, 6)
    ch = {0: 1, 2: 3, 6: 4}[ctype]
    bpp = ch * bitd // 8
    stride = W * bpp
    raw = zlib.decompress(b"".join(idat))
    rows = H if max_rows is None else min(H, max_rows)
    out = np.zeros((rows, stride), np.uint8)
    prev = bytearray(stride)
    for y in range(rows):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if f == 1:
            arr = np.frombuffer(bytes(line), np.uint8).reshape(-1, bpp)
            line = bytearray(np.cumsum(arr, axis=0, dtype=np.uint8).tobytes())
        elif f == 2:
            line = bytearray(((np.frombuffer(bytes(line), np.uint8).astype(np.uint16)
                               + np.frombuffer(bytes(prev), np.uint8)) & 255).astype(np.uint8).tobytes())
        elif f == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out[y] = np.frombuffer(bytes(line), np.uint8)
        prev = line
    if bitd == 16:
        px = out.view(">u2").astype(np.float64) / 65535.0
    else:
        px = out.astype(np.float64) / 255.0
    px = px.reshape(rows, W, ch)
    if ch == 1:
        return px[..., 0]
    return 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]


# ======================================================================================
# independent pupil detector (soft-threshold centroid, glint fill by radius) --
# coordinates: continuous, origin = top-left image corner, pixel (i, j) centre = (i+0.5, j+0.5)
# ======================================================================================
def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def my_detect(luma, roi_xy, roi_r, rp_px):
    H, W = luma.shape
    cx, cy = roi_xy
    R = int(math.ceil(roi_r)) + 2
    x0, x1 = max(0, int(cx) - R), min(W, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(H, int(cy) + R + 1)
    sub = luma[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    # 1) coarse: minimum of the pupil-sized box average inside the iris ROI
    sm = box_blur(sub, max(1, int(round(rp_px * 0.6))))
    inroi = (xx - cx) ** 2 + (yy - cy) ** 2 <= (roi_r - rp_px) ** 2
    sm = np.where(inroi, sm, np.inf)
    iy, ix = np.unravel_index(np.argmin(sm), sm.shape)
    c = np.array([xx[iy, ix], yy[iy, ix]])
    # 2) refine: sub-pixel boundary points along 90 rays + robust algebraic circle fit (Kasa).
    #    Each ray uses its own half-way threshold between the pupil level and the level just
    #    outside the pupil on that ray, so an adjacent dark lid margin does not drag the centre,
    #    and rays start at 0.45 r_p so corneal glints inside the pupil are ignored.
    def sample(px, py):                               # bilinear, continuous corner-origin coords
        fx_, fy_ = px - 0.5, py - 0.5
        ix0 = np.clip(np.floor(fx_).astype(int), 0, W - 2)
        iy0 = np.clip(np.floor(fy_).astype(int), 0, H - 2)
        ax, ay = fx_ - ix0, fy_ - iy0
        return ((1 - ax) * (1 - ay) * luma[iy0, ix0] + ax * (1 - ay) * luma[iy0, ix0 + 1]
                + (1 - ax) * ay * luma[iy0 + 1, ix0] + ax * ay * luma[iy0 + 1, ix0 + 1])

    angs = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rr = np.arange(0.45 * rp_px, 1.9 * rp_px, 0.05)
    keep = None
    for _ in range(3):
        X = c[0] + np.cos(angs)[:, None] * rr[None, :]
        Y = c[1] + np.sin(angs)[:, None] * rr[None, :]
        prof = sample(X, Y)
        Ip = float(np.median(prof[:, rr < 0.7 * rp_px]))
        edges = []
        for i in range(len(angs)):
            Io = float(np.median(prof[i, (rr > 1.15 * rp_px) & (rr < 1.45 * rp_px)]))
            thr = 0.5 * (Ip + Io)
            above = np.nonzero(prof[i] > thr)[0]
            if Io - Ip < 0.05 or above.size == 0 or above[0] == 0:
                edges.append(np.nan)
                continue
            j = above[0]
            t = (thr - prof[i, j - 1]) / (prof[i, j] - prof[i, j - 1])
            edges.append(rr[j - 1] + t * (rr[j] - rr[j - 1]))
        edges = np.array(edges)
        ok = np.isfinite(edges)
        ex_all = c[0] + np.cos(angs) * np.where(ok, edges, 0)
        ey_all = c[1] + np.sin(angs) * np.where(ok, edges, 0)
        sel = ok.copy()
        for _it in range(8):                          # trimmed LS: keep the 70 % best-fitting rays
            Am = np.column_stack([ex_all[sel], ey_all[sel], np.ones(sel.sum())])
            sol_ = np.linalg.lstsq(Am, ex_all[sel] ** 2 + ey_all[sel] ** 2, rcond=None)[0]
            cc_ = np.array([sol_[0] / 2, sol_[1] / 2])
            Rf = math.sqrt(sol_[2] + cc_ @ cc_)
            res = np.abs(np.hypot(ex_all - cc_[0], ey_all - cc_[1]) - Rf)
            res[~ok] = np.inf
            sel = res <= np.quantile(res[ok], 0.70)
        c = cc_
        keep = int(sel.sum())
    return c, keep


# ======================================================================================
# load scene (read-only)
# ======================================================================================
scene_path = os.path.join(PKG, "scene.blend")
bpy.ops.wm.open_mainfile(filepath=scene_path)
scene = bpy.context.scene
rig = bpy.data.objects["EyeRig"]
cal = json.loads(rig["calibration_json"])
meta = json.load(open(os.path.join(PKG, "metadata.json")))
SIDES = ("L", "R")

rig_mw = M4(rig.matrix_world)
for pb in rig.pose.bones:                         # pose bones to rest (we never save the file)
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
    pb.scale = (1, 1, 1)
bpy.context.view_layer.update()

bone_local = {s: M4(rig.data.bones[f"eye.{s}"].matrix_local) for s in SIDES}
head_local = M4(rig.data.bones["head"].matrix_local)
rest_world = {s: rig_mw @ bone_local[s] for s in SIDES}      # head pose = identity (checked below)

# ---------------------------------------------------------------- item 2: eye centres
print("\n== 2. Eye centres ==")
C_bone = {s: rest_world[s][:3, 3] for s in SIDES}
C_ball = {s: M4(bpy.data.objects[f"Eyeball.{s}"].matrix_world)[:3, 3] for s in SIDES}
C_cal = {s: np.array(cal[f"eye_center_{s}"]) for s in SIDES}
d_bc = max(np.linalg.norm(C_bone[s] - C_cal[s]) for s in SIDES)
d_eb = max(np.linalg.norm(C_ball[s] - C_bone[s]) for s in SIDES)
check("eye centres: bone heads == calibration JSON", d_bc < 1e-6, f"max {d_bc:.2e} m")
check("eye centres: Eyeball.L/R world origin == bone heads", d_eb < 1e-6, f"max {d_eb:.2e} m")
C = C_bone
M = 0.5 * (C["L"] + C["R"])
ipd = float(np.linalg.norm(C["L"] - C["R"]))
check("midpoint == calibration eye_midpoint", np.linalg.norm(M - np.array(cal["eye_midpoint"])) < 1e-6,
      f"M = {np.round(M, 7).tolist()}")
check("IPD (|C_L - C_R|) == 58.7 mm (README)", abs(ipd * 1000 - 58.7) < 0.05, f"IPD = {ipd*1000:.3f} mm")
sym = float(np.linalg.norm(C["L"] * [-1, 1, 1] - C["R"]))
check("eye centres mirror-symmetric about x = 0, same y/z", sym < 1e-7, f"asym {sym:.1e} m")
SUMMARY["eye_centres"] = {"C_L": C["L"].tolist(), "C_R": C["R"].tolist(), "M": M.tolist(), "IPD_mm": ipd * 1000}
head_basis_ok = all(np.allclose(M4(pb.matrix_basis), np.eye(4)) for pb in rig.pose.bones)
check("rig world matrix identity-like & head pose = rest", np.allclose(rig_mw, np.eye(4)) and head_basis_ok, "")

# ---------------------------------------------------------------- head frame
f = unit(cal["head_forward_axis"])
r = unit(cal["head_right_axis_image"])
u = unit(cal["head_up_axis"])
# derive the head frame independently from the head bone (rest): bone +Y = up, bone +Z = forward
f_bone, u_bone = unit(head_local[:3, 2]), unit(head_local[:3, 1])
check("head frame: cal axes orthonormal (r x u = f) and match the head bone (+Z = f, +Y = u)",
      abs(np.dot(f, r)) < 1e-12 and abs(np.dot(f, u)) < 1e-12 and abs(np.dot(r, u)) < 1e-12
      and np.allclose(np.cross(r, u), f) and np.allclose(f_bone, f, atol=1e-6) and np.allclose(u_bone, u, atol=1e-6),
      f"f={f.tolist()} r={r.tolist()} u={u.tolist()} ; head-bone +Z={np.round(f_bone,6).tolist()} +Y={np.round(u_bone,6).tolist()}")

# primary-position optical axis of each eye: bone +Y in world (rest)
A = {s: unit(rest_world[s][:3, 1]) for s in SIDES}
check("bone +Y (primary axis) == head forward for both eyes",
      max(np.linalg.norm(A[s] - f) for s in SIDES) < 1e-6,
      f"A_L={np.round(A['L'],9).tolist()} roll: bone X={np.round(rest_world['L'][:3,0],6).tolist()} Z={np.round(rest_world['L'][:3,2],6).tolist()}")


# ---------------------------------------------------------------- item 1: target
def target(h, v):
    return M + cal["fixation_distance_m"] * f + h * cal["max_offset_h_m"] * r + v * cal["max_offset_v_m"] * u


print("\n== 1. Fixation target ==")
F0 = M + 0.35 * f
check("calibration constants are 0.35 / 0.40 / 0.05 / 0.05 m",
      (cal["fixation_distance_m"], cal["camera_distance_m"], cal["max_offset_h_m"], cal["max_offset_v_m"])
      == (0.35, 0.40, 0.05, 0.05), "")
errT, plane, offs = 0.0, 0.0, []
for rec in meta:
    h, v = rec["input"]["h"], rec["input"]["v"]
    T = target(h, v)
    errT = max(errT, float(np.linalg.norm(T - np.array(rec["extra"]["fixation_target"]))))
    plane = max(plane, abs(float(np.dot(T - M, f)) - 0.35))
    offs.append((float(np.dot(T - F0, r)), float(np.dot(T - F0, u))))
check("T(h,v) recomputed == metadata extra.fixation_target", errT < 1e-6, f"max {errT:.2e} m")
check("T lies in the 35 cm plane: (T-M).f = 0.35", plane < 1e-6, f"max |dev| {plane:.2e} m")
mx = max(max(abs(a), abs(b)) for a, b in offs)
check("|offsets| <= 5 cm per axis", mx <= 0.05 + 1e-9, f"max |offset| {mx*100:.4f} cm")
ext_corners = [np.linalg.norm(target(hh, vv) - F0) for hh in (-1, 1) for vv in (-1, 1)]
SUMMARY["target"] = {"F0": F0.tolist(), "max_err_vs_metadata_m": errT, "plane_dev_m": plane,
                     "corner_radial_offset_cm": float(max(ext_corners) * 100)}

# ---------------------------------------------------------------- item 3: rotations
print("\n== 3. Binocular rotations ==")
SEQ = [(rec["frame"], rec["input"]["h"], rec["input"]["v"]) for rec in meta]
EXT = [("LEFT", -1.0, 0.0), ("RIGHT", 1.0, 0.0), ("UP", 0.0, 1.0), ("DOWN", 0.0, -1.0), ("NEUTRAL", 0.0, 0.0)]


def solve(h, v):
    T = target(h, v)
    out = {"T": T}
    for s in SIDES:
        d = unit(T - C[s])
        qw = shortest_arc(A[s], d)
        Rw = q_to_R(qw)
        g = Rw @ A[s]
        tt = T - C[s]
        miss = float(np.linalg.norm(tt - np.dot(tt, g) * g))
        yaw = math.atan2(np.dot(g, r), np.dot(g, f))
        pitch = math.asin(np.dot(g, u))
        out[s] = dict(q=qw, R=Rw, g=g, miss=miss, yaw=yaw, pitch=pitch, ang=2 * math.acos(min(1, qw[0])),
                      adduction=(-yaw if np.dot(C[s] - M, r) > 0 else yaw))
    out["verg"] = math.acos(max(-1, min(1, float(np.dot(out["L"]["g"], out["R"]["g"])))))
    return out


# sequence sampling
ok_seq = all(abs(rec["theta"] - 2 * math.pi * k / 10) < 1e-12 and abs(rec["input"]["h"] - math.sin(rec["theta"])) < 1e-12
             and abs(rec["input"]["v"] - math.cos(rec["theta"])) < 1e-12 for k, rec in enumerate(meta)) and len(meta) == 10
check("sequence: 10 frames, theta_k = 2*pi*k/10, h = sin, v = cos, frame 0 = top (h=0, v=1)",
      ok_seq and meta[0]["input"]["v"] == 1.0, "")

worst_q, worst_e, worst_miss, rows = 0.0, 0.0, 0.0, []
for rec in meta:
    k, h, v = rec["frame"], rec["input"]["h"], rec["input"]["v"]
    sol = solve(h, v)
    for s in SIDES:
        q_meta = np.array(rec["extra"][f"eye_rotation_quat_wxyz_{s}"])
        R_meta_q = q_to_R(q_meta)
        R_meta_e = euler_xyz_to_R(rec[f"eye_rotation_{s}"])
        worst_q = max(worst_q, rot_angle_between(sol[s]["R"], R_meta_q))
        worst_e = max(worst_e, rot_angle_between(sol[s]["R"], R_meta_e))
        worst_miss = max(worst_miss, sol[s]["miss"])
    rows.append((k, h, v, sol))
check("numpy shortest-arc rotation == metadata quaternion (all 20 eyes)", worst_q < 2e-6, f"max {worst_q:.2e} rad")
check("numpy rotation == metadata Euler XYZ with R = Rz.Ry.Rx", worst_e < 2e-6, f"max {worst_e:.2e} rad")
# Euler order sensitivity: does an intrinsic-XYZ (Rx.Ry.Rz) reading also match? (it should NOT)
alt = max(rot_angle_between(rows[i][3][s]["R"], Rx(meta[i][f"eye_rotation_{s}"][0]) @ Ry(meta[i][f"eye_rotation_{s}"][1]) @ Rz(meta[i][f"eye_rotation_{s}"][2]))
          for i in range(10) for s in SIDES)
print(f"      info: reading the same Euler triple as Rx.Ry.Rz would be wrong by up to {deg(alt):.4f} deg")
check("gaze rays (numpy) pass through T", worst_miss < 1e-9, f"max miss {worst_miss:.2e} m")
mmiss = max(rec["extra"][f"gaze_ray_miss_m_{s}"] for rec in meta for s in SIDES)
check("metadata gaze_ray_miss_m < 1 um", mmiss < 1e-6, f"max {mmiss:.2e} m")

print("      frame   h      v     yawL    yawR   pitchL  pitchR  addL   addR   |rotL|  |rotR|  verg  (deg)")
seq_table = []
for k, h, v, sol in rows:
    L, Rr = sol["L"], sol["R"]
    seq_table.append(dict(frame=k, h=h, v=v, yaw_L=deg(L["yaw"]), yaw_R=deg(Rr["yaw"]), pitch_L=deg(L["pitch"]),
                          pitch_R=deg(Rr["pitch"]), add_L=deg(L["adduction"]), add_R=deg(Rr["adduction"]),
                          rot_L=deg(L["ang"]), rot_R=deg(Rr["ang"]), vergence=deg(sol["verg"])))
    print(f"      {k:3d} {h:+.3f} {v:+.3f} {deg(L['yaw']):+7.3f} {deg(Rr['yaw']):+7.3f} {deg(L['pitch']):+7.3f} "
          f"{deg(Rr['pitch']):+7.3f} {deg(L['adduction']):6.3f} {deg(Rr['adduction']):6.3f} {deg(L['ang']):7.3f} "
          f"{deg(Rr['ang']):7.3f} {deg(sol['verg']):6.3f}")
ext_table = {}
for name, h, v in EXT:
    sol = solve(h, v)
    ext_table[name] = dict(h=h, v=v, **{f"{key}_{s}": deg(sol[s][key]) for s in SIDES
                                        for key in ("yaw", "pitch", "adduction", "ang")}, vergence=deg(sol["verg"]),
                           miss_max=max(sol[s]["miss"] for s in SIDES))
    e_ = ext_table[name]
    print(f"      {name:7s} ({h:+.0f},{v:+.0f}) yaw L {e_['yaw_L']:+.3f} R {e_['yaw_R']:+.3f} | pitch {e_['pitch_L']:+.3f}/{e_['pitch_R']:+.3f}"
          f" | adduction L {e_['adduction_L']:+.3f} R {e_['adduction_R']:+.3f} | vergence {e_['vergence']:.3f}")
check("extremes: gaze rays through T", max(e["miss_max"] for e in ext_table.values()) < 1e-9, "")
SUMMARY["rotations"] = {"max_quat_err_rad": worst_q, "max_euler_err_rad": worst_e, "max_miss_m": worst_miss,
                        "sequence": seq_table, "extremes": ext_table}

# README numbers
n = ext_table["NEUTRAL"]
check("README '4.79 deg inward' at neutral", abs(n["adduction_L"] - 4.79) < 0.005 and abs(n["adduction_R"] - 4.79) < 0.005,
      f"{n['adduction_L']:.4f} / {n['adduction_R']:.4f} deg")
check("README '+-8.1 deg elevation' at (0,+-1)", abs(ext_table["UP"]["pitch_L"] - 8.1) < 0.05 and abs(ext_table["DOWN"]["pitch_L"] + 8.1) < 0.05,
      f"UP {ext_table['UP']['pitch_L']:.4f}, DOWN {ext_table['DOWN']['pitch_L']:.4f}")
yawsL = [ext_table[nm]["yaw_L"] for nm in ("LEFT", "RIGHT")]
yawsR = [ext_table[nm]["yaw_R"] for nm in ("LEFT", "RIGHT")]
print(f"      eye-in-head azimuth range: L eye [{min(yawsL):+.3f}, {max(yawsL):+.3f}]  R eye [{min(yawsR):+.3f}, {max(yawsR):+.3f}] deg (+ = image right)")
addm = max(ext_table[nm][f"adduction_{s}"] for nm in ("LEFT", "RIGHT") for s in SIDES)
abdm = -min(ext_table[nm][f"adduction_{s}"] for nm in ("LEFT", "RIGHT") for s in SIDES)
check("README '-12.8...+3.4 deg per-eye azimuth' == 12.8 adduction / 3.4 abduction",
      abs(addm - 12.8) < 0.05 and abs(abdm - 3.4) < 0.05,
      f"max adduction {addm:.3f}, max abduction {abdm:.3f} deg; NOTE the signed range is mirrored for the R eye")
grid = np.linspace(-1, 1, 201)
maxpitch = max(abs(solve(hh, 1.0)["L"]["pitch"]) for hh in grid)
print(f"      max |elevation| over the full input square: {deg(maxpitch):.3f} deg (at h where the eye is straight below/above T)")
SUMMARY["readme_numbers"] = {"inward_deg": n["adduction_L"], "elev_deg": ext_table["UP"]["pitch_L"],
                             "max_adduction_deg": addm, "max_abduction_deg": abdm, "max_elev_square_deg": deg(maxpitch)}

# ---------------------------------------------------------------- item 4: conjugation & Blender evaluation
print("\n== 4. Quaternion conjugation / bone space ==")
# rest eyeball world rotation must be identity for 'eyeball world rotation == rotation from primary'
R_ball_rest = {s: M4(bpy.data.objects[f"Eyeball.{s}"].matrix_world)[:3, :3] for s in SIDES}
d_rest = max(rot_angle_between(R_ball_rest[s], np.eye(3)) for s in SIDES)
check("Eyeball world rotation at rest pose == identity (so metadata = rotation from primary)", d_rest < 1e-6,
      f"{d_rest:.2e} rad")
# generic algebraic check with arbitrary rest frames (roll != 0): posed = R_rest @ R(q_local) must equal R(q_world) @ R_rest
rng = np.random.default_rng(1)
worst_alg, worst_wrong = 0.0, 0.0
for _ in range(200):
    q_rest = unit(rng.normal(size=4))
    q_world = unit(rng.normal(size=4))
    q_loc = q_mul(q_mul(q_conj(q_rest), q_world), q_rest)
    worst_alg = max(worst_alg, np.abs(q_to_R(q_rest) @ q_to_R(q_loc) - q_to_R(q_world) @ q_to_R(q_rest)).max())
    q_bad = q_mul(q_mul(q_rest, q_world), q_conj(q_rest))
    worst_wrong = max(worst_wrong, np.abs(q_to_R(q_rest) @ q_to_R(q_bad) - q_to_R(q_world) @ q_to_R(q_rest)).max())
check("q_local = q_rest^-1 q_world q_rest is the correct conjugation (random rest frames incl. roll)",
      worst_alg < 1e-12 and worst_wrong > 0.1, f"err {worst_alg:.1e}; reversed order err {worst_wrong:.2f}")
q_rest_rig = {s: R_to_q(rest_world[s][:3, :3]) for s in SIDES}
print(f"      rig rest quaternions: L {np.round(q_rest_rig['L'], 9).tolist()}  (180 deg about Z: both "
      f"conjugation orders coincide for THIS rig, so the rig cannot distinguish them)")
# Blender evaluation with OUR q_local
worst_ball, worst_ctr, worst_pup3d = 0.0, 0.0, 0.0
pupil_off = cal["pupil_offset_local_m"]
pupil_local = M4(bpy.data.objects["Pupil.L"].matrix_local)
for k, h, v, sol in rows + [(nm, hh, vv, solve(hh, vv)) for nm, hh, vv in EXT]:
    for s in SIDES:
        q_loc = q_mul(q_mul(q_conj(q_rest_rig[s]), sol[s]["q"]), q_rest_rig[s])
        pb = rig.pose.bones[f"eye.{s}"]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = tuple(q_loc)
    bpy.context.view_layer.update()
    for s in SIDES:
        mw = M4(bpy.data.objects[f"Eyeball.{s}"].matrix_world)
        worst_ball = max(worst_ball, rot_angle_between(mw[:3, :3], sol[s]["R"]))
        worst_ctr = max(worst_ctr, float(np.linalg.norm(mw[:3, 3] - C[s])))
        P3 = C[s] + sol[s]["R"] @ np.array([0.0, -pupil_off, 0.0])
        Pb = M4(bpy.data.objects[f"Pupil.{s}"].matrix_world)[:3, 3]
        worst_pup3d = max(worst_pup3d, float(np.linalg.norm(P3 - Pb)))
        sol[s]["P3"] = P3
check("Blender-evaluated Eyeball world rotation == numpy q_world (24 eyes)", worst_ball < 2e-6, f"max {worst_ball:.2e} rad")
check("eyeball centres never translate (rotation about own centre)", worst_ctr < 1e-6, f"max {worst_ctr:.2e} m")
check("Pupil object origin == C + R.[0,-pupil_offset,0]", worst_pup3d < 1e-6, f"max {worst_pup3d:.2e} m; offset {pupil_off*1000:.3f} mm")
for pb in rig.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
bpy.context.view_layer.update()

# ---------------------------------------------------------------- item 5: camera
print("\n== 5. Camera projection ==")
cam = bpy.data.objects["RenderCamera"]
cd = cam.data
rs = scene.render
W = rs.resolution_x * rs.resolution_percentage / 100.0
Hh = rs.resolution_y * rs.resolution_percentage / 100.0
fit = cd.sensor_fit
if fit == "AUTO":
    fit = "HORIZONTAL" if W * rs.pixel_aspect_x >= Hh * rs.pixel_aspect_y else "VERTICAL"
sensor = cd.sensor_width if fit == "HORIZONTAL" else cd.sensor_height
ref = W if fit == "HORIZONTAL" else Hh
fx = cd.lens / sensor * ref
fy = fx * rs.pixel_aspect_x / rs.pixel_aspect_y
K = np.array([[fx, 0, W / 2 - cd.shift_x * max(W, Hh)], [0, fy, Hh / 2 + cd.shift_y * max(W, Hh)], [0, 0, 1]])
cam_mw = M4(cam.matrix_world)
Rc2w = cam_mw[:3, :3] / np.linalg.norm(cam_mw[:3, :3], axis=0)
tc = cam_mw[:3, 3]
B2CV = np.diag([1.0, -1.0, -1.0])
Rcv = B2CV @ Rc2w.T
tcv = -Rcv @ tc
P = K @ np.hstack([Rcv, tcv[:, None]])
print(f"      lens {cd.lens} mm, sensor {cd.sensor_width}x{cd.sensor_height} fit {cd.sensor_fit}, shift ({cd.shift_x},{cd.shift_y}), "
      f"res {rs.resolution_x}x{rs.resolution_y}@{rs.resolution_percentage}%, pixel aspect {rs.pixel_aspect_x}:{rs.pixel_aspect_y}, type {cd.type}")
print(f"      K = {np.round(K, 4).tolist()}")
check("resolution 1024 x 1024 @100%, square pixels, perspective, no shift", (rs.resolution_x, rs.resolution_y, rs.resolution_percentage) == (1024, 1024, 100)
      and rs.pixel_aspect_x == rs.pixel_aspect_y and cd.type == "PERSP" and cd.shift_x == 0 and cd.shift_y == 0, "")
check("fx = 62/36*1024 = 1763.556 px (README)", abs(fx - 1763.5556) < 1e-3, f"fx {fx:.4f}")


def proj(P_, X):
    x = P_ @ np.append(X, 1.0)
    return x[:2] / x[2]


dP, dK, dW2C = 0.0, 0.0, 0.0
pts = [C[s] + rr for s in SIDES for rr in (np.zeros(3), np.array([0.01, -0.02, 0.005]), np.array([-0.03, -0.1, -0.02]))] + [M, F0]
for rec in meta:
    Pm = np.array(rec["camera_matrix"])
    for X in pts:
        dP = max(dP, float(np.linalg.norm(proj(P, X) - proj(Pm, X))))
    dK = max(dK, float(np.abs(np.array(rec["extra"]["camera_intrinsics_K"]) - K).max()))
for X in pts:
    co = world_to_camera_view(scene, cam, __import__("mathutils").Vector(X.tolist()))
    dW2C = max(dW2C, float(np.linalg.norm(proj(P, X) - np.array([co.x * W, (1 - co.y) * Hh]))))
check("rebuilt K == metadata camera_intrinsics_K", dK < 1e-3, f"max {dK:.1e} px")
check("rebuilt P == metadata camera_matrix (reprojection of test points)", dP < 1e-3, f"max {dP:.1e} px")
check("rebuilt P == bpy_extras world_to_camera_view * (W, H) with y flipped", dW2C < 1e-3, f"max {dW2C:.1e} px")
same_cam = max(float(np.abs(np.array(rec["camera_matrix"]) - np.array(meta[0]["camera_matrix"])).max()) for rec in meta)
check("camera_matrix identical in all 10 records (camera fixed)", same_cam == 0.0, f"max diff {same_cam}")
dvec = tc - M
along = float(np.dot(dvec, f))
lat = float(np.linalg.norm(dvec - along * f))
check("camera centre 0.40 m from M along head forward axis", abs(along - 0.40) < 1e-6 and lat < 1e-6,
      f"along {along:.9f} m, lateral {lat:.1e} m, |d| {np.linalg.norm(dvec):.9f}")
optax = -Rc2w[:, 2]
check("camera optical axis = -forward (looks back at the face), image up = head up",
      np.linalg.norm(optax + f) < 1e-6 and np.linalg.norm(Rc2w[:, 1] - u) < 1e-6 and np.linalg.norm(Rc2w[:, 0] - r) < 1e-6,
      f"optical axis {np.round(optax, 8).tolist()}")
pM = proj(P, M)
check("M projects to (512, 512) (corner-origin convention)", np.linalg.norm(pM - [512, 512]) < 1e-3, f"{pM.tolist()}")
SUMMARY["camera"] = {"K": K.tolist(), "P": P.tolist(), "dist_along_m": along, "lateral_m": lat, "M_px": pM.tolist(),
                     "max_P_vs_meta_px": dP, "max_P_vs_w2cv_px": dW2C}

# ---------------------------------------------------------------- item 6: pixel convention
print("\n== 6. Pixel-coordinate convention ==")
print("      metadata/P use continuous coords with the image spanning [0,1024]; pixel (i,j) centre = (i+0.5, j+0.5).")
print("      OpenCV/integer-index convention (pixel centre = integer): u_cv = u_meta - 0.5, v_cv = v_meta - 0.5,")
print("      equivalently K_cv = K with cx, cy = 511.5. Using P as-is with integer pixel indices biases by +0.5 px.")
# synthetic anti-aliased discs (8x8 supersampled pixel coverage, pixel (i,j) covers [i,i+1)x[j,j+1))
# at 25 known sub-pixel centres: a detector following the corner-origin convention has ~0 mean bias,
# one that forgot the +0.5 would show a -0.5 px bias on both axes.
def synth_disc(cx_, cy_, rad=9.0, size=120, SS=8):
    yy_, xx_ = np.mgrid[0:size, 0:size].astype(float)
    acc_ = np.zeros((size, size))
    for a in range(SS):
        for b in range(SS):
            acc_ += ((xx_ + (a + 0.5) / SS - cx_) ** 2 + (yy_ + (b + 0.5) / SS - cy_) ** 2 <= rad * rad)
    return 0.6 - 0.58 * acc_ / SS ** 2


true_list = [(60.0 + 0.2 * i + 0.03, 55.0 + 0.2 * j + 0.07) for i in range(5) for j in range(5)]
synths = [synth_disc(*tc_) for tc_ in true_list]
err_m = np.array([my_detect(im, (58.0, 58.0), 24.0, 9.0)[0] - np.array(tc_) for im, tc_ in zip(synths, true_list)])
check("own detector on 25 synthetic discs: |bias| < 0.02 px, max error < 0.1 px",
      np.abs(err_m.mean(0)).max() < 0.02 and np.linalg.norm(err_m, axis=1).max() < 0.1,
      f"bias ({err_m[:,0].mean():+.4f},{err_m[:,1].mean():+.4f}) max {np.linalg.norm(err_m, axis=1).max():.4f} px")
SUMMARY["convention"] = {"own_detector_synth_bias_px": err_m.mean(0).tolist(),
                         "own_detector_synth_max_px": float(np.linalg.norm(err_m, axis=1).max())}
if USE_WRAPPER:
    try:
        sys.path.insert(0, os.path.join(PKG, "wrapper"))
        import eyes as pkg_eyes  # noqa: E402
        err_p = np.array([np.array(pkg_eyes.detect_pupil_in_image(im, (58.0, 58.0), 24.0)[0]) - np.array(tc_)
                          for im, tc_ in zip(synths, true_list)])
        bias = err_p.mean(0)
        check("package detect_pupil_in_image follows the same corner-origin (+0.5) convention (|bias| < 0.05 px)",
              np.abs(bias).max() < 0.05,
              f"bias ({bias[0]:+.4f},{bias[1]:+.4f}) px, max single error {np.linalg.norm(err_p, axis=1).max():.3f} px "
              f"(binary-threshold quantisation); a missing +0.5 would give a -0.5 bias")
        SUMMARY["convention"]["pkg_detector_synth_bias_px"] = bias.tolist()
        SUMMARY["convention"]["pkg_detector_synth_max_px"] = float(np.linalg.norm(err_p, axis=1).max())
    except Exception as ex:  # pragma: no cover
        check("package detector convention test ran", False, repr(ex))

# ---------------------------------------------------------------- item 7: pupils
print("\n== 7. Pupil-centre projection ==")
d3, d2 = 0.0, 0.0
for (k, h, v, sol), rec in zip(rows, meta):
    for s in SIDES:
        P3 = sol[s]["P3"]
        d3 = max(d3, float(np.linalg.norm(P3 - np.array(rec["extra"][f"pupil_center_3D_{s}"]))))
        p2 = proj(P, P3)
        sol[s]["p2"] = p2
        d2 = max(d2, float(np.linalg.norm(p2 - np.array(rec[f"pupil_center_2D_{s}"]))))
check("pupil 3-D (numpy) == metadata extra.pupil_center_3D", d3 < 1e-6, f"max {d3:.2e} m")
check("pupil 2-D (numpy P) == metadata pupil_center_2D", d2 < 1e-3, f"max {d2:.2e} px")
# travel at extremes
neutral = solve(0, 0)
trav = {}
for nm, hh, vv in EXT[:4]:
    sol = solve(hh, vv)
    for s in SIDES:
        p_n = proj(P, C[s] + neutral[s]["R"] @ [0, -pupil_off, 0])
        p_e = proj(P, C[s] + sol[s]["R"] @ [0, -pupil_off, 0])
        trav[f"{nm}_{s}"] = (p_e - p_n).tolist()
print("      projected pupil travel from neutral (dx, dy px): " +
      "; ".join(f"{key} ({vv_[0]:+.2f},{vv_[1]:+.2f})" for key, vv_ in trav.items()))
SUMMARY["pupil_travel_px"] = trav
if USE_IMAGES:
    errs, signed, qa_diff = [], [], []
    for (k, h, v, sol), rec in zip(rows, meta):
        img = os.path.join(PKG, rec["extra"].get("image", f"renders/frame_{k:02d}.png"))
        luma = load_png_luma(img, max_rows=700)
        for s in SIDES:
            depth = float(np.linalg.norm(sol[s]["P3"] - tc))
            rp_px = cal["pupil_radius_m"] * fx / depth
            roi = 0.95 * cal["iris_radius_m"] * fx / float(np.linalg.norm(C[s] - tc))
            c_img, area = my_detect(luma, proj(P, C[s]), roi, rp_px)
            dlt = c_img - sol[s]["p2"]
            errs.append(float(np.linalg.norm(dlt)))
            signed.append((s, dlt))
            qd = rec["extra"].get("qa", {}).get(f"pupil_center_2D_detected_{s}")
            if qd:
                qa_diff.append(float(np.linalg.norm(c_img - np.array(qd))))
            print(f"      frame {k} {s}: proj ({sol[s]['p2'][0]:8.3f},{sol[s]['p2'][1]:8.3f})  img ({c_img[0]:8.3f},{c_img[1]:8.3f})"
                  f"  d=({dlt[0]:+.3f},{dlt[1]:+.3f}) |d|={errs[-1]:.3f}  r_p={rp_px:.2f}px")
    e = np.array(errs)
    dx = np.array([d[0] for _, d in signed])
    dy = np.array([d[1] for _, d in signed])
    dxL = np.array([d[0] for s_, d in signed if s_ == "L"])
    dxR = np.array([d[0] for s_, d in signed if s_ == "R"])
    print(f"      |error| px: mean {e.mean():.3f}  median {np.median(e):.3f}  RMS {math.sqrt((e**2).mean()):.3f}  max {e.max():.3f}  (n={len(e)})")
    print(f"      signed: mean dx {dx.mean():+.3f} (L {dxL.mean():+.3f}, R {dxR.mean():+.3f}), mean dy {dy.mean():+.3f}")
    check("own image detection vs projected labels: max < 1.0 px", e.max() < 1.0, f"max {e.max():.3f}, mean {e.mean():.3f}")
    check("no half-pixel convention bias in x (|mean dx| < 0.25 px; eyes not mirrored)", abs(dx.mean()) < 0.25,
          f"mean dx {dx.mean():+.3f} px")
    if qa_diff:
        print(f"      own detector vs package extra.qa detections: mean {np.mean(qa_diff):.3f}, max {np.max(qa_diff):.3f} px")
    SUMMARY["pupil_image_check"] = {"n": len(e), "mean_px": float(e.mean()), "median_px": float(np.median(e)),
                                    "rms_px": float(math.sqrt((e ** 2).mean())), "max_px": float(e.max()),
                                    "mean_dx": float(dx.mean()), "mean_dy": float(dy.mean()),
                                    "mean_dx_L": float(dxL.mean()), "mean_dx_R": float(dxR.mean()),
                                    "per_pupil_px": errs,
                                    "vs_pkg_qa_mean_px": float(np.mean(qa_diff)) if qa_diff else None}

    # the four +-1 extremes + neutral, if the direction-test renders are shipped
    dt_err = {}
    for nm, hh, vv in EXT:
        img = os.path.join(PKG, "reports", "direction_test", f"{nm.lower()}.png")
        if not os.path.isfile(img):
            continue
        sol = solve(hh, vv)
        luma = load_png_luma(img, max_rows=700)
        for s in SIDES:
            P3 = C[s] + sol[s]["R"] @ np.array([0.0, -pupil_off, 0.0])
            p2 = proj(P, P3)
            rp_px = cal["pupil_radius_m"] * fx / float(np.linalg.norm(P3 - tc))
            roi = 0.95 * cal["iris_radius_m"] * fx / float(np.linalg.norm(C[s] - tc))
            c_img, _ = my_detect(luma, proj(P, C[s]), roi, rp_px)
            dt_err[f"{nm}_{s}"] = (c_img - p2).tolist()
    if dt_err:
        print("      direction-test renders (+-1 extremes, neutral): image - projection (px): " +
              "; ".join(f"{key} ({vv_[0]:+.3f},{vv_[1]:+.3f})" for key, vv_ in dt_err.items()))
        mx_dt = max(math.hypot(*vv_) for vv_ in dt_err.values())
        check("own image detection vs projection on direction-test renders: max < 1.0 px", mx_dt < 1.0, f"max {mx_dt:.3f} px")
        SUMMARY["pupil_image_check"]["direction_test_px"] = dt_err

# ---------------------------------------------------------------- item 8: Listing
print("\n== 8. Listing's law ==")
oop = 0.0
for rec in meta:
    for s in SIDES:
        q = np.array(rec["extra"][f"eye_rotation_quat_wxyz_{s}"])
        ax = q[1:] / np.linalg.norm(q[1:])
        oop = max(oop, abs(float(np.dot(ax, A[s]))))
for nm, hh, vv in EXT:
    sol = solve(hh, vv)
    for s in SIDES:
        ax = sol[s]["q"][1:] / np.linalg.norm(sol[s]["q"][1:])
        oop = max(oop, abs(float(np.dot(ax, A[s]))))
check("all rotation axes lie in Listing's plane (axis . primary axis)", oop < 1e-6, f"max |component| {oop:.2e}")
print("      torsion of each eye in Fick and Helmholtz gimbals (deg; + = EXTORSION, i.e. 12 o'clock meridian temporal)")
fick_rows, maxF, maxH = [], 0.0, 0.0
for label, hh, vv in [(f"f{k}", h, v) for k, h, v in SEQ] + [(nm, hh, vv) for nm, hh, vv in EXT]:
    sol = solve(hh, vv)
    rowd = {"pose": label, "h": hh, "v": vv}
    for s in SIDES:
        temporal_sign = 1.0 if np.dot(C[s] - M, r) > 0 else -1.0     # L eye: temporal = +X
        thF, phF, psF = fick(sol[s]["R"])
        thH, phH, psH = helmholtz(sol[s]["R"])
        # closed-form Listing->Fick relation  tan(psi/2) = -tan(theta/2) tan(phi/2)  (derived; checked here)
        rowd.update({f"fick_tors_{s}": deg(psF) * temporal_sign, f"helm_tors_{s}": deg(psH) * temporal_sign,
                     f"fick_elev_{s}": -deg(phF), f"helm_elev_{s}": -deg(phH)})
        maxF, maxH = max(maxF, abs(deg(psF))), max(maxH, abs(deg(psH)))
    fick_rows.append(rowd)
    print(f"      {label:8s} ({hh:+.3f},{vv:+.3f})  Fick tors L {rowd['fick_tors_L']:+.3f} R {rowd['fick_tors_R']:+.3f} | "
          f"Helmholtz tors L {rowd['helm_tors_L']:+.3f} R {rowd['helm_tors_R']:+.3f} | elev Fick {rowd['fick_elev_L']:+.3f}/{rowd['fick_elev_R']:+.3f} "
          f"Helm {rowd['helm_elev_L']:+.3f}/{rowd['helm_elev_R']:+.3f}")
print(f"      max |Fick torsion| {maxF:.3f} deg, max |Helmholtz torsion| {maxH:.3f} deg (non-zero 'false torsion' of a Listing eye)")
helm_vdiff = max(abs(rw["helm_elev_L"] - rw["helm_elev_R"]) for rw in fick_rows)
fick_vdiff = max(abs(rw["fick_elev_L"] - rw["fick_elev_R"]) for rw in fick_rows)
check("Helmholtz elevations of the two eyes equal (common plane of regard)", helm_vdiff < 1e-6,
      f"max |dElev_H| {helm_vdiff:.1e} deg; Fick elevations differ by up to {fick_vdiff:.4f} deg")


# L2 comparison: Listing's planes rotated temporally by alpha = k * (vergence/2) (Mok et al. 1992: ~9.4 deg for 15 deg/eye)
def l2_orientation(s, d, alpha):
    temporal = 1.0 if np.dot(C[s] - M, r) > 0 else -1.0
    beta = temporal * alpha            # rotation about +Z (up) moving -Y toward +X is positive
    Rp = Rz(beta)
    p = Rp @ A[s]
    return q_to_R(shortest_arc(p, d)) @ Rp


def cyclo(s, R_model, R_l2):
    """torsion of R_l2 relative to R_model about the common gaze line; + = extorsion"""
    D = R_l2 @ R_model.T
    q = R_to_q(D)
    g = R_model @ A[s]
    ang = 2 * math.atan2(float(np.dot(q[1:], g)), q[0])
    # + rotation about gaze direction g (pointing out of the eye) moves 12 o'clock toward ... compute sign:
    top = R_model @ u
    moved = D @ top
    temporal_dir = r if np.dot(C[s] - M, r) > 0 else -r
    sign = 1.0 if np.dot(moved - top, temporal_dir) > 0 else -1.0
    return sign * abs(deg(ang))


k_mok = 2 * 4.7 / 15.0
# sanity: reproduce Mok's ~2.5 deg extorsion at 30 deg down with 15 deg/eye convergence
dtest = unit(np.array([-math.sin(math.radians(15)) * math.cos(math.radians(30)), -math.cos(math.radians(15)) * math.cos(math.radians(30)), -math.sin(math.radians(30))]))
san = cyclo("L", q_to_R(shortest_arc(A["L"], dtest)), l2_orientation("L", dtest, math.radians(9.4)))
print(f"      L2 model sanity: 15 deg/eye vergence, 30 deg down, Listing plane +9.4 deg temporal -> {san:+.2f} deg (Mok et al. 1992 report ~2.5 deg extorsion)")
l2_rows = []
for label, hh, vv in [(f"f{k}", h, v) for k, h, v in SEQ] + [(nm, hh, vv) for nm, hh, vv in EXT]:
    sol = solve(hh, vv)
    alpha = k_mok * sol["verg"] / 2
    tors = {s: cyclo(s, sol[s]["R"], l2_orientation(s, sol[s]["g"], alpha)) for s in SIDES}
    l2_rows.append({"pose": label, "h": hh, "v": vv, "alpha_deg": deg(alpha), "L": tors["L"], "R": tors["R"],
                    "cyclovergence": tors["L"] + tors["R"]})
    print(f"      {label:8s} ({hh:+.3f},{vv:+.3f}) L2 plane rot {deg(alpha):.2f} deg -> missing torsion L {tors['L']:+.3f} R {tors['R']:+.3f}"
          f" deg (cyclovergence {tors['L']+tors['R']:+.3f}, + = excyclo)")
mx_l2 = max(abs(rw["cyclovergence"]) for rw in l2_rows)
print(f"      max |L2 cyclovergence| not modelled: {mx_l2:.3f} deg (k = {k_mok:.3f} x half-vergence; scales linearly with k)")
SUMMARY["listing"] = {"max_axis_out_of_plane": oop, "max_fick_torsion_deg": maxF, "max_helmholtz_torsion_deg": maxH,
                      "fick_helm_rows": fick_rows, "L2_k": k_mok, "L2_rows": l2_rows, "L2_sanity_mok_deg": san,
                      "max_L2_cyclovergence_deg": mx_l2}

# ---------------------------------------------------------------- wrapper cross-check
if USE_WRAPPER:
    print("\n== wrapper cross-check (package eyes.move vs this script) ==")
    try:
        ew = pkg_eyes.get()
        wq, wm, wmeta = 0.0, 0.0, 0.0
        for nm, hh, vv in [(f"f{k}", h, v) for k, h, v in SEQ] + [(nm, hh, vv) for nm, hh, vv in EXT]:
            ew.move(hh, vv)
            sol = solve(hh, vv)
            for s in SIDES:
                Rw = M4(ew.eyeballs[s].matrix_world)[:3, :3]
                wq = max(wq, rot_angle_between(Rw, sol[s]["R"]))
                P3 = C[s] + sol[s]["R"] @ np.array([0.0, -pupil_off, 0.0])
                wm = max(wm, float(np.linalg.norm(np.array(ew.project(ew.pupils[s].matrix_world.to_translation()))
                                                  - proj(P, P3))))
        for rec in meta:
            ew.move(rec["input"]["h"], rec["input"]["v"])
            md = ew.metadata()
            for s in SIDES:
                wmeta = max(wmeta, float(np.linalg.norm(np.array(md[f"pupil_center_2D_{s}"]) - np.array(rec[f"pupil_center_2D_{s}"]))),
                            float(np.abs(np.array(md[f"eye_rotation_{s}"]) - np.array(rec[f"eye_rotation_{s}"])).max()))
        ew.reset()
        check("package move() rotations == numpy (14 poses)", wq < 2e-6, f"max {wq:.2e} rad")
        check("package project() == numpy P (pupils, 14 poses)", wm < 1e-3, f"max {wm:.2e} px")
        check("package metadata() reproduces metadata.json (10 frames)", wmeta < 1e-5, f"max {wmeta:.2e}")
        SUMMARY["wrapper"] = {"rot_rad": wq, "proj_px": wm, "meta_repro": wmeta}
    except Exception as ex:  # pragma: no cover
        check("wrapper cross-check ran", False, repr(ex))

# ---------------------------------------------------------------- final
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
SUMMARY["checks"] = [{"name": nme, "pass": ok, "detail": d} for nme, ok, d in RESULTS]
SUMMARY["n_checks"], SUMMARY["n_fail"] = len(RESULTS), n_fail
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks PASS")
compact = {k: v for k, v in SUMMARY.items() if k not in ("checks",)}
for key in ("rotations",):
    compact[key] = {kk: vv for kk, vv in SUMMARY[key].items() if kk not in ("sequence",)}
if "listing" in compact:
    compact["listing"] = {kk: vv for kk, vv in SUMMARY["listing"].items() if kk not in ("fick_helm_rows", "L2_rows")}
print("JSON_SUMMARY " + json.dumps(compact, default=float))
if JSON_OUT:
    with open(JSON_OUT, "w") as fh:
        json.dump(SUMMARY, fh, indent=1, default=float)
sys.exit(1 if n_fail else 0)
