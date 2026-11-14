"""
eyes.py -- calibrated programmatic eye articulation (Samsung PRISM XR Hackathon 2026, Track 1)

Public API (module level, so `import eyes` works as the challenge describes):

    import eyes
    eyes.move(h, v)        # h, v in [-1, +1];  +h -> gaze towards image RIGHT, +v -> gaze UP
    eyes.reset()           # calibrated neutral pose: both eyes fixate the nominal 35 cm target
    path = eyes.render()   # renders the current pose to a 1024x1024 PNG, returns its absolute path
    eyes.metadata()        # ground-truth record for the current pose (rotations, 2-D pupils, camera)

The same functionality is available as a class (`eyes.Eyes`) for scripted use.

Geometry (see README "Coordinate conventions"):
  * World = Blender world, metres, Z-up. The head looks along -Y. +X is the subject's LEFT,
    which is the RIGHT side of the rendered image (the camera faces the subject).
  * Eye midpoint M = (C_L + C_R) / 2, where C_* are the eyeball rotation centres (eye-bone heads).
  * Nominal fixation  F0 = M + 0.35 m * forward.     Camera optical centre = M + 0.40 m * forward.
  * Target            T  = F0 + h * 0.05 m * image_right + v * 0.05 m * up     (|h|,|v| <= 1)
  * Each eye independently: q = shortest-arc rotation taking its primary-position optical axis
    onto (T - C_eye). Shortest-arc from a fixed straight-ahead primary position puts every rotation
    axis in Listing's plane (monocular Listing's law, L1; zero torsion in Listing coordinates --
    Fick/Helmholtz "false torsion" still reaches ~0.5 deg). The vergence-dependent binocular
    extension (L2, cyclovergence) is NOT modelled; see README section 9.
    The quaternion is applied to that eye's pose bone, i.e. about the eyeball's own centre.

Runs inside Blender (>= 4.2; developed/tested on 5.0.1) or with the `bpy` pip module.
No GUI interaction is needed: if the scene is not already open, scene.blend next to the
wrapper folder (or $EYES_SCENE) is opened automatically.
"""
from __future__ import annotations

import json
import math
import re
import os
import sys
import time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

__all__ = ["Eyes", "move", "reset", "render", "metadata", "get", "detect_pupil_in_image"]

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WRAPPER_DIR)
DEFAULT_SCENE = os.environ.get("EYES_SCENE", os.path.join(ROOT_DIR, "scene.blend"))
# ad-hoc renders (eyes.render() without a path, CLI without --out) never go into the official renders/
DEFAULT_RENDER_DIR = os.path.join(ROOT_DIR, "demo_run", "renders")

EULER_ORDER = "XYZ"          # export order for eye_rotation_L / eye_rotation_R (radians, world frame)
_INPUT_EPS = 1e-9            # tolerance on the [-1, 1] input range (floating-point round-off)


# ======================================================================================
# helpers
# ======================================================================================
def _ensure_scene(scene_path: str | None = None) -> None:
    """Open scene.blend unless a compatible scene (EyeRig with calibration) is already loaded."""
    rig = bpy.data.objects.get("EyeRig")
    if rig is not None and "calibration_json" in rig.keys():
        return
    path = os.path.abspath(scene_path or DEFAULT_SCENE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scene not found: {path}")
    bpy.ops.wm.open_mainfile(filepath=path)
    rig = bpy.data.objects.get("EyeRig")
    if rig is None or "calibration_json" not in rig.keys():
        raise RuntimeError(f"{path} does not contain the EyeRig calibration")


def _validate_input(name: str, x) -> float:
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number in [-1, 1], got bool")
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be a real number in [-1, 1], got {type(x).__name__} {x!r}")
    try:
        xf = float(x)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number in [-1, 1], got {x!r}") from None
    if not math.isfinite(xf):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if xf < -1.0 - _INPUT_EPS or xf > 1.0 + _INPUT_EPS:
        raise ValueError(f"{name} must be within [-1, 1], got {xf}")
    return max(-1.0, min(1.0, xf))


def _tf_get(o):
    """Exact transform channels of an object (restoring these avoids matrix decomposition round-off)."""
    return (o.rotation_mode, tuple(o.location), tuple(o.rotation_euler), tuple(o.rotation_quaternion),
            tuple(o.rotation_axis_angle), tuple(o.scale), tuple(o.delta_location),
            tuple(o.delta_rotation_euler), tuple(o.delta_rotation_quaternion), tuple(o.delta_scale))


def _tf_set(o, t):
    (o.rotation_mode, o.location, o.rotation_euler, o.rotation_quaternion, o.rotation_axis_angle,
     o.scale, o.delta_location, o.delta_rotation_euler, o.delta_rotation_quaternion, o.delta_scale) = t


def relpath_safe(path: str, start: str) -> str:
    """os.path.relpath that falls back to the absolute path (Windows: different drives)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)


_OFFICIAL_FRAME = re.compile(r"frame_\d\d\.png$", re.IGNORECASE)


def is_official_output(path: str) -> bool:
    """True for the submission files renders/frame_NN.png and metadata.json in the package root."""
    rp = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(os.path.realpath(ROOT_DIR))
    if rp == os.path.join(root, "metadata.json"):
        return True
    return os.path.dirname(rp) == os.path.join(root, "renders") and bool(_OFFICIAL_FRAME.search(rp))


def _vec(v):
    return [float(c) for c in v]


def _mat(m):
    return [[float(c) for c in row] for row in m]


def _matrices_equal(a: Matrix, b: Matrix, tol=1e-6) -> bool:      # 1 µm / 1e-6 (float32 storage)
    return all(abs(a[i][j] - b[i][j]) <= tol for i in range(len(a)) for j in range(len(a[0])))


# ======================================================================================
# image-based pupil detection (independent check of the projected ground truth)
# ======================================================================================
def _load_luma(path: str) -> np.ndarray:
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    px = px.reshape(h, w, 4)[::-1]                         # row 0 = top of the image
    return 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]


def detect_pupil_in_image(image_or_path, roi_center_xy, roi_radius_px, dark_threshold=None, pupil_radius_px=None):
    """Centroid of the darkest connected blob (the pupil) inside a circular ROI.

    If `pupil_radius_px` is given (the pupil's physical radius x fx / depth: a scene constant that
    does not depend on where the pupil is), the blob centroid is refined to sub-pixel accuracy:
    edge points are found along 120 rays from the current centre, each with its own half-way
    threshold between the pupil level and the level just outside the pupil on that ray, and a
    trimmed least-squares (Kasa) circle fit keeps the 70 % best-fitting rays. A dark eyelid margin,
    lash shadow or corneal glint on one side therefore cannot drag the centre. (Method adopted from
    the independent maths audit, docs/audit/math_audit.md.) The projected label is never used.

    The ROI is centred on the projected *eyeball centre* (independent of gaze), so the
    measurement is not biased toward the predicted pupil position. Holes inside the blob
    (corneal specular highlights) are filled before taking the centroid.
    Returns ((x, y), area_px) in image pixel coordinates (origin top-left corner, y down),
    or (None, 0) if nothing dark was found."""
    luma = _load_luma(image_or_path) if isinstance(image_or_path, str) else image_or_path
    H, W = luma.shape
    cx, cy = roi_center_xy
    r = int(math.ceil(roi_radius_px))
    x0, x1 = max(0, int(cx) - r - 1), min(W, int(cx) + r + 2)
    y0, y1 = max(0, int(cy) - r - 1), min(H, int(cy) + r + 2)
    sub = luma[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inroi = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= roi_radius_px ** 2
    vals = sub[inroi]
    if vals.size == 0:
        return None, 0
    if dark_threshold is None:
        lo = float(np.percentile(vals, 1.0))
        med = float(np.median(vals))
        dark_threshold = lo + 0.30 * (med - lo)   # pupil is near-black; lid shadows are mid-tone
    dark = (sub < dark_threshold) & inroi
    if not dark.any():
        return None, 0
    # keep the largest 4-connected dark component
    lab = np.zeros(dark.shape, dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    for (sy, sx) in zip(*np.nonzero(dark)):
        if lab[sy, sx]:
            continue
        cur += 1
        stack, n = [(sy, sx)], 0
        lab[sy, sx] = cur
        while stack:
            py, px_ = stack.pop()
            n += 1
            for ny, nx in ((py + 1, px_), (py - 1, px_), (py, px_ + 1), (py, px_ - 1)):
                if 0 <= ny < dark.shape[0] and 0 <= nx < dark.shape[1] and dark[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
        if n > best_n:
            best, best_n = cur, n
    blob = lab == best
    # fill holes (specular glints): background = non-blob pixels reachable from the ROI border
    bg = np.zeros_like(blob)
    hgt, wid = blob.shape
    stack = [(y, x) for y in range(hgt) for x in (0, wid - 1)] + [(y, x) for x in range(wid) for y in (0, hgt - 1)]
    stack = [(y, x) for (y, x) in stack if not blob[y, x]]
    for y, x in stack:
        bg[y, x] = True
    while stack:
        py, px_ = stack.pop()
        for ny, nx in ((py + 1, px_), (py - 1, px_), (py, px_ + 1), (py, px_ - 1)):
            if 0 <= ny < hgt and 0 <= nx < wid and not blob[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                stack.append((ny, nx))
    filled = ~bg
    ys, xs = np.nonzero(filled)
    centre = (float(xs.mean() + x0 + 0.5), float(ys.mean() + y0 + 0.5))
    area = int(filled.sum())
    if pupil_radius_px:
        refined = _refine_pupil_circle(luma, centre, float(pupil_radius_px))
        if refined is not None:
            centre = refined
    return centre, area


def _refine_pupil_circle(luma, c0, rp):
    """Sub-pixel pupil centre: per-ray half-way edge points + trimmed Kasa circle fit.
    Coordinates are continuous with the origin at the top-left image corner (pixel centres at +0.5)."""
    H, W = luma.shape

    def sample(px, py):                                   # bilinear
        fx_, fy_ = px - 0.5, py - 0.5
        ix0 = np.clip(np.floor(fx_).astype(int), 0, W - 2)
        iy0 = np.clip(np.floor(fy_).astype(int), 0, H - 2)
        ax, ay = fx_ - ix0, fy_ - iy0
        return ((1 - ax) * (1 - ay) * luma[iy0, ix0] + ax * (1 - ay) * luma[iy0, ix0 + 1]
                + (1 - ax) * ay * luma[iy0 + 1, ix0] + ax * ay * luma[iy0 + 1, ix0 + 1])

    c = np.array(c0, dtype=float)
    angs = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rr = np.arange(0.45 * rp, 1.9 * rp, 0.05)
    for _ in range(3):
        X = c[0] + np.cos(angs)[:, None] * rr[None, :]
        Y = c[1] + np.sin(angs)[:, None] * rr[None, :]
        prof = sample(X, Y)
        Ip = float(np.median(prof[:, rr < 0.7 * rp]))
        edges = np.full(len(angs), np.nan)
        for i in range(len(angs)):
            Io = float(np.median(prof[i, (rr > 1.15 * rp) & (rr < 1.45 * rp)]))
            thr = 0.5 * (Ip + Io)
            above = np.nonzero(prof[i] > thr)[0]
            if Io - Ip < 0.02 or above.size == 0 or above[0] == 0:
                continue
            j = above[0]
            t = (thr - prof[i, j - 1]) / (prof[i, j] - prof[i, j - 1])
            edges[i] = rr[j - 1] + t * (rr[j] - rr[j - 1])
        ok = np.isfinite(edges)
        if ok.sum() < 12:
            return None
        ex = c[0] + np.cos(angs) * np.where(ok, edges, 0)
        ey = c[1] + np.sin(angs) * np.where(ok, edges, 0)
        sel = ok.copy()
        for _it in range(8):                              # trimmed LS: keep the 70 % best-fitting rays
            A = np.column_stack([ex[sel], ey[sel], np.ones(sel.sum())])
            sol = np.linalg.lstsq(A, ex[sel] ** 2 + ey[sel] ** 2, rcond=None)[0]
            cc = np.array([sol[0] / 2, sol[1] / 2])
            Rf = math.sqrt(max(sol[2] + cc @ cc, 0.0))
            res = np.abs(np.hypot(ex - cc[0], ey - cc[1]) - Rf)
            res[~ok] = np.inf
            sel = res <= np.quantile(res[ok], 0.70)
        c = cc
    return float(c[0]), float(c[1])


# ======================================================================================
# main class
# ======================================================================================
class Eyes:
    """Binocular eye rig controller bound to the scene's EyeRig armature."""

    SIDES = ("L", "R")

    def __init__(self, scene_path: str | None = None, render_dir: str | None = None, **config_overrides):
        _ensure_scene(scene_path)
        self.scene = bpy.context.scene
        self.rig = bpy.data.objects["EyeRig"]
        self.cal = json.loads(self.rig["calibration_json"])
        unknown = set(config_overrides) - set(self.cal)
        if unknown:
            raise KeyError(f"unknown calibration keys: {sorted(unknown)}")
        self.cal.update(config_overrides)               # e.g. Eyes(max_offset_h_m=0.04)
        self.camera = bpy.data.objects["RenderCamera"]
        self.target_empty = bpy.data.objects.get("FixationTarget")
        self.eyeballs = {s: bpy.data.objects[f"Eyeball.{s}"] for s in self.SIDES}
        self.pupils = {s: bpy.data.objects[f"Pupil.{s}"] for s in self.SIDES}
        self.render_dir = os.path.abspath(render_dir or DEFAULT_RENDER_DIR)
        self._render_count = 0
        self.h = 0.0
        self.v = 0.0
        self.last_render_path = None
        self.last_render_seconds = None
        self.last_metadata = None
        self.allow_official_outputs = False             # set only by generate_sequence.py (quality profile)
        # PNG metadata: keep render stats but never the absolute .blend path (machine-specific)
        self.scene.render.use_stamp_filename = False
        self._snapshot = self._capture_static_state()
        self.reset()
        self._check_calibration()

    # ---------------------------------------------------------------- static state / lock
    def _capture_static_state(self):
        """Everything that must never change between frames (restored by reset())."""
        lights = {o.name: (o.matrix_world.copy(), o.data.energy, tuple(o.data.color))
                  for o in self.scene.objects if o.type == "LIGHT"}
        self._channels = {o.name: _tf_get(o) for o in [self.rig, self.camera] +
                          [o for o in self.scene.objects if o.type == "LIGHT"]}
        r = self.scene.render
        return {
            "rig_matrix": self.rig.matrix_world.copy(),
            "camera_matrix": self.camera.matrix_world.copy(),
            "camera_lens": self.camera.data.lens,
            "camera_sensor": (self.camera.data.sensor_width, self.camera.data.sensor_fit,
                              self.camera.data.shift_x, self.camera.data.shift_y),
            "resolution": (r.resolution_x, r.resolution_y, r.resolution_percentage),
            "lights": lights,
            "world_strength": self._world_strength(),
        }

    def _world_strength(self):
        w = self.scene.world
        if w and w.node_tree and "Background" in w.node_tree.nodes:
            return float(w.node_tree.nodes["Background"].inputs["Strength"].default_value)
        return None

    def static_state_unchanged(self) -> dict:
        """Lock check used by render() and the tests: camera, lights, head, resolution untouched."""
        s = self._snapshot
        r = self.scene.render
        res = {
            "camera_pose_locked": _matrices_equal(self.camera.matrix_world, s["camera_matrix"]),
            "camera_intrinsics_locked": self.camera.data.lens == s["camera_lens"] and
            (self.camera.data.sensor_width, self.camera.data.sensor_fit,
             self.camera.data.shift_x, self.camera.data.shift_y) == s["camera_sensor"],
            "head_locked": _matrices_equal(self.rig.matrix_world, s["rig_matrix"]) and
            self.rig.pose.bones["head"].matrix_basis == Matrix.Identity(4),
            "resolution_locked": (r.resolution_x, r.resolution_y, r.resolution_percentage) == s["resolution"],
            "lighting_locked": all(
                _matrices_equal(bpy.data.objects[n].matrix_world, m) and bpy.data.objects[n].data.energy == e
                and tuple(bpy.data.objects[n].data.color) == c for n, (m, e, c) in s["lights"].items()
            ) and self._world_strength() == s["world_strength"],
        }
        res["all"] = all(res.values())
        return res

    # ---------------------------------------------------------------- geometry
    def _bone_rest_world(self, side):
        """World matrix of the eye bone in its rest orientation under the current head pose."""
        pb = self.rig.pose.bones[f"eye.{side}"]
        parent = pb.parent
        rel = parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        return self.rig.matrix_world @ parent.matrix @ rel

    def eye_center(self, side) -> Vector:
        return self._bone_rest_world(side).to_translation()

    def eye_midpoint(self) -> Vector:
        return 0.5 * (self.eye_center("L") + self.eye_center("R"))

    def head_axes(self):
        """(forward, image_right, up) unit vectors of the head frame in world space."""
        R = (self.rig.matrix_world @ self.rig.pose.bones["head"].matrix
             @ self.rig.pose.bones["head"].bone.matrix_local.inverted()).to_3x3().normalized()
        f = (R @ Vector(self.cal["head_forward_axis"])).normalized()
        r = (R @ Vector(self.cal["head_right_axis_image"])).normalized()
        u = (R @ Vector(self.cal["head_up_axis"])).normalized()
        return f, r, u

    def nominal_fixation(self) -> Vector:
        f, _, _ = self.head_axes()
        return self.eye_midpoint() + self.cal["fixation_distance_m"] * f

    def target_for(self, h: float, v: float) -> Vector:
        _, r, u = self.head_axes()
        return (self.nominal_fixation() + (h * self.cal["max_offset_h_m"]) * r
                + (v * self.cal["max_offset_v_m"]) * u)

    def _check_calibration(self):
        """Live rig must agree with the stored calibration (catches an edited/broken scene)."""
        tol = 1e-5
        for s in self.SIDES:
            d = (self.eye_center(s) - Vector(self.cal[f"eye_center_{s}"])).length
            if d > tol:
                raise RuntimeError(f"eye centre {s} deviates from calibration by {d*1000:.3f} mm")
        f, _, _ = self.head_axes()
        cam_d = (self.camera.matrix_world.to_translation() - self.eye_midpoint())
        if abs(cam_d.length - self.cal["camera_distance_m"]) > tol or cam_d.normalized().dot(f) < 1 - 1e-9:
            raise RuntimeError("camera is not at the calibrated distance/axis")

    # ---------------------------------------------------------------- public API
    def move(self, h, v) -> dict:
        """Point both eyes at T(h, v). Synchronous: the scene is updated before returning."""
        h = _validate_input("h", h)
        v = _validate_input("v", v)
        T = self.target_for(h, v)
        for s in self.SIDES:
            pb = self.rig.pose.bones[f"eye.{s}"]
            rest = self._bone_rest_world(s)
            C = rest.to_translation()
            R_rest = rest.to_3x3().normalized()
            axis_rest = R_rest.col[1].normalized()          # bone +Y == primary-position optical axis
            desired = (T - C).normalized()
            q_world = axis_rest.rotation_difference(desired)  # shortest arc: axis in Listing's plane (L1)
            q_rest = R_rest.to_quaternion()
            q_local = q_rest.inverted() @ q_world @ q_rest    # express the world rotation in bone space
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = q_local.normalized()
            pb.location = (0.0, 0.0, 0.0)                     # eyes never translate
            pb.scale = (1.0, 1.0, 1.0)
        if self.target_empty is not None:
            self.target_empty.location = T
        self.h, self.v = h, v
        bpy.context.view_layer.update()                       # synchronous depsgraph evaluation
        return self.state()

    def reset(self) -> dict:
        """Restore the calibrated neutral pose (head, camera, lights, both eyes on the 35 cm target)."""
        s = self._snapshot
        for n, t in self._channels.items():            # head, camera, lights: exact channel restore
            o = bpy.data.objects[n]
            if _tf_get(o) != t:
                _tf_set(o, t)
        for pb in self.rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        self.camera.data.lens = s["camera_lens"]
        (self.camera.data.sensor_width, self.camera.data.sensor_fit,
         self.camera.data.shift_x, self.camera.data.shift_y) = s["camera_sensor"]
        r = self.scene.render
        r.resolution_x, r.resolution_y, r.resolution_percentage = s["resolution"]
        for n, (m, e, c) in s["lights"].items():
            o = bpy.data.objects[n]
            o.data.energy = e
            o.data.color = c
        w = self.scene.world
        if s["world_strength"] is not None and self._world_strength() != s["world_strength"]:
            w.node_tree.nodes["Background"].inputs["Strength"].default_value = s["world_strength"]
        bpy.context.view_layer.update()
        return self.move(0.0, 0.0)

    def render(self, filepath: str | None = None) -> str:
        """Render the current pose to a 1024x1024 PNG and return its absolute path.

        The metadata of exactly this pose is kept in `self.last_metadata`."""
        lock = self.static_state_unchanged()
        if not lock["all"]:
            raise RuntimeError(f"static scene state changed since calibration: {lock}")
        if filepath is None:
            self._render_count += 1
            filepath = os.path.join(self.render_dir, f"render_{self._render_count:04d}.png")
        filepath = os.path.abspath(filepath)
        if not filepath.lower().endswith(".png"):
            filepath += ".png"
        if is_official_output(filepath) and not self.allow_official_outputs:
            raise PermissionError(f"refusing to overwrite the official submission frame {filepath}; "
                                  "regenerate it with wrapper/generate_sequence.py (quality profile)")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        r = self.scene.render
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGB"
        r.filepath = filepath
        self.last_metadata = self.metadata()                  # pose captured before rendering
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        self.last_render_seconds = time.perf_counter() - t0
        if not os.path.isfile(filepath):
            raise RuntimeError(f"render did not produce {filepath}")
        self.last_render_path = filepath
        return filepath

    # ---------------------------------------------------------------- measurements
    def project(self, world_point) -> list:
        """World point -> image pixel coordinates [x, y]; origin = top-left image corner, y down."""
        r = self.scene.render
        W = r.resolution_x * r.resolution_percentage / 100.0
        H = r.resolution_y * r.resolution_percentage / 100.0
        co = world_to_camera_view(self.scene, self.camera, Vector(world_point))
        return [float(co.x * W), float((1.0 - co.y) * H)]

    def camera_matrices(self) -> dict:
        """OpenCV-convention pinhole model of the render camera (pixel origin = top-left corner)."""
        r = self.scene.render
        W = r.resolution_x * r.resolution_percentage / 100.0
        H = r.resolution_y * r.resolution_percentage / 100.0
        cd = self.camera.data
        fit = cd.sensor_fit
        if fit == "AUTO":
            fit = "HORIZONTAL" if W >= H else "VERTICAL"
        sensor = cd.sensor_width if fit == "HORIZONTAL" else cd.sensor_height
        ref = W if fit == "HORIZONTAL" else H
        f_px = cd.lens / sensor * ref
        aspect = r.pixel_aspect_y / r.pixel_aspect_x
        fx, fy = f_px, f_px * aspect
        cx = W / 2.0 - cd.shift_x * ref
        cy = H / 2.0 + cd.shift_y * ref
        K = Matrix(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
        cam_w = self.camera.matrix_world.normalized()
        R_w2b = cam_w.to_3x3().transposed()
        t_w2b = -(R_w2b @ cam_w.to_translation())
        B2CV = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))    # Blender cam (x right, y up, -z fwd) -> OpenCV
        R = B2CV @ R_w2b
        t = B2CV @ t_w2b
        Rt = [list(R[i]) + [t[i]] for i in range(3)]
        P = [[sum(K[i][k] * Rt[k][j] for k in range(3)) for j in range(4)] for i in range(3)]
        return {"P": P, "K": _mat(K), "Rt": Rt, "camera_to_world_blender": _mat(cam_w)}

    def state(self) -> dict:
        """Geometric state of the current pose (actual, post-depsgraph values)."""
        T = self.target_for(self.h, self.v)
        out = {"input": {"h": self.h, "v": self.v}, "fixation_target": _vec(T)}
        dirs = {}
        f, rgt, up = self.head_axes()
        for s in self.SIDES:
            mw = self.eyeballs[s].matrix_world
            C = mw.to_translation()
            g = (mw.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()   # optical axis (eye local -Y)
            dirs[s] = g
            to_t = T - C
            miss = (to_t - to_t.dot(g) * g).length                         # target distance from gaze ray
            out[f"gaze_dir_{s}"] = _vec(g)
            out[f"gaze_ray_miss_m_{s}"] = float(miss)
            out[f"yaw_deg_{s}"] = math.degrees(math.atan2(g.dot(rgt), g.dot(f)))
            out[f"pitch_deg_{s}"] = math.degrees(math.asin(max(-1.0, min(1.0, g.dot(up)))))
        out["vergence_deg"] = math.degrees(dirs["L"].angle(dirs["R"]))
        return out

    def metadata(self) -> dict:
        """Challenge metadata record for the current pose (all values read from the live scene)."""
        cams = self.camera_matrices()
        rec = {"input": {"h": float(self.h), "v": float(self.v)}}
        for s in self.SIDES:
            q = self.eyeballs[s].matrix_world.to_quaternion()
            rec[f"eye_rotation_{s}"] = _vec(q.to_euler(EULER_ORDER))
        for s in self.SIDES:
            rec[f"pupil_center_2D_{s}"] = self.project(self.pupils[s].matrix_world.to_translation())
        rec["camera_matrix"] = cams["P"]
        # ---- extra (optional) fields for reproducibility ----
        st = self.state()
        ext = {
            "eye_rotation_convention": f"world-frame rotation of each eyeball from its primary (straight-ahead) "
                                       f"position; Euler {EULER_ORDER} radians (Blender extrinsic-XYZ convention)",
            "eye_rotation_quat_wxyz_L": _vec(self.eyeballs["L"].matrix_world.to_quaternion()),
            "eye_rotation_quat_wxyz_R": _vec(self.eyeballs["R"].matrix_world.to_quaternion()),
            "eye_center_3D_L": _vec(self.eyeballs["L"].matrix_world.to_translation()),
            "eye_center_3D_R": _vec(self.eyeballs["R"].matrix_world.to_translation()),
            "pupil_center_3D_L": _vec(self.pupils["L"].matrix_world.to_translation()),
            "pupil_center_3D_R": _vec(self.pupils["R"].matrix_world.to_translation()),
            "eye_center_2D_L": self.project(self.eyeballs["L"].matrix_world.to_translation()),
            "eye_center_2D_R": self.project(self.eyeballs["R"].matrix_world.to_translation()),
            "eye_midpoint_3D": _vec(self.eye_midpoint()),
            "camera_matrix_convention": "3x4 projection P = K [R|t] (OpenCV axes: x right, y down, z forward); "
                                        "[x, y, 1]^T ~ P [X, Y, Z, 1]^T gives pupil_center_2D in continuous pixel "
                                        "coordinates with the origin at the top-left image CORNER (pixel i spans "
                                        "[i, i+1), centres at i+0.5, so cx = cy = 512). For OpenCV's pixel-centre "
                                        "convention subtract 0.5 from x and y (cx = cy = 511.5). 'OpenCV axes' "
                                        "refers to the camera axis directions only",
            "camera_intrinsics_K": cams["K"],
            "camera_extrinsics_Rt_world_to_camera": cams["Rt"],
            "camera_to_world_blender": cams["camera_to_world_blender"],
            "image_size": [self.scene.render.resolution_x, self.scene.render.resolution_y],
        }
        ext.update({k: st[k] for k in st if k != "input"})
        rec["extra"] = ext
        return rec


# ======================================================================================
# module-level singleton API  (eyes.move / eyes.reset / eyes.render)
# ======================================================================================
_INSTANCE: Eyes | None = None


def get(**kwargs) -> Eyes:
    """Return the shared Eyes instance (created on first use)."""
    global _INSTANCE
    if _INSTANCE is None or kwargs:
        _INSTANCE = Eyes(**kwargs)
    else:
        try:  # instance invalid if another .blend was opened meanwhile
            _INSTANCE.rig.name
        except ReferenceError:
            _INSTANCE = Eyes()
    return _INSTANCE


def move(h, v) -> dict:
    return get().move(h, v)


def reset() -> dict:
    return get().reset()


def render(filepath: str | None = None) -> str:
    return get().render(filepath)


def metadata() -> dict:
    return get().metadata()


# ======================================================================================
# CLI:  blender -b scene.blend --python wrapper/eyes.py -- --move 1 0 --out left.png
#       python wrapper/eyes.py --move 0 1 --out up.png
# ======================================================================================
if __name__ == "__main__":
    import argparse

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Move the eyes and render one frame")
    ap.add_argument("--move", nargs=2, type=float, metavar=("H", "V"), default=None)
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--samples", type=int, default=None)
    a = ap.parse_args(argv)
    e = get()
    if a.samples:
        e.scene.cycles.samples = a.samples
    if a.move:
        e.move(*a.move)
    p = e.render(a.out)
    print(json.dumps({"render": p, "metadata": e.last_metadata}, indent=1))
