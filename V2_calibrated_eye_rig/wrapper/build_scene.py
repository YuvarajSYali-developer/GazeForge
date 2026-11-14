"""
build_scene.py -- deterministically (re)builds ../scene.blend from wrapper/assets/head_basemesh.npz

The shipped scene.blend was produced by this script; it is included so the scene can be
regenerated in any Blender >= 4.2 (e.g. if an evaluator's Blender cannot open a 5.0 file).

    blender -b --factory-startup --python wrapper/build_scene.py            # Blender executable
    python wrapper/build_scene.py                                         # pip "bpy" module

Scene contents (names are the contract used by wrapper/eyes.py):
    EyeRig            armature; bones "head", "eye.L", "eye.R" (eye bones head = eyeball centre)
    Face              textured MakeHuman head mesh, bone-parented to "head"
    Eyeball.L/.R      sclera sphere, bone-parented to "eye.L"/"eye.R"   (rotation centre = origin)
      Iris.L/.R       child of Eyeball
        Pupil.L/.R    child of Iris; object origin == 3-D pupil centre (projected for metadata)
      Cornea.L/.R     child of Eyeball (transparent + specular, IOR-free, see README)
    RenderCamera      fixed camera, 0.40 m in front of the eye midpoint, looking back along the face axis
    Key/Fill/Rim      fixed area lights
    FixationTarget    empty (not rendered) moved by the wrapper to the current fixation point
Calibration is stored as JSON in EyeRig["calibration_json"] (single source of truth).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSET = os.path.join(HERE, "assets", "head_basemesh.npz")

# --------------------------------------------------------------------------------------
# Calibration constants (documented in README; all lengths in metres)
# --------------------------------------------------------------------------------------
CAL = {
    "units": "metres, radians",
    "fixation_distance_m": 0.35,      # nominal fixation point: 35 cm forward of the eye midpoint
    "camera_distance_m": 0.40,        # camera optical centre: 40 cm forward of the eye midpoint
    "max_offset_h_m": 0.05,           # |h| = 1  ->  target displaced 5 cm along image-right axis
    "max_offset_v_m": 0.05,           # |v| = 1  ->  target displaced 5 cm along image-up axis
    "eye_radius_m": 0.0125,           # sclera sphere radius (anatomical ~12 mm; fitted to MakeHuman lids)
    "iris_radius_m": 0.0059,          # limbus radius (11.8 mm iris diameter)
    "pupil_radius_m": 0.0020,         # 4.0 mm pupil diameter (photopic)
    "cornea_radius_m": 0.0078,        # anterior corneal radius of curvature
    "resolution": [1024, 1024],
    "focal_length_mm": 62.0,
    "sensor_width_mm": 36.0,
    "render_engine": "CYCLES",
    "samples": 32,
    "seed": 0,
}

SEED = 20261118


# ======================================================================================
# helpers
# ======================================================================================
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class FourierNoise:
    """Deterministic smooth 3-D noise: sum of random plane waves (no external deps)."""

    def __init__(self, seed, n=24, freq=1.0, spread=0.5):
        rng = np.random.default_rng(seed)
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        mags = freq * np.exp(rng.uniform(-spread, spread, n))
        self.k = d * mags[:, None]
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.n = n

    def __call__(self, p):  # p (...,3) -> (...) roughly in [-1, 1]
        s = np.zeros(p.shape[:-1], dtype=np.float32)
        for k, ph in zip(self.k, self.ph):       # accumulate wave by wave (bounded memory)
            s += np.sin(p @ k + ph).astype(np.float32)
        return s / math.sqrt(self.n / 2.0)


def new_image(name, arr_srgb, alpha=None):
    """Create a packed 8-bit sRGB image from a float (H,W,3) array in [0,1] (row 0 = bottom)."""
    import tempfile
    h, w, _ = arr_srgb.shape
    tmp = bpy.data.images.new(name + "_tmp", width=w, height=h, alpha=False)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr_srgb, 0, 1)
    if alpha is not None:
        rgba[..., 3] = alpha
    tmp.pixels.foreach_set(rgba.ravel())
    path = os.path.join(tempfile.mkdtemp(prefix="eyescene_"), name + ".png")
    tmp.filepath_raw = path
    tmp.file_format = "PNG"
    tmp.save()
    bpy.data.images.remove(tmp)
    img = bpy.data.images.load(path)          # real FILE image -> can be packed into the .blend
    img.name = name
    img.colorspace_settings.name = "sRGB"
    img.pack()
    img.filepath = "//textures/" + name + ".png"   # packed; path is informational only
    return img


def set_input(node, names, value):
    """Set the first existing socket among `names` (Blender versions rename sockets)."""
    for n in names if isinstance(names, (list, tuple)) else [names]:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def mesh_from(name, verts, faces, uv_loops=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    if uv_loops is not None:
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(uv_loops, dtype=np.float32).ravel())
    me.validate(clean_customdata=False)
    if smooth:
        me.shade_smooth() if hasattr(me, "shade_smooth") else None
        for p in me.polygons:
            p.use_smooth = True
    me.update()
    return me


def lathe(profile, segments, uv_v=None):
    """Revolve a profile [(rho, d), ...] about the eye's forward axis (local -Y).

    Local eye frame: x = subject's left (+X world at rest), -y = forward (gaze), z = up.
    rho == 0 at an end collapses to a single pole vertex.  Returns verts, faces, per-loop UVs
    (u = azimuth/2pi, v = profile parameter)."""
    n = len(profile)
    uv_v = uv_v if uv_v is not None else [i / (n - 1) for i in range(n)]
    verts, ring_idx = [], []
    for i, (rho, d) in enumerate(profile):
        if rho < 1e-9:
            ring_idx.append([len(verts)] * (segments + 1))
            verts.append((0.0, -d, 0.0))
            continue
        ids = []
        for s in range(segments):
            a = 2 * math.pi * s / segments
            ids.append(len(verts))
            verts.append((rho * math.cos(a), -d, rho * math.sin(a)))
        ids.append(ids[0])
        ring_idx.append(ids)
    faces, uvs = [], []
    for i in range(n - 1):
        for s in range(segments):
            a, b = ring_idx[i][s], ring_idx[i][s + 1]
            c, d_ = ring_idx[i + 1][s + 1], ring_idx[i + 1][s]
            quad = [a, b, c, d_]
            uq = [(s / segments, uv_v[i]), ((s + 1) / segments, uv_v[i]),
                  ((s + 1) / segments, uv_v[i + 1]), (s / segments, uv_v[i + 1])]
            # drop duplicate pole vertices -> triangles
            keep = [k for k in range(4) if quad[k] not in quad[:k]]
            if len(keep) < 3:
                continue
            faces.append([quad[k] for k in keep])
            uvs.extend([uq[k] for k in keep])
    # faces wind so normals point outward when profile goes from front (small d? no) - fix later via recalc
    return verts, faces, uvs


def recalc_normals(obj, inside=False):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if inside:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()


def orient_outward(obj, centre_local=(0.0, 0.0, 0.0)):
    """Make every face normal point away from `centre_local` (robust for open surfaces)."""
    import bmesh
    c = Vector(centre_local)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    flip = [f for f in bm.faces if f.normal.length > 0 and f.normal.dot(f.calc_center_median() - c) < 0]
    if flip:
        bmesh.ops.reverse_faces(bm, faces=flip)
    bm.to_mesh(obj.data)
    bm.free()


# ======================================================================================
# textures (numpy, deterministic)
# ======================================================================================
def make_iris_texture(seed, size=2048):
    """Procedural brown iris (sRGB albedo) + height map (0..1, for bump).

    Layers (inner -> outer): pupillary ruff, collarette (jagged, lighter amber ring at ~1/3 of the
    iris width), radial stroma fibres that twist slightly with radius, crypts (dark lacunae),
    circumferential contraction furrows, dark limbal ring with a soft outer edge. Pupil area is black.
    The texture spans the iris disc (radius = iris_radius_m) on planar UVs."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    x = (x + 0.5) / size * 2 - 1
    y = (y + 0.5) / size * 2 - 1
    rho = np.hypot(x, y)
    th = np.arctan2(y, x)
    p0 = CAL["pupil_radius_m"] / CAL["iris_radius_m"]
    s = np.clip((rho - p0) / (1 - p0), 0, 1)                 # 0 at pupil margin, 1 at limbus

    def angular_noise(n_terms, lo, hi, twist):
        acc = np.zeros_like(rho)
        for _ in range(n_terms):
            k = rng.integers(lo, hi)
            acc += rng.uniform(0.3, 1.0) * np.sin(k * th + rng.uniform(0, 6.283) + rng.uniform(-twist, twist) * s)
        return acc / np.abs(acc).max()

    fib = angular_noise(110, 30, 300, 2.0)                   # trabecular fibres
    fine = angular_noise(60, 300, 900, 5.0)                  # fine striation
    wob = angular_noise(12, 3, 14, 0.5)                      # low-frequency patchiness

    collar_r = 0.33 + 0.030 * np.sin(7 * th + 1.3) + 0.018 * np.sin(13 * th + 0.4) + 0.010 * np.sin(29 * th)
    t_in = smoothstep(collar_r + 0.09, collar_r - 0.05, s)
    inner = np.array([0.37, 0.245, 0.125])                   # amber-brown pupillary zone
    outer = np.array([0.255, 0.16, 0.092])                   # dark brown ciliary zone
    col = outer * (1 - t_in[..., None]) + inner * t_in[..., None]
    # fibres: lighter ridges, darker gaps, stronger in the ciliary zone
    fib_amp = 0.26 + 0.16 * (1 - t_in)
    col *= (1 + fib_amp[..., None] * fib[..., None] + 0.10 * fine[..., None] + 0.08 * wob[..., None])
    # golden flecks along the collarette
    ridge = np.exp(-((s - collar_r) / 0.028) ** 2)
    col += (0.025 * ridge * (0.6 + 0.4 * fib))[..., None] * np.array([1.0, 0.78, 0.45])
    height = 0.5 + 0.22 * fib + 0.06 * fine + 0.25 * ridge
    # crypts: elongated radially, darker and lower
    for _ in range(34):
        cs, ct = rng.uniform(0.18, 0.85), rng.uniform(-np.pi, np.pi)
        rr = p0 + cs * (1 - p0)
        cx, cy = rr * math.cos(ct), rr * math.sin(ct)
        lr, lt = rng.uniform(0.030, 0.070), rng.uniform(0.012, 0.030)
        dx, dy = x - cx, y - cy
        u_r = dx * math.cos(ct) + dy * math.sin(ct)
        u_t = -dx * math.sin(ct) + dy * math.cos(ct)
        g = np.exp(-((u_r / lr) ** 2 + (u_t / lt) ** 2))
        col *= (1 - 0.45 * g)[..., None]
        height -= 0.35 * g
    # contraction furrows
    furrow = sum(np.exp(-((s - c - 0.01 * np.sin(5 * th + c * 9)) / 0.010) ** 2) for c in (0.66, 0.76, 0.86))
    col *= (1 - 0.20 * furrow)[..., None]
    height -= 0.20 * furrow
    # pupillary ruff (thin dark-brown pigment frill at the pupil margin)
    ruff = smoothstep(0.045, 0.0, s) * (0.85 + 0.15 * fine)
    col = col * (1 - 0.75 * ruff[..., None]) + np.array([0.10, 0.06, 0.04]) * 0.75 * ruff[..., None]
    # limbal ring: dark, soft-edged, slightly irregular
    lr0 = 0.84 + 0.015 * wob
    limb = smoothstep(lr0, 0.985, s)
    col = col * (1 - 0.85 * limb[..., None]) + np.array([0.07, 0.05, 0.04]) * 0.85 * limb[..., None]
    col[rho > 1.0] = [0.08, 0.06, 0.05]
    col[rho < p0] = [0.015, 0.015, 0.015]
    height = np.clip(height, 0, 1)
    height[rho < p0] = 0.3
    return np.clip(col, 0, 1), height


def make_sclera_texture(seed, w=2048, h=1024):
    """Equirectangular texture over the sclera lathe: u = azimuth, v: 0 = limbus edge, 1 = back pole."""
    rng = np.random.default_rng(seed)
    v = (np.arange(h) + 0.5) / h
    base_front = np.array([0.69, 0.655, 0.62])              # healthy sclera is off-white, not paper white
    base_back = np.array([0.70, 0.57, 0.53])
    t = smoothstep(0.02, 0.40, v)[:, None, None]
    img = np.broadcast_to(base_front * (1 - t) + base_back * t, (h, w, 3)).copy()
    # large-scale tint variation (slightly yellow / pink patches), periodic in u
    uu = (np.arange(w) + 0.5) / w * 2 * np.pi
    patch = np.zeros((h, w))
    for _ in range(10):
        k = rng.integers(1, 6)
        patch += np.sin(k * uu[None, :] + rng.uniform(0, 6.28)) * np.sin(rng.uniform(2, 9) * v[:, None] + rng.uniform(0, 6.28))
    patch /= np.abs(patch).max()
    img *= (1 + 0.035 * patch[..., None] * np.array([1.0, 0.97, 0.90]))
    # faint limbal shadow next to the cornea
    img *= (1 - 0.20 * smoothstep(0.035, 0.0, v))[:, None, None]
    # blood vessels: random walks from the periphery toward the limbus, branching
    vessel = np.zeros((h, w), dtype=np.float32)

    def walk(u0, v0, width, depth):
        u, vv = u0, v0
        ang = rng.uniform(-0.5, 0.5)
        steps = rng.integers(140, 320)
        for i in range(steps):
            ang += rng.normal(0, 0.18)
            ang = float(np.clip(ang, -1.2, 1.2))
            u = (u + 0.0012 * math.sin(ang) * 2) % 1.0
            vv -= 0.0018 * math.cos(ang)
            if vv < 0.03:
                break
            px, py = int(u * w), int(vv * h)
            wpx = max(1, int(width * (0.5 + vv)))
            fade = smoothstep(0.03, 0.30, vv)
            y0, y1 = max(0, py - wpx), min(h, py + wpx + 1)
            x0, x1 = max(0, px - wpx), min(w, px + wpx + 1)
            vessel[y0:y1, x0:x1] = np.maximum(vessel[y0:y1, x0:x1], fade * (0.35 + 0.4 * width / 3))
            if depth < 2 and rng.random() < 0.012:
                walk(u, vv, max(1.0, width * 0.6), depth + 1)

    for _ in range(70):
        walk(rng.uniform(0, 1), rng.uniform(0.30, 0.62), rng.uniform(0.8, 2.4), 0)
    vcol = np.array([0.66, 0.22, 0.22])
    a = (vessel * 0.42)[..., None]
    img = img * (1 - a) + vcol * a
    return np.clip(img, 0, 1)


# ======================================================================================
# head mesh + albedo
# ======================================================================================
def raster_attrs(tri_uv, tri_attr, W, H):
    """Rasterise per-corner attributes over UV triangles. tri_uv (T,3,2) in [0,1]; tri_attr (T,3,C).
    The LAST attribute channel is used as depth: where triangles overlap in UV space (e.g. mouth /
    orbit interior under a projective layout) the outermost surface (largest value) wins."""
    C = tri_attr.shape[2]
    out = np.zeros((H, W, C), dtype=np.float32)
    zbuf = np.full((H, W), -np.inf, dtype=np.float32)
    cov = np.zeros((H, W), dtype=bool)
    P = tri_uv * np.array([W, H])
    for t in range(len(P)):
        a, b, c = P[t]
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(W - 1, math.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(H - 1, math.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px, py = xs + 0.5, ys + 0.5
        l0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / den
        l1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        vals = l0[..., None] * tri_attr[t, 0] + l1[..., None] * tri_attr[t, 1] + l2[..., None] * tri_attr[t, 2]
        zsub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (vals[..., -1] > zsub)
        sub = out[y0:y1 + 1, x0:x1 + 1]
        sub[win] = vals[win]
        zsub[win] = vals[..., -1][win]
        cov[y0:y1 + 1, x0:x1 + 1] |= inside
    # dilate into empty texels (avoid seams under bilinear filtering / subdivision)
    for _ in range(12):
        if cov.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(cov.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sc = np.roll(cov, (dy, dx), axis=(0, 1))
            acc += np.roll(out, (dy, dx), axis=(0, 1)) * sc[..., None]
            cnt += sc
        fill = (~cov) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        cov = cov | fill
    return out, cov


def brow_hair_strokes(P, density, t_along, ex, ez, n_per_brow=750):
    """Eyebrow as individual hair strokes rasterised in 3-D (texel position map P).

    Roots are sampled from the brow density; each hair is a short segment whose direction turns
    from upward (medial head of the brow) to lateral and slightly downward (tail), with jitter.
    Coverage = 1 - prod(1 - a_i) over hairs. Deterministic (fixed seed)."""
    rng = np.random.default_rng(SEED + 31)
    H, W = density.shape
    # surface normal per texel from the position map (hairs must lie in the skin's tangent plane,
    # otherwise a straight 3-D segment leaves the curved brow after ~1 mm and renders as a dot)
    dPu = np.gradient(P, axis=1)
    dPv = np.gradient(P, axis=0)
    nrm = np.cross(dPu, dPv)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    region = density > 0.02
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return np.zeros_like(density)
    pts = P[ys, xs]
    wts = density[ys, xs].astype(np.float64)
    cover = np.ones(len(ys))
    for side in (1.0, -1.0):
        m = np.sign(pts[:, 0]) == side
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        prob = wts[idx] ** 1.2
        prob /= prob.sum()
        roots = idx[rng.choice(len(idx), size=n_per_brow, p=prob)]
        Q = pts[idx]
        for r in roots:
            p0 = pts[r]
            ta = float(np.clip(t_along[ys[r], xs[r]], 0, 1))
            ang = math.radians(np.interp(ta, [0.0, 0.18, 0.55, 1.0], [80.0, 45.0, 12.0, -18.0]) + rng.normal(0, 9))
            lateral = np.array([side, 0.0, 0.0])
            d = math.cos(ang) * lateral + math.sin(ang) * np.array([0.0, 0.0, 1.0])
            nn = nrm[ys[r], xs[r]]
            d = d - (d @ nn) * nn                         # into the tangent plane
            d /= max(np.linalg.norm(d), 1e-9)
            length = rng.uniform(0.0070, 0.0115) * (0.8 + 0.2 * (1 - ta))
            half_w = rng.uniform(0.060e-3, 0.095e-3)
            aa = 0.09e-3
            # distance from each brow texel to the segment (in 3-D, then the hair lies on the skin)
            v = Q - p0
            s_ = np.clip(v @ d / length, 0.0, 1.0)
            dist = np.linalg.norm(v - np.outer(s_ * length, d), axis=1)
            taper = 1.0 - 0.6 * s_
            a = smoothstep(half_w * taper + aa, half_w * taper - aa * 0.3, dist) * (0.55 + 0.35 * rng.random())
            cover[idx] *= (1.0 - a)
    out = np.zeros_like(density)
    out[ys, xs] = (1.0 - cover).astype(out.dtype)
    return np.clip(out, 0, 1)


def socket_weights(V, F, centres):
    """Per-vertex weight in [0, 1]: 1 on the inner eyelid / orbital socket surface that faces the
    eyeball (conjunctiva, caruncle, waterline), 0 on the outer skin.

    The MakeHuman lids fold inwards into a closed socket around the eyeball. Under the spherical
    UV layout those inner faces share texels with the outer lid skin, so they cannot be painted
    through the albedo; the shader blends a conjunctiva material by this attribute instead.
    Criterion: vertex normal pointing towards the eye centre (dot > 0) within 19 mm of it."""
    fn = np.cross(V[F[:, 2]] - V[F[:, 0]], V[F[:, 3]] - V[F[:, 1]])
    vn = np.zeros_like(V)
    for k in range(4):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    w = np.zeros(len(V))
    for c in centres:
        P = V - c
        dist = np.linalg.norm(P, axis=1)
        dot = np.einsum("ij,ij->i", vn, -P / np.maximum(dist[:, None], 1e-12))
        w = np.maximum(w, smoothstep(-0.05, 0.55, dot) * smoothstep(0.0195, 0.0175, dist))
    return w, vn


def build_head(data, eyeC_L, eyeC_R):
    V = data["verts"]
    F = data["faces"]
    mh_uv = data["uvs"][data["face_uv_idx"]]          # (Q,4,2) original MakeHuman UVs per corner
    M = 0.5 * (eyeC_L + eyeC_R)
    socket_w, _ = socket_weights(V, F, (eyeC_L, eyeC_R))

    # ---- spherical UV layout about the head centre (seam at the back of the head) ----
    hc = np.array([0.0, M[1] + 0.085, M[2] + 0.005])
    rel = V - hc
    radial = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 0], -rel[:, 1])             # 0 = straight ahead
    phi = np.arcsin(np.clip(rel[:, 2] / radial, -1, 1))   # elevation
    u = 0.5 + theta / (2 * np.pi)
    v = 0.5 + phi / np.pi
    quv = np.stack([u[F], v[F]], axis=-1)                  # (Q,4,2)
    span = quv[..., 0].max(1) - quv[..., 0].min(1)
    wrap = span > 0.5
    quv[wrap, :, 0] = np.where(quv[wrap, :, 0] < 0.5, quv[wrap, :, 0] + 1.0, quv[wrap, :, 0])

    # ---- albedo ----
    W, H = 4096, 2048
    tris = np.concatenate([np.stack([F[:, 0], F[:, 1], F[:, 2]], 1), np.stack([F[:, 0], F[:, 2], F[:, 3]], 1)])
    tuv = np.concatenate([quv[:, [0, 1, 2]], quv[:, [0, 2, 3]]])
    tmh = np.concatenate([mh_uv[:, [0, 1, 2]], mh_uv[:, [0, 2, 3]]])
    attr = np.concatenate([V[tris], tmh, radial[tris][..., None]], axis=2)   # xyz, mh uv, depth
    # duplicate wrapped triangles one period to the left so both edges of the seam are painted
    wr = tuv[..., 0].max(1) > 1.0
    tuv = np.concatenate([tuv, tuv[wr] - np.array([1.0, 0.0])])
    attr = np.concatenate([attr, attr[wr]])
    pos_map, _ = raster_attrs(tuv, attr, W, H)
    P = pos_map[..., :3].astype(np.float64)
    MHuv = pos_map[..., 3:5]

    def sample_mask(mask):
        m = mask.astype(np.float32) / 255.0
        hh, ww = m.shape
        xi = np.clip((MHuv[..., 0] * ww).astype(int), 0, ww - 1)
        yi = np.clip(((1 - MHuv[..., 1]) * hh).astype(int), 0, hh - 1)
        return m[yi, xi]

    ex = abs(eyeC_L[0])
    ez, ey = M[2], M[1]
    ax, dz, dy = np.abs(P[..., 0]), P[..., 2] - ez, P[..., 1] - ey     # dy < 0 : in front of eyes

    n_lo = FourierNoise(SEED + 1, 28, 25.0)(P)
    n_mid = FourierNoise(SEED + 2, 28, 90.0)(P)
    n_hi = FourierNoise(SEED + 3, 40, 900.0)(P)
    n_red = FourierNoise(SEED + 4, 20, 18.0)(P)

    skin = np.array([0.56, 0.36, 0.25])                    # sRGB, medium-brown South-Asian skin tone
    col = np.broadcast_to(skin, P.shape).copy()
    col *= (1 + 0.05 * n_lo + 0.025 * n_mid + 0.02 * n_hi)[..., None]
    # warmer/redder cheeks + nose, slightly darker periorbital area
    red = smoothstep(0.2, 1.0, n_red) * smoothstep(0.0, -0.06, dz) * smoothstep(0.01, -0.03, dy)
    col += (0.04 * red)[..., None] * np.array([1.0, -0.4, -0.5])
    peri = np.exp(-(((ax - ex) / 0.022) ** 2 + ((dz + 0.004) / 0.016) ** 2)) * smoothstep(0.02, -0.01, dy)
    col *= (1 - 0.10 * peri)[..., None]

    # lips (MPFB mask, remapped) and eyelid line
    if "mask_lips" in data:
        lips = smoothstep(0.15, 0.7, sample_mask(data["mask_lips"]))
        col = col * (1 - lips[..., None]) + np.array([0.50, 0.30, 0.25]) * (1 + 0.04 * n_hi[..., None]) * lips[..., None]
    if "mask_eyelids" in data:
        lids = smoothstep(0.1, 0.8, sample_mask(data["mask_eyelids"]))
        col *= (1 - 0.30 * lids)[..., None]

    # light beard shadow on the lower face
    beard = smoothstep(-0.048, -0.075, dz) * smoothstep(0.008, -0.004, dy + 0.0) \
        * (1 - smoothstep(0.75, 0.95, (ax / 0.06)))
    beard = np.maximum(beard, smoothstep(-0.050, -0.080, dz) * smoothstep(0.03, 0.06, ax) * smoothstep(0.075, 0.03, dy))
    stub = smoothstep(0.0, 0.9, n_hi * 0.5 + 0.5)
    col *= (1 - 0.13 * beard * (0.6 + 0.4 * stub))[..., None]

    # eyebrows: arched band above each orbit, hair-strand texture
    t = (ax - (ex - 0.017)) / 0.046                         # 0 at medial end, 1 at lateral tail
    zc = 0.021 + 0.0075 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.8) - 0.006 * np.clip(t - 0.6, 0, 1)
    half = 0.0042 * (1.0 - 0.55 * np.clip(t, 0, 1)) + 0.0006
    band = smoothstep(half + 0.0022, half - 0.0015, np.abs(dz - zc))
    ends = smoothstep(-0.05, 0.08, t) * smoothstep(1.05, 0.85, t)
    front = smoothstep(0.01, -0.012, dy)
    brow_density = band * ends * front
    brow = brow_hair_strokes(P, brow_density, t, ex, ez)
    hair_col = np.array([0.085, 0.062, 0.048])
    # soft underlying shadow (skin between hairs looks slightly darker in real brows)
    col *= (1 - 0.25 * smoothstep(0.0, 0.6, brow_density))[..., None]
    col = col * (1 - 0.90 * brow[..., None]) + hair_col * (0.90 * brow[..., None])

    # short (buzz-cut) scalp hair
    back = smoothstep(0.03, 0.11, dy)
    side = smoothstep(0.050, 0.075, ax)
    hairline = 0.064 - 0.006 * np.cos(np.clip(P[..., 0] / 0.05, -1.5, 1.5) * 2.0) - 0.030 * side - 0.065 * back
    ear_gap = np.exp(-(((dy - 0.075) / 0.032) ** 2 + ((dz + 0.005) / 0.03) ** 2)) * side
    scalp = smoothstep(hairline - 0.004, hairline + 0.006, dz) * (1 - ear_gap)
    # stubble: per-texel follicle noise (fine) x clumping (coarse) -> reads as hair, not paint
    follicle = smoothstep(-0.6, 0.9, FourierNoise(SEED + 7, 64, 6000.0, 0.4)(P))
    clumps = smoothstep(-0.9, 0.7, FourierNoise(SEED + 8, 24, 700.0)(P))
    dens = 0.84 + 0.16 * follicle * (0.75 + 0.25 * clumps)
    col = col * (1 - 0.86 * scalp[..., None] * dens[..., None]) + hair_col * (0.86 * scalp * dens)[..., None]

    albedo = np.clip(col, 0, 1).astype(np.float32)
    img = new_image("Face_Albedo", albedo)
    hmask = np.clip(np.maximum(scalp * dens, brow), 0, 1)[::2, ::2]
    mask_img = new_image("Face_HairMask", np.repeat(hmask[..., None], 3, axis=2))
    mask_img.colorspace_settings.name = "Non-Color"
    img["hair_mask_image"] = mask_img.name

    # ---- mesh ----
    me = mesh_from("Face", V, F, uv_loops=quv.reshape(-1, 2))
    att = me.attributes.new("socket", "FLOAT", "POINT")
    att.data.foreach_set("value", socket_w.astype(np.float32))
    obj = bpy.data.objects.new("Face", me)
    return obj, img, mask_img


# ======================================================================================
# eyes
# ======================================================================================
def build_eye(side, centre, iris_img, sclera_img, mats):
    R = CAL["eye_radius_m"]
    ri = CAL["iris_radius_m"]
    rp = CAL["pupil_radius_m"]
    rc = CAL["cornea_radius_m"]
    alpha = math.asin(ri / R)
    dL = R * math.cos(alpha)                     # forward distance of the limbus plane
    recess = 0.0006                              # iris sits behind the limbus plane (anterior chamber)
    seg = 96

    # sclera: sphere minus the corneal cap
    n = 56
    prof = [(R * math.sin(alpha + (math.pi - alpha) * i / (n - 1)), R * math.cos(alpha + (math.pi - alpha) * i / (n - 1))) for i in range(n)]
    prof[-1] = (0.0, -R)
    v_, f_, uv_ = lathe(prof, seg)
    sclera = bpy.data.objects.new(f"Eyeball.{side}", mesh_from(f"Eyeball.{side}", v_, f_, uv_))
    orient_outward(sclera)

    # iris: annulus from limbus to pupil margin, gently convex toward the pupil
    iris_prof = [(ri, dL - 0.00005), (ri * 0.985, dL - recess)]
    for i in range(1, 25):
        s = i / 24.0
        rho = ri * 0.985 + (rp - ri * 0.985) * s
        iris_prof.append((rho, dL - recess + 0.00035 * s ** 1.5))
    v_, f_, _ = lathe(iris_prof, seg)
    uv_ = []
    # planar UVs for the iris texture (disc radius ri maps to the full texture)
    for f in f_:
        for vi in f:
            x, _, z = v_[vi]
            uv_.append((0.5 + x / (2 * ri), 0.5 + z / (2 * ri)))
    iris = bpy.data.objects.new(f"Iris.{side}", mesh_from(f"Iris.{side}", v_, f_, uv_))
    orient_outward(iris, (0.0, 0.0, 0.0))

    # pupil: black disc just behind the pupil margin; ORIGIN = 3-D pupil centre
    d_pupil = dL - recess + 0.00035 - 0.00008
    pv, pf, _ = lathe([(rp * 1.08, 0.0), (rp * 0.5, 0.0), (0.0, 0.0)], 48)
    pupil = bpy.data.objects.new(f"Pupil.{side}", mesh_from(f"Pupil.{side}", pv, pf, None))
    orient_outward(pupil, (0.0, 0.0 + 0.001, 0.0))       # normals face forward (-Y local)

    # cornea: spherical cap over the limbus (apex ~13 mm from the rotation centre)
    cc = dL - math.sqrt(rc * rc - ri * ri)
    cprof = []
    for i in range(18):
        rho = ri * 1.01 * (1 - i / 17.0)
        cprof.append((rho, cc + math.sqrt(max(rc * rc - rho * rho, 0.0))))
    v_, f_, _ = lathe(cprof, seg)
    cornea = bpy.data.objects.new(f"Cornea.{side}", mesh_from(f"Cornea.{side}", v_, f_, None))
    orient_outward(cornea, (0.0, -cc, 0.0))

    for o, m in ((sclera, mats["sclera"]), (iris, mats[f"iris_{side}"]), (pupil, mats["pupil"]), (cornea, mats["cornea"])):
        o.data.materials.append(m)
        bpy.context.scene.collection.objects.link(o)

    # local placement (world matrices are fixed after bone parenting)
    geo = {"d_pupil": d_pupil, "apex": cc + rc, "limbus_d": dL}
    return sclera, iris, pupil, cornea, geo


# ======================================================================================
# materials
# ======================================================================================
def _sock(sockets, identifier):
    for so in sockets:
        if so.identifier == identifier:
            return so
    raise KeyError(identifier)


def mix_float(nt, fac, a, b):
    """Float lerp a -> b by fac (sockets or constants) using the generic Mix node (Blender >= 3.4)."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "FLOAT"
    for ident, val in (("Factor_Float", fac), ("A_Float", a), ("B_Float", b)):
        so = _sock(n.inputs, ident)
        if isinstance(val, (int, float)):
            so.default_value = val
        else:
            nt.links.new(val, so)
    return _sock(n.outputs, "Result_Float")


def mix_color(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    for ident, val in (("Factor_Float", fac), ("A_Color", a), ("B_Color", b)):
        so = _sock(n.inputs, ident)
        if isinstance(val, (int, float, tuple)):
            so.default_value = val
        else:
            nt.links.new(val, so)
    return _sock(n.outputs, "Result_Color")


def principled(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    return m, nt, bsdf


def build_materials(face_img, iris_imgs, sclera_img, hair_mask_img=None):
    mats = {}
    # --- skin (+ conjunctiva / caruncle / waterline blended by the per-vertex "socket" attribute) ---
    m, nt, b = principled("Skin")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = face_img
    tex.interpolation = "Cubic"
    sock = nt.nodes.new("ShaderNodeAttribute")
    sock.attribute_type = "GEOMETRY"
    sock.attribute_name = "socket"
    w = sock.outputs["Fac"]
    conj = (0.36, 0.085, 0.075, 1.0)                        # linear; moist pink-red mucosa
    nt.links.new(mix_color(nt, w, tex.outputs["Color"], conj), b.inputs["Base Color"])
    set_input(b, ["Subsurface Weight", "Subsurface"], 0.22)
    set_input(b, ["Subsurface Radius"], (1.0, 0.35, 0.20))
    set_input(b, ["Subsurface Scale"], 0.004)
    spec_in = "Specular IOR Level" if "Specular IOR Level" in b.inputs else "Specular"
    rough_skin, spec_skin = 0.48, 0.45
    if hair_mask_img is not None:
        # hair (scalp stubble + brows) is rough and only weakly specular
        mt = nt.nodes.new("ShaderNodeTexImage")
        mt.image = hair_mask_img
        r_map = nt.nodes.new("ShaderNodeMapRange")
        set_input(r_map, ["To Min"], 0.48)
        set_input(r_map, ["To Max"], 0.92)
        nt.links.new(mt.outputs["Color"], r_map.inputs["Value"])
        rough_skin = r_map.outputs["Result"]
        s_map = nt.nodes.new("ShaderNodeMapRange")
        set_input(s_map, ["To Min"], 0.45)
        set_input(s_map, ["To Max"], 0.10)
        nt.links.new(mt.outputs["Color"], s_map.inputs["Value"])
        spec_skin = s_map.outputs["Result"]
    # mucosa is wet: low roughness, a thin glossy coat -> tear-film highlight along the lid margins
    nt.links.new(mix_float(nt, w, rough_skin, 0.16), b.inputs["Roughness"])
    nt.links.new(mix_float(nt, w, spec_skin, 0.6), b.inputs[spec_in])
    coat_in = "Coat Weight" if "Coat Weight" in b.inputs else "Clearcoat"
    nt.links.new(mix_float(nt, w, 0.0, 0.45), b.inputs[coat_in])
    set_input(b, ["Coat Roughness", "Clearcoat Roughness"], 0.06)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    set_input(noise, ["Scale"], 900.0)
    set_input(noise, ["Detail"], 2.0)
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    set_input(bump, ["Strength"], 0.06)
    set_input(bump, ["Distance"], 0.0002)
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mats["skin"] = m

    # --- sclera (wet: clearcoat) ---
    m, nt, b = principled("Sclera")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = sclera_img
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    set_input(b, ["Roughness"], 0.35)
    set_input(b, ["Subsurface Weight", "Subsurface"], 0.10)
    set_input(b, ["Subsurface Scale"], 0.001)
    set_input(b, ["Coat Weight", "Clearcoat"], 0.5)
    set_input(b, ["Coat Roughness", "Clearcoat Roughness"], 0.06)
    # faint surface irregularity so the wet sclera highlight is not a perfect mirror
    coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    set_input(noise, ["Scale"], 2500.0)
    set_input(noise, ["Detail"], 3.0)
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    set_input(bump, ["Strength"], 0.03)
    set_input(bump, ["Distance"], 0.00005)
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mats["sclera"] = m

    # --- irises ---
    for side in ("L", "R"):
        m, nt, b = principled(f"Iris.{side}")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = iris_imgs[side][0]
        tex.interpolation = "Cubic"
        nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
        set_input(b, ["Roughness"], 0.55)
        set_input(b, ["Specular IOR Level", "Specular"], 0.2)
        # stromal relief from the height map (shading only; geometry and pupil position unchanged)
        ht = nt.nodes.new("ShaderNodeTexImage")
        ht.image = iris_imgs[side][1]
        ht.interpolation = "Cubic"
        bump = nt.nodes.new("ShaderNodeBump")
        set_input(bump, ["Strength"], 0.45)
        set_input(bump, ["Distance"], 0.00012)
        nt.links.new(ht.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        mats[f"iris_{side}"] = m

    # --- pupil ---
    m, nt, b = principled("Pupil")
    set_input(b, ["Base Color"], (0.004, 0.004, 0.004, 1.0))
    set_input(b, ["Roughness"], 1.0)
    set_input(b, ["Specular IOR Level", "Specular"], 0.0)
    mats["pupil"] = m

    # --- cornea: transparent, specular-only (no refraction -> geometric pupil == imaged pupil) ---
    m = bpy.data.materials.new("Cornea")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    set_input(gl, ["Roughness"], 0.015)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    set_input(lw, ["Blend"], 0.06)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Fresnel"], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    mats["cornea"] = m

    # --- eyelashes ---
    m, nt, b = principled("Eyelash")
    set_input(b, ["Base Color"], (0.010, 0.008, 0.007, 1.0))
    set_input(b, ["Roughness"], 0.45)
    set_input(b, ["Specular IOR Level", "Specular"], 0.35)
    mats["lash"] = m
    return mats


# ======================================================================================
# eyelashes (static, parented to the head bone)
# ======================================================================================
def lid_margins(face_obj, centre, side, cam_loc):
    """Upper / lower lid margin curves of one eye AS SEEN BY THE RENDER CAMERA.

    The subdivided (render-level) face mesh is ray-cast from the camera along radial lines in the
    image around the eye centre (1.5-degree sectors). Inside the palpebral aperture the rays pass
    the (absent from the BVH) eyeball and hit the socket far behind; the margin is the first radius
    at which a ray hits the front of the lid (>= 4 mm in front of the eye centre), i.e. the lid edge
    exactly as the camera sees it.
    Returns {"upper": (N,3), "lower": (N,3)} ordered temporal -> nasal."""
    from mathutils.bvhtree import BVHTree
    sub = face_obj.modifiers.get("Subdivision")
    old = sub.levels if sub else None
    if sub:
        sub.levels = sub.render_levels
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = face_obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        mw = face_obj.matrix_world
        co = [mw @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        w = np.empty(len(me.vertices))
        me.attributes["socket"].data.foreach_get("value", w)
    finally:
        ev.to_mesh_clear()
        if sub:
            sub.levels = old
    bvh = BVHTree.FromPolygons(co, polys)
    pw = np.array([w[list(pv)].mean() for pv in polys])
    c = Vector(tuple(centre))
    cam = Vector(tuple(cam_loc))
    sx = 1.0 if side == "L" else -1.0
    temporal, up = Vector((sx, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))
    out = {}
    for name, angles in (("upper", np.arange(8.0, 172.0, 1.5)), ("lower", np.arange(-8.0, -172.0, -1.5))):
        pts = []
        for a in angles:
            ar = math.radians(float(a))
            dirv = math.cos(ar) * temporal + math.sin(ar) * up
            for r in np.arange(0.0015, 0.0200, 0.00005):
                q = c + float(r) * dirv
                hit, nrm, fi, dist = bvh.ray_cast(cam, (q - cam).normalized())
                if hit is None:
                    continue
                if (c.y - hit.y) > 0.004:                      # first hit on the FRONT lid = visible margin
                    pts.append(np.array(hit))
                    break
        out[name] = np.array(pts)
    return out


def build_lashes(face_obj, eyeC, mat, head_parent, cam_loc):
    rng = np.random.default_rng(SEED + 41)
    verts, faces = [], []
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.array([0.0, -1.0, 0.0])
    sides_n = 3
    for side in ("L", "R"):
        c = np.asarray(eyeC[side])
        margins = lid_margins(face_obj, c, side, cam_loc)
        for lid, count, (len_lo, len_hi), curl_dir, lift in (
                ("upper", 160, (0.0060, 0.0090), up, 0.12),
                ("lower", 60, (0.0020, 0.0040), -up, 0.25)):
            curve = margins[lid]
            if len(curve) < 6:
                continue
            seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            for _ in range(count):
                u = rng.beta(1.1, 1.35) * 0.90 + 0.04          # slightly denser temporally, sparse nasally
                a_ = u * arc[-1]
                root = np.array([np.interp(a_, arc, curve[:, j]) for j in range(3)])
                outward = root - c
                outward[1] = 0.0
                outward /= max(np.linalg.norm(outward), 1e-9)
                root = root + 0.00012 * outward + 0.00010 * fwd   # on the lid skin just outside the visible margin
                taper_len = 1.0 - 0.45 * max(0.0, u - 0.7) / 0.3   # shorter towards the nasal corner
                L = rng.uniform(len_lo, len_hi) * taper_len
                d0 = fwd + lift * curl_dir * 0.6 + 0.25 * outward + rng.normal(0, 0.12, 3)
                d0 /= np.linalg.norm(d0)
                curl = 0.28 + rng.uniform(-0.08, 0.10)          # mostly towards the viewer, gentle upward curl
                r0, r1 = rng.uniform(0.055e-3, 0.075e-3), 0.010e-3
                nseg = 6
                ring_ids = []
                for k in range(nseg + 1):
                    t = k / nseg
                    pos = root + L * (t * d0 + curl * t * t * curl_dir)
                    tang = d0 + 2 * curl * t * curl_dir
                    tang /= np.linalg.norm(tang)
                    a_ax = np.cross(tang, [1.0, 0.0, 0.0])
                    if np.linalg.norm(a_ax) < 1e-6:
                        a_ax = np.cross(tang, [0.0, 1.0, 0.0])
                    a_ax /= np.linalg.norm(a_ax)
                    b_ax = np.cross(tang, a_ax)
                    rad = r0 + (r1 - r0) * t
                    ids = []
                    for j in range(sides_n):
                        ang = 2 * math.pi * j / sides_n
                        ids.append(len(verts))
                        verts.append(pos + rad * (math.cos(ang) * a_ax + math.sin(ang) * b_ax))
                    ring_ids.append(ids)
                for k in range(nseg):
                    for j in range(sides_n):
                        j2 = (j + 1) % sides_n
                        faces.append([ring_ids[k][j], ring_ids[k][j2], ring_ids[k + 1][j2], ring_ids[k + 1][j]])
                faces.append(ring_ids[0][::-1])
    me = mesh_from("Eyelashes", np.array(verts), faces, None, smooth=True)
    obj = bpy.data.objects.new("Eyelashes", me)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    head_parent(obj)
    return obj


# ======================================================================================
# main
# ======================================================================================
def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = d.to_track_quat("-Z", "Y")


def build(out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "EyeScene"
    data = dict(np.load(ASSET))
    eyeC = {"L": np.array(data["eye_center_L"]), "R": np.array(data["eye_center_R"])}
    M = 0.5 * (eyeC["L"] + eyeC["R"])

    # ----- textures + materials -----
    face_obj, face_img, hair_mask_img = build_head(data, eyeC["L"], eyeC["R"])
    iris_imgs = {}
    for side, sd in (("L", SEED + 11), ("R", SEED + 12)):
        col, height = make_iris_texture(sd)
        himg = new_image(f"Iris_{side}_Height", np.repeat(height[..., None], 3, axis=2))
        himg.colorspace_settings.name = "Non-Color"
        iris_imgs[side] = (new_image(f"Iris_{side}_Albedo", col), himg)
    sclera_img = new_image("Sclera_Albedo", make_sclera_texture(SEED + 13))
    mats = build_materials(face_img, iris_imgs, sclera_img, hair_mask_img)
    face_obj.data.materials.append(mats["skin"])
    scene.collection.objects.link(face_obj)
    sub = face_obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2

    # ----- armature: head + two eye bones (bone head = eyeball centre, +Y_bone = gaze) -----
    arm_data = bpy.data.armatures.new("EyeRig")
    arm = bpy.data.objects.new("EyeRig", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    head = eb.new("head")
    head.head = Vector((0.0, M[1] + 0.08, M[2] - 0.11))
    head.tail = Vector((0.0, M[1] + 0.08, M[2] + 0.10))
    for side in ("L", "R"):
        e = eb.new(f"eye.{side}")
        c = Vector(eyeC[side].tolist())
        e.head = c
        e.tail = c + Vector((0.0, -0.025, 0.0))   # bone +Y axis == primary-position gaze (-Y world)
        e.roll = 0.0
        e.parent = head
        e.use_connect = False
        e.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()

    def bone_parent(obj, bone, world):
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone
        bpy.context.view_layer.update()
        obj.matrix_world = world
        bpy.context.view_layer.update()

    bone_parent(face_obj, "head", Matrix.Identity(4))

    geo = {}
    for side in ("L", "R"):
        sclera, iris, pupil, cornea, g = build_eye(side, eyeC[side], iris_imgs[side], sclera_img, mats)
        geo[side] = g
        C = Matrix.Translation(Vector(eyeC[side].tolist()))
        bone_parent(sclera, f"eye.{side}", C)
        for child, parent, loc in ((iris, sclera, (0, 0, 0)), (cornea, sclera, (0, 0, 0)),
                                   (pupil, iris, (0, -g["d_pupil"], 0))):
            child.parent = parent
            child.matrix_parent_inverse = Matrix.Identity(4)
            child.location = loc
        bpy.context.view_layer.update()

    # ----- eyelashes: static strands rooted on the lid skin outside the margins -----
    cam_loc = Vector(M.tolist()) + CAL["camera_distance_m"] * Vector((0.0, -1.0, 0.0))
    build_lashes(face_obj, eyeC, mats["lash"], lambda o: bone_parent(o, "head", Matrix.Identity(4)), cam_loc)

    # ----- fixation target (empty, not rendered) -----
    fwd = Vector((0.0, -1.0, 0.0))
    Mv = Vector(M.tolist())
    tgt = bpy.data.objects.new("FixationTarget", None)
    tgt.empty_display_type = "SPHERE"
    tgt.empty_display_size = 0.005
    tgt.location = Mv + CAL["fixation_distance_m"] * fwd
    tgt.hide_render = True
    scene.collection.objects.link(tgt)

    # ----- camera: 40 cm in front of the eye midpoint, optical axis through the midpoint -----
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.lens = CAL["focal_length_mm"]
    cam_data.sensor_width = CAL["sensor_width_mm"]
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.02
    cam_data.clip_end = 5.0
    cam = bpy.data.objects.new("RenderCamera", cam_data)
    cam.location = Mv + CAL["camera_distance_m"] * fwd
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)           # looks along +Y (back at the face), up = +Z
    cam.lock_location = (True, True, True)
    cam.lock_rotation = (True, True, True)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # ----- lights (fixed) -----
    def area(name, offset, energy, size, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        lo.location = Mv + Vector(offset)
        scene.collection.objects.link(lo)
        look_at(lo, Mv)
        return lo

    area("KeyLight", (-0.30, -0.55, 0.28), 26.0, 0.55, (1.0, 0.96, 0.92))
    area("FillLight", (0.45, -0.45, 0.02), 9.0, 0.70, (0.95, 0.97, 1.0))
    area("RimLight", (0.28, 0.50, 0.12), 10.0, 0.40)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    set_input(bg, ["Color"], (0.20, 0.205, 0.21, 1.0))
    set_input(bg, ["Strength"], 0.6)
    scene.world = world

    # ----- render settings (deterministic) -----
    r = scene.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y = CAL["resolution"]
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.film_transparent = False
    r.use_persistent_data = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.use_stamp_filename = False          # no absolute .blend path in the PNG metadata
    c = scene.cycles
    c.device = "CPU"
    c.samples = CAL["samples"]
    c.use_adaptive_sampling = False
    c.seed = CAL["seed"]
    c.use_animated_seed = False
    c.use_denoising = True
    try:
        c.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    c.max_bounces = 8
    c.transparent_max_bounces = 8
    c.caustics_reflective = False
    c.caustics_refractive = False
    try:                                   # same look in every Blender version (shipped scene: Filmic)
        scene.view_settings.view_transform = "Filmic"
    except TypeError:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.frame_set(1)

    # ----- calibration record (single source of truth) -----
    cal = dict(CAL)
    cal.update({
        "eye_center_L": list(map(float, eyeC["L"])),
        "eye_center_R": list(map(float, eyeC["R"])),
        "eye_midpoint": list(map(float, M)),
        "interpupillary_distance_m": float(np.linalg.norm(eyeC["L"] - eyeC["R"])),
        "head_forward_axis": [0.0, -1.0, 0.0],
        "head_right_axis_image": [1.0, 0.0, 0.0],
        "head_up_axis": [0.0, 0.0, 1.0],
        "nominal_fixation": list(Mv + CAL["fixation_distance_m"] * fwd),
        "camera_location": list(cam.location),
        "camera_rotation_euler_xyz": list(cam.rotation_euler),
        "pupil_offset_local_m": geo["L"]["d_pupil"],
        "corneal_apex_from_centre_m": geo["L"]["apex"],
        "eye_L_definition": "subject's anatomical LEFT eye (world +X); appears on the RIGHT of the image",
        "eye_R_definition": "subject's anatomical RIGHT eye (world -X); appears on the LEFT of the image",
        "world_axes": "Blender world, Z-up; the face looks along -Y; +X = subject's left = image right",
        "asset_source": str(data.get("source", "")),
        "blender_version_built": bpy.app.version_string,
    })
    arm["calibration_json"] = json.dumps(cal, indent=1)
    cal_obj = bpy.data.objects.new("Calibration", None)
    cal_obj["calibration_json"] = arm["calibration_json"]
    scene.collection.objects.link(cal_obj)
    cal_obj.hide_render = True

    # organise
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    print("[build_scene] saved", out_path)
    return cal


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = os.path.abspath(argv[0]) if argv else os.path.join(ROOT, "scene.blend")
    build(out)
