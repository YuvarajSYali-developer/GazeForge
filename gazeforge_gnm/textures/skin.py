"""Procedural, photo-referenced skin textures in GNM's UDIM tile 1001.

All continuous fields are evaluated at each texel's 3D position on the GNM template
(so there are no UV seams); discrete features (pores, freckles, moles, stubble,
eyebrow hairs) are sampled on the 3D surface and splatted at their exact UVs.
Outputs per set: albedo (sRGB), height (bump), roughness, plus 3D hair strands
(eyebrows, eyelashes) that are also used as geometry in the Blender scene.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from .noise import fbm

TEXEL_MM_4096 = 0.117   # measured mean texel size on the face at 4096^2


def srgb_to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb8(x):
    x = np.clip(x, 0, 1)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def vertex_normals(V, faces):
    n = np.zeros_like(V)
    a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def catmull(points, n):
    P = np.concatenate([points[:1] * 2 - points[1:2], points, points[-1:] * 2 - points[-2:-1]])
    out = []
    segs = len(points) - 1
    for i in range(segs):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, n // segs, endpoint=(i == segs - 1)):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


class SkinContext:
    """Set-independent geometry. Per-texel fields are evaluated in chunks (memory-bounded at 4096^2);
    the same field functions are evaluated at mesh vertices for surface sampling."""

    def __init__(self, gnm, raster, landmarks):
        self.g, self.R, self.LM = gnm, raster, landmarks
        self.V = gnm.template.astype(np.float32)
        self.vn = vertex_normals(gnm.template, gnm.triangles).astype(np.float32)
        self.res = raster.res
        self.texel_mm = TEXEL_MM_4096 * 4096 / raster.res
        self.eyeL, self.eyeR = gnm.joints0[2], gnm.joints0[3]
        self.brow_y = float(landmarks[17:27, 1].mean())
        L = landmarks
        self.c_cheekL = (L[13] * 0.35 + L[45] * 0.25 + L[54] * 0.4)
        self.c_cheekR = (L[3] * 0.35 + L[36] * 0.25 + L[48] * 0.4)
        # KD-tree of texels around brows/eyes for snapping hair strands to the surface
        sel_all, Ps, Ns = [], [], []
        for sl in raster.chunks():
            P = raster.interp(self.V, sl)
            m = (P[:, 1] > 0.27) & (P[:, 1] < 0.37) & (P[:, 2] > 0.03)
            idx = np.arange(sl.start, sl.stop)[m]
            sel_all.append(idx); Ps.append(P[m])
            N = raster.interp(self.vn, sl)[m]
            Ns.append(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9))
        self.kd_idx = np.concatenate(sel_all)
        self.kd_P = np.concatenate(Ps); self.kd_N = np.concatenate(Ns)
        self.kd = cKDTree(self.kd_P)
        yx = raster.yx
        self.kd_xy = np.stack([yx[1][self.kd_idx] + 0.5, yx[0][self.kd_idx] + 0.5], 1).astype(np.float32)

    def texel_fields(self, sl):
        R = self.R
        P = R.interp(self.V, sl)
        N = R.interp(self.vn, sl)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        return self._fields(P, N, lambda n: R.interp(self.g.group(n), sl))

    def vertex_fields(self):
        return self._fields(self.V, self.vn, lambda n: self.g.group(n).astype(np.float32))

    def _fields(self, P, N, grp):
        L = self.LM
        f = {"P": P, "N": N}

        def blob(c, r):
            return np.exp(-np.sum((P - np.asarray(c, np.float32)) ** 2, 1) / (r * r)).astype(np.float32)
        f["face"] = grp("hockey_mask"); f["ears"] = grp("ears")
        f["lips"] = np.clip(grp("upper_lip") + grp("lower_lip"), 0, 1)
        f["sockets"] = grp("eye_sockets"); f["mouth"] = grp("mouth_sock")
        f["forehead"] = grp("forehead_region"); nose = grp("nose_region")
        f["cheeks"] = blob(self.c_cheekL, 0.024) + blob(self.c_cheekR, 0.024)
        f["nosetip"] = blob(L[30], 0.011) + 0.6 * blob(L[33], 0.009)
        f["chinb"] = blob(L[8] + np.array([0, 0.012, 0]), 0.016)
        f["lids"] = blob(self.eyeL + np.array([0, 0.004, 0.012]), 0.011) + blob(self.eyeR + np.array([0, 0.004, 0.012]), 0.011)
        f["undereye"] = blob(self.eyeL + np.array([0.004, -0.011, 0.012]), 0.010) + \
            blob(self.eyeR + np.array([-0.004, -0.011, 0.012]), 0.010)
        f["nostril"] = ((np.linalg.norm(P - L[33], axis=1) < 0.016) * (1 - f["face"]) * (N[:, 1] < 0.6)).astype(np.float32)
        f["tzone"] = np.clip(blob(L[27] + np.array([0, 0.035, 0]), 0.03) + nose, 0, 1)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        brow_y = self.brow_y
        front = smoothstep(-0.02, 0.05, z)
        hair_y = front * (brow_y + 0.056 - 14.0 * x * x) + (1 - front) * (0.255 + 0.02 * smoothstep(0.0, 0.06, np.abs(x)))
        side = smoothstep(0.058, 0.072, np.abs(x)) * smoothstep(-0.02, 0.0, z) * (1 - smoothstep(0.03, 0.05, z))
        hair_y = np.minimum(hair_y, np.where(side > 0, 0.30 - 0.03 * side, 9))
        sc = smoothstep(-0.0015, 0.004, y - hair_y + 0.002 * fbm(P, 0.004, 2, 5)) * (1 - f["ears"]) * \
            (1 - f["face"] * smoothstep(0.0, 0.01, hair_y - y + 0.01))
        f["scalp"] = np.clip(sc, 0, 1).astype(np.float32)
        upper = np.where(np.abs(x) < 0.03, L[33, 1] - 0.002, L[33, 1] - 0.002 - (np.abs(x) - 0.03) * 0.25)
        upper = np.minimum(upper, 0.302 - 0.01 * smoothstep(0.04, 0.07, np.abs(x)))
        upper = np.where(np.abs(x) > 0.042, np.minimum(upper, L[33, 1] + 0.004 - (np.abs(x) - 0.042) * 0.9 +
                                                        0.035 * smoothstep(0.055, 0.07, np.abs(x))), upper)
        bz = smoothstep(0.0, 0.006, upper - y) * smoothstep(L[8, 1] - 0.05, L[8, 1] - 0.025, y)
        bz *= smoothstep(-0.03, 0.0, z) * (1 - smoothstep(0.0, 0.5, f["lips"])) * (1 - f["ears"]) * (1 - f["nostril"])
        f["beard"] = np.clip(bz * (1 - f["scalp"]), 0, 1).astype(np.float32)
        return f

    def snap(self, pts):
        d, i = self.kd.query(pts)
        return self.kd_P[i], self.kd_N[i], self.kd_xy[i]


# ---------------------------------------------------------------------------
def splat(img, xy, radius, amp):
    """Add Gaussian dots (radius in px = sigma) at float pixel positions."""
    H, W = img.shape[:2]
    rad = np.maximum(radius, 0.35)
    K = np.ceil(rad * 2.5).astype(int)
    for k in np.unique(K):
        m = K == k
        oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
        cx = np.floor(xy[m, 0]).astype(int); cy = np.floor(xy[m, 1]).astype(int)
        px = cx[:, None, None] + ox[None]; py = cy[:, None, None] + oy[None]
        dx = px + 0.5 - xy[m, 0][:, None, None]; dy = py + 0.5 - xy[m, 1][:, None, None]
        w = np.exp(-(dx * dx + dy * dy) / (2 * rad[m][:, None, None] ** 2))
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        a = amp[m]
        if img.ndim == 2:
            np.add.at(img, (py[ok], px[ok]), (w * a[:, None, None])[ok])
        else:
            for c in range(img.shape[2]):
                np.add.at(img[..., c], (py[ok], px[ok]), (w * a[:, c][:, None, None])[ok])


def sample_surface(ctx, density_per_vertex, n, rng):
    """Sample n points on tile-1001 triangles with probability ~ area x density. Returns P, xy(px)."""
    g, R = ctx.g, ctx.R
    tris = R.tris
    V = g.template
    a = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    dens = density_per_vertex[tris].mean(1)
    p = a * dens
    p = p / p.sum()
    t = rng.choice(len(tris), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    P = (V[tris[t]] * b[..., None]).sum(1)
    sel = np.nonzero(g.triangle_tiles == 1001)[0]
    uv = (g.triangle_uvs[sel][t] * b[..., None]).sum(1)
    xy = np.stack([uv[:, 0] * R.res, (1 - uv[:, 1]) * R.res], 1)
    return P, xy


def vertex_field(ctx, fn):
    """Evaluate a function of (P) at vertices -> per-vertex density."""
    return fn(ctx.g.template)


# ---------------------------------------------------------------------------
def brow_strands(ctx, density, rng):
    """3D eyebrow hairs following the GNM brow landmarks; herringbone growth pattern."""
    L = ctx.LM
    strands = []
    for side, idx in (("R", [21, 20, 19, 18, 17]), ("L", [22, 23, 24, 25, 26])):
        pts = L[idx].astype(np.float64)
        curve = catmull(pts, 200)
        tang = np.gradient(curve, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        n_roots = int(650 * density)
        s = rng.random(n_roots) ** 0.9
        t = rng.uniform(-1, 1, n_roots)
        keep = rng.random(n_roots) < (1 - np.abs(t) ** 3) * (1 - 0.6 * s ** 4)
        s, t = s[keep], t[keep]
        width = 0.0032 + 0.0062 * (1 - s) ** 1.3
        ci = np.minimum((s * (len(curve) - 1)).astype(int), len(curve) - 1)
        c = curve[ci]; T = tang[ci]
        _, Nn, _ = ctx.snap(c)
        up = np.array([0, 1, 0.0]) - Nn * Nn[:, 1:2]; up /= np.linalg.norm(up, axis=1, keepdims=True)
        root = c - up * 0.0012 + up * (t * width / 2)[:, None]
        # growth angle relative to the lateral tangent (deg): head points up, body herringbone, tail down
        ang = np.where(s < 0.14, 80 - 220 * s, np.where(t < 0, 28, -8) * (1 - s) + (-18) * s)
        ang = np.radians(ang + rng.normal(0, 9, len(s)))
        length = np.where(s < 0.14, 0.0055, 0.0085) * rng.uniform(0.75, 1.25, len(s))
        for k in range(len(s)):
            p0, n0, _ = ctx.snap(root[k:k + 1]); p0, n0 = p0[0], n0[0]
            d = np.cos(ang[k]) * T[k] + np.sin(ang[k]) * up[k]
            d -= n0 * (d @ n0); d /= np.linalg.norm(d)
            pts_k = []
            p = p0.astype(np.float64)
            seg = length[k] / 6
            bend = rng.normal(0, 0.12)
            for j in range(7):
                sp, sn, _ = ctx.snap(p[None])
                lift = 0.00008 + 0.00035 * np.sin(np.pi * j / 6) ** 0.7
                pts_k.append(sp[0] + sn[0] * lift)
                d = d + bend * np.cross(sn[0], d) * 0.3 - sn[0] * (d @ sn[0])
                d /= np.linalg.norm(d)
                p = sp[0] + d * seg
            strands.append(np.array(pts_k))
    return strands


def lash_strands(ctx, rng, length_scale=1.0):
    """Eyelashes along the GNM lid-margin landmarks (upper: 36-39 / 42-45, lower: 39-41-36 / 45-47-42)."""
    L = ctx.LM
    strands, kinds = [], []
    specs = [("R", [36, 37, 38, 39], True), ("R", [39, 40, 41, 36], False),
             ("L", [45, 44, 43, 42], True), ("L", [42, 47, 46, 45], False)]
    for side, idx, upper in specs:
        eye = ctx.eyeL if side == "L" else ctx.eyeR
        curve = catmull(L[idx].astype(np.float64), 120)
        n = 110 if upper else 38
        s = np.sort(rng.random(n))
        # inner corner (high index for upper R/L sequences ending at the inner canthus) gets fewer, shorter lashes
        inner = s if idx[-1] in (39, 42) else 1 - s
        keep = rng.random(n) < (1 - 0.85 * smoothstep(0.75, 1.0, inner)) * (1 - 0.6 * smoothstep(0.0, 0.08, 0.08 - inner))
        s, inner = s[keep], inner[keep]
        ci = np.minimum((s * (len(curve) - 1)).astype(int), len(curve) - 1)
        for k, c in enumerate(curve[ci]):
            out = c - eye; out /= np.linalg.norm(out)       # radial from the eyeball centre
            fwd = np.array([0, 0, 1.0])
            vert = np.array([0, 1.0, 0]) if upper else np.array([0, -1.0, 0])
            lat = np.sign(c[0] - eye[0]) * np.array([1.0, 0, 0])
            prof = np.sin(np.pi * np.clip(0.15 + 0.85 * (1 - inner[k]), 0, 1)) ** 0.6
            ln = (0.0095 if upper else 0.0045) * (0.55 + 0.45 * prof) * rng.uniform(0.85, 1.15) * length_scale
            root = c + out * 0.0003 + vert * 0.0002
            d0 = 0.35 * out + 1.0 * fwd + (0.12 if upper else 0.05) * vert + 0.15 * lat * (1 - inner[k])
            d0 /= np.linalg.norm(d0)
            pts = [root]
            d = d0.copy()
            seg = ln / 7
            curl = (0.16 if upper else 0.10) + rng.normal(0, 0.03)
            for j in range(7):
                d = d + curl * vert + rng.normal(0, 0.03, 3)
                d /= np.linalg.norm(d)
                pts.append(pts[-1] + d * seg)
            strands.append(np.array(pts)); kinds.append(1 if upper else 0)
    return strands, kinds


# ---------------------------------------------------------------------------
def make_skin(ctx, spec, out_prefix, rng_seed=None):
    rng = np.random.default_rng(spec["seed"])
    res = ctx.res
    alb_img = np.zeros((res, res, 3), np.float32)
    h_img = np.zeros((res, res), np.float32)
    r_img = np.zeros((res, res), np.float32)
    sc = np.zeros((res, res), np.float32)
    beard_img = np.zeros((res, res), np.float32)
    for sl in ctx.R.chunks():
        f = ctx.texel_fields(sl)
        a, h, r = _pointwise(f, spec, ctx)
        ctx.R.put(alb_img, sl, a); ctx.R.put(h_img, sl, h); ctx.R.put(r_img, sl, r)
        ctx.R.put(sc, sl, f["scalp"]); ctx.R.put(beard_img, sl, f["beard"])
        del f, a, h, r
    vf = ctx.vertex_fields()
    return _features(ctx, spec, rng, out_prefix, alb_img, h_img, r_img, sc, beard_img, vf)


def _pointwise(f, spec, ctx):
    P = f["P"]
    n = len(P)
    sd = spec["seed"]
    base = srgb_to_lin(spec["skin_srgb"])
    tone_mult = {"warm": (1.02, 0.99, 0.93), "olive": (0.98, 1.0, 0.92),
                 "pink": (1.02, 0.97, 0.98), "neutral": (1.0, 1.0, 0.98)}[spec["undertone"]]
    alb = np.tile(base * np.array(tone_mult, np.float32), (n, 1)).astype(np.float32)
    dark = float(1.0 - base.mean() / 0.6)
    lum = 1 + 0.07 * fbm(P, 0.035, 3, sd + 1) + 0.045 * fbm(P, 0.007, 3, sd + 2) + 0.02 * fbm(P, 0.0018, 2, sd + 3)
    alb *= lum[:, None]
    red = (0.55 * f["cheeks"] + 0.65 * f["nosetip"] + 0.55 * f["ears"] + 0.3 * f["chinb"] + 0.35 * f["lids"] +
           0.12 * f["forehead"]) * (0.75 + 0.25 * fbm(P, 0.009, 3, sd + 5))
    red = np.clip(red * spec["redness"], 0, 1.3)
    if spec["redness"] > 1.0:
        cap = np.clip(fbm(P, 0.003, 3, sd + 6, ridged=True) - 0.55, 0, 1) * 2.5
        red += cap * (f["cheeks"] + f["nosetip"]) * 0.35 * (spec["redness"] - 1.0) * 3
    red_tint = np.array([1.0, 0.70, 0.70], np.float32)
    alb *= (1 - (red * (0.55 - 0.3 * dark))[:, None] * (1 - red_tint))
    pe = np.clip(f["undereye"] * (0.6 + 0.4 * fbm(P, 0.006, 2, sd + 7)), 0, 1)
    alb *= (1 - pe[:, None] * (1 - np.array([0.86, 0.80, 0.84], np.float32)) * (0.8 + 0.4 * spec["age"]))
    alb *= (1 - f["lids"][:, None] * 0.06)
    if spec["age"] > 0.4:
        spots = np.clip(fbm(P, 0.004, 3, sd + 8) - 0.35, 0, 1) * 3 * (f["cheeks"] + 0.6 * f["forehead"])
        alb *= (1 - np.clip(spots, 0, 1)[:, None] * (1 - np.array([0.86, 0.80, 0.74], np.float32)) * (spec["age"] - 0.3))
    lipw = smoothstep(0.15, 0.85, f["lips"] + 0.12 * fbm(P, 0.0015, 2, sd + 9))
    lipcol = alb * np.array([0.80, 0.50, 0.49], np.float32) * (1 - 0.18 * dark)
    lipcol *= (1 + 0.06 * fbm(P, 0.0012, 2, sd + 10))[:, None]
    alb = alb * (1 - lipw[:, None]) + lipcol * lipw[:, None]
    wet = np.array([0.50, 0.17, 0.15], np.float32) * (1 - 0.3 * dark)
    sock = smoothstep(0.2, 0.8, f["sockets"])
    alb = alb * (1 - sock[:, None]) + wet * sock[:, None]
    m = smoothstep(0.2, 0.8, f["mouth"])
    alb = alb * (1 - m[:, None]) + np.array([0.22, 0.05, 0.05], np.float32) * m[:, None]
    alb *= (1 - f["nostril"][:, None] * np.array([0.55, 0.7, 0.7], np.float32))
    height = np.zeros(n, np.float32)
    micro = fbm(P, 0.0011, 3, sd + 11, ridged=True)
    grain = fbm(P, 0.00035, 2, sd + 12)
    relief_w = (1 - 0.6 * lipw) * (1 - 0.7 * sock) * (0.6 + 0.4 * f["face"])
    height += relief_w * (0.22 * micro + 0.14 * grain)
    lipline = np.cos(P[:, 0] * 2 * np.pi / 0.0011 + 3 * fbm(P, 0.002, 2, sd + 13))
    height += lipw * 0.07 * lipline
    if spec["age"] > 0.2:
        a = spec["age"]
        fy = P[:, 1] - ctx.brow_y
        lines = np.zeros(n, np.float32)
        for k, off in enumerate((0.012, 0.019, 0.026, 0.033)):
            yy = off + 0.0015 * fbm(P, 0.01, 2, sd + 20 + k) + 1.5 * P[:, 0] ** 2
            lines += np.exp(-((fy - yy) / 0.0006) ** 2) * smoothstep(0.045, 0.02, np.abs(P[:, 0]))
        height -= lines * f["forehead"] * 1.2 * a
        for eye, sgn in ((ctx.eyeL, 1), (ctx.eyeR, -1)):
            c = eye + np.array([sgn * 0.017, 0.0, 0.004])
            d = P - c
            r = np.linalg.norm(d[:, :2], axis=1)
            th = np.arctan2(d[:, 1], d[:, 0] * sgn)
            crow = (np.cos(th * 9 + 2 * fbm(P, 0.005, 2, sd + 30)) * 0.5 + 0.5) ** 6
            crow *= smoothstep(0.002, 0.006, r) * smoothstep(0.016, 0.008, r) * (np.abs(th) < 0.9)
            height -= crow * 0.9 * a
            ue = np.exp(-((P[:, 1] - (eye[1] - 0.0085 - 0.6 * (P[:, 0] - eye[0]) ** 2 * 40)) / 0.0005) ** 2)
            ue *= smoothstep(0.012, 0.004, np.abs(P[:, 0] - eye[0] - sgn * 0.002))
            height -= ue * 0.6 * a
    rough = 0.56 + 0.05 * fbm(P, 0.01, 2, sd + 14) - 0.08 * f["tzone"] - 0.12 * lipw - 0.38 * sock
    return alb, height, rough.astype(np.float32)


def _features(ctx, spec, rng, out_prefix, alb_img, h_img, r_img, sc, beard_img, vf):
    sd = spec["seed"]
    dark = float(1.0 - srgb_to_lin(spec["skin_srgb"]).mean() / 0.6)
    # ---- discrete features splatted in UV --------------------------------
    tex = ctx.texel_mm
    Vt = ctx.g.template
    gv = lambda name: ctx.g.group(name)  # noqa: E731
    # pores: density per vertex by region
    dens = (0.35 * gv("hockey_mask") + 0.9 * gv("nose_region") + 0.6 * (gv("left_cheek_region") + gv("right_cheek_region") +
            gv("left_zygomatic_region") + gv("right_zygomatic_region")) + 0.35 * gv("forehead_region") +
            0.4 * gv("chin_region") + 0.08) * (1 - gv("upper_lip") - gv("lower_lip")).clip(0, 1) * (1 - gv("eye_sockets"))
    npores = 140000
    _, xy = sample_surface(ctx, dens.astype(np.float64) + 1e-6, npores, rng)
    rad = rng.uniform(0.55, 1.15, npores) * (0.117 / tex)
    hp = np.zeros_like(h_img); splat(hp, xy, rad, -rng.uniform(0.6, 1.2, npores))
    h_img += hp
    alb_img *= (1 + 0.07 * np.clip(hp, -1.5, 0))[..., None]
    # beard stubble (follicles) and shadow
    if spec["beard"] > 0:
        hair = srgb_to_lin(spec["hair_srgb"])
        shadow = beard_img * spec["beard"]
        shadow = cv2.GaussianBlur(shadow, (0, 0), 6)
        alb_img *= (1 - shadow[..., None] * (1 - np.array([0.80, 0.80, 0.84], np.float32)) * (0.9 - 0.6 * dark))
        bv = vf["beard"].astype(np.float64)
        nb = int(90000 * spec["beard"])
        _, bxy = sample_surface(ctx, bv + 1e-9, nb, rng)
        follic = np.zeros_like(h_img); splat(follic, bxy, rng.uniform(0.45, 0.8, nb) * (0.117 / tex), rng.uniform(0.5, 1.0, nb))
        follic = np.clip(follic, 0, 1)
        alb_img = alb_img * (1 - follic[..., None] * 0.75) + hair * follic[..., None] * 0.75
        h_img += follic * 0.25
        r_img += follic * 0.15
    # freckles
    if spec["freckles"] > 0:
        fd = (gv("nose_region") + 0.8 * (gv("left_cheek_region") + gv("right_cheek_region") + gv("left_zygomatic_region") +
              gv("right_zygomatic_region")) + 0.25 * gv("forehead_region")) * gv("hockey_mask")
        nf = int(1800 * spec["freckles"])
        _, fxy = sample_surface(ctx, fd + 1e-6, nf, rng)
        fr = np.zeros(alb_img.shape[:2], np.float32)
        splat(fr, fxy, rng.uniform(2.0, 6.0, nf) * (0.117 / tex), rng.uniform(0.25, 0.7, nf))
        fr = np.clip(fr, 0, 1)
        alb_img *= (1 - fr[..., None] * (1 - np.array([0.72, 0.56, 0.45], np.float32)))
    # moles
    nm = rng.integers(3, 7)
    _, mxy = sample_surface(ctx, gv("hockey_mask") + 0.05, nm, rng)
    mo = np.zeros(alb_img.shape[:2], np.float32)
    splat(mo, mxy, rng.uniform(3, 6, nm) * (0.117 / tex), np.ones(nm) * 1.4)
    mo = np.clip(mo, 0, 1)
    alb_img *= (1 - mo[..., None] * (1 - np.array([0.42, 0.30, 0.24], np.float32) * (1 - 0.4 * dark)))
    h_img += mo * 0.4

    # scalp hair (short / buzz length) – anisotropic strands painted in 3D noise
    if sc.max() > 0:
        hair = srgb_to_lin(spec["hair_srgb"])
        hn = np.zeros_like(sc)
        for sl in ctx.R.chunks():
            ctx.R.put(hn, sl, fbm(ctx.R.interp(ctx.V, sl), 0.0006, 2, sd + 15, ridged=True))
        hcol = hair[None, None] * (0.75 + 0.5 * hn[..., None])
        cover = np.clip(sc * (0.85 + 0.15 * hn), 0, 1)
        alb_img = alb_img * (1 - cover[..., None]) + hcol * cover[..., None]
        r_img = r_img * (1 - cover) + 0.55 * cover
        h_img += cover * 0.3 * hn

    # eyebrows: hairs painted along the same 3D strands used for geometry
    strands = brow_strands(ctx, spec["brow_density"], rng)
    hair = srgb_to_lin(spec["hair_srgb"])
    bm = np.zeros(alb_img.shape[:2], np.float32)
    for s in strands:
        _, _, xy = ctx.snap(s)
        pts = np.round(xy * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(bm, [pts], False, 1.0, thickness=1, lineType=cv2.LINE_AA, shift=4)
    bm = np.clip(bm, 0, 1)
    under = cv2.GaussianBlur(bm, (0, 0), 7) * 1.6
    alb_img *= (1 - np.clip(under, 0, 1)[..., None] * 0.18)
    alb_img = alb_img * (1 - bm[..., None] * 0.85) + hair * 0.8 * bm[..., None] * 0.85
    h_img += bm * 0.3
    lashes, lash_kind = lash_strands(ctx, rng)

    # hair coverage mask (scalp hair, brows, stubble): no subsurface scattering through hair
    hmask = np.clip(sc * 0.95 + bm * 0.9 + (follic if spec["beard"] > 0 else 0) * 0.8, 0, 1)
    cv2.imwrite(f"{out_prefix}_hairmask.1001.png", (ctx.R.pad(hmask) * 255 + 0.5).astype(np.uint8))
    # pad gutters so filtering never pulls in background (seam-free at all mip levels)
    alb_img = ctx.R.pad(np.clip(alb_img, 0, 1))
    h_img = ctx.R.pad(h_img)
    r_img = ctx.R.pad(np.clip(r_img, 0.08, 0.95))
    # high-pass: keep only meso/micro relief (large-scale shape comes from the GNM geometry), so
    # region-dependent pore density cannot create a low-frequency slope in the bump map
    h_img = h_img - cv2.GaussianBlur(h_img, (0, 0), 24 * ctx.res / 4096)
    hn_ = (h_img - np.percentile(h_img, 0.5)) / (np.percentile(h_img, 99.5) - np.percentile(h_img, 0.5))
    cv2.imwrite(f"{out_prefix}_albedo.1001.png", cv2.cvtColor(lin_to_srgb8(alb_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out_prefix}_height.1001.png", (np.clip(hn_, 0, 1) * 65535).astype(np.uint16))
    cv2.imwrite(f"{out_prefix}_roughness.1001.png", (r_img * 255 + 0.5).astype(np.uint8))
    return strands, lashes, lash_kind
