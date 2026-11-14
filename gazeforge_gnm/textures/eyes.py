"""Iris + sclera textures for GNM eye-interior tiles (1002 left, 1004 right), teeth/gums (1006),
tongue (1007) and cornea tiles (1003/1005). Iris structure is built in the eye's own polar
coordinates (radius from the optical axis, angle around it) computed from 3D texel positions,
so the left and right irises are distinct, high-resolution and correctly centred."""
import numpy as np
import cv2
from .noise import fbm, Perlin3
from .skin import srgb_to_lin, lin_to_srgb8, smoothstep
from .uvraster import TileRaster


def polar_noise(r, th, scale_r, n_ang, seed, octaves=3):
    """Noise that is periodic in angle: sample 3D noise on a cylinder (cos, sin, r)."""
    pts = np.stack([np.cos(th) * n_ang / (2 * np.pi), np.sin(th) * n_ang / (2 * np.pi), r / scale_r], 1).astype(np.float32)
    tot, amp, norm = 0, 1.0, 0
    for o in range(octaves):
        tot = tot + amp * Perlin3(seed * 31 + o)(pts * (2 ** o))
        norm += amp
        amp *= 0.5
    return tot / norm


def make_eye(gnm, tile, joint_index, palette, seed, res, out_prefix):
    R = TileRaster(gnm, tile, res)
    c = gnm.joints0[joint_index]
    P = (R.P - c).astype(np.float32)
    rng = np.random.default_rng(seed)
    V = gnm.template - c
    side = "left_eye" if joint_index == 2 else "right_eye"
    m_iris = (gnm.group("irises") > 0.5) & (gnm.group(side) > 0.5)
    m_pup = (gnm.group("pupils") > 0.5) & (gnm.group(side) > 0.5)
    r_iris = np.hypot(V[m_iris, 0], V[m_iris, 1]).max()
    r_pup = np.hypot(V[m_pup, 0], V[m_pup, 1]).max()
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(y, x) * (1 if joint_index == 2 else -1)   # mirror the pattern between eyes
    front = z > 0.0115          # the iris disc lies at z ~ 12.8-13.3 mm
    rn = np.clip((r - r_pup) / (r_iris - r_pup), 0, 1.2)   # 0 at pupil edge, 1 at limbus

    pal = {k: srgb_to_lin(v) for k, v in palette.items() if k != "crypt"}
    n = len(P)
    # --- sclera: slightly warm white, bluish deep, pinker towards the canthi (exposed nasal/temporal) ---
    scl = np.tile(srgb_to_lin((236, 228, 220)), (n, 1)).astype(np.float32)
    scl *= (1 + 0.04 * fbm(P, 0.004, 3, seed + 1))[:, None]
    horiz = np.abs(x) / 0.0145
    pink = smoothstep(0.35, 0.95, horiz) * smoothstep(-0.006, 0.006, z)
    scl *= (1 - pink[:, None] * (1 - np.array([1.0, 0.82, 0.78], np.float32)) * 0.6)
    yel = smoothstep(0.0, 0.004, -y) * 0.15
    scl *= (1 - yel[:, None] * (1 - np.array([1.0, 0.95, 0.85], np.float32)))
    # --- iris ---
    fib = polar_noise(rn, th, 0.45, 150, seed + 2, 3)            # radial fibres (high angular freq)
    fib2 = polar_noise(rn, th, 0.12, 60, seed + 3, 3)
    blobs = polar_noise(rn, th, 0.25, 24, seed + 4, 2)
    collar_r = 0.42 + 0.06 * polar_noise(rn * 0, th, 1, 14, seed + 5, 2)     # zig-zag collarette
    inner = smoothstep(collar_r + 0.03, collar_r - 0.03, rn)
    iris = pal["base"] * (1 - inner)[:, None] + pal["inner"] * inner[:, None]
    f = np.clip(0.5 + 1.6 * fib + 0.6 * fib2, 0, 1)
    iris = iris * (0.70 + 0.45 * f)[:, None] + pal["fibre"] * (np.clip(fib, 0, 1) ** 2 * 0.55)[:, None]
    # collarette ridge highlight
    ridge = np.exp(-((rn - collar_r) / 0.025) ** 2)
    iris = iris * (1 + 0.35 * ridge)[:, None]
    # crypts (dark lacunae) mostly in the ciliary zone just outside the collarette
    crypt = np.clip(-blobs - 0.25, 0, 1) * 2.5 * smoothstep(collar_r, collar_r + 0.08, rn) * smoothstep(0.95, 0.7, rn)
    iris *= (1 - palette["crypt"] * 0.55 * np.clip(crypt, 0, 1))[:, None]
    # concentric contraction furrows in the outer zone
    fur = (np.cos(rn * 2 * np.pi * 5.5 + 2 * blobs) * 0.5 + 0.5) ** 8 * smoothstep(0.6, 0.85, rn)
    iris *= (1 - 0.25 * fur)[:, None]
    # limbal ring and pupillary ruff
    limb = smoothstep(0.78, 1.0, rn)
    iris = iris * (1 - limb[:, None] * 0.85) + pal["limbus"] * (limb * 0.85)[:, None]
    ruff = np.exp(-((rn - 0.02) / 0.03) ** 2) * (0.6 + 0.4 * (fib > 0))
    iris = iris * (1 - 0.6 * ruff)[:, None] + srgb_to_lin((40, 24, 14)) * (0.6 * ruff)[:, None]
    # compose: pupil (black) / iris / limbal blend into sclera
    w_iris = (front & (r < r_iris * 1.0)).astype(np.float32)
    blend = smoothstep(r_iris * 1.07, r_iris * 0.97, r) * smoothstep(0.0095, 0.0118, z)
    col = scl * (1 - blend[:, None]) + iris * blend[:, None]
    # dark limbal shadow on the sclera side (limbal ring extends ~0.5 mm outside the iris)
    halo = np.exp(-((r - r_iris * 1.04) / 0.0006) ** 2) * smoothstep(0.009, 0.012, z)
    col *= (1 - 0.45 * halo)[:, None]
    pup = smoothstep(r_pup * 1.02, r_pup * 0.96, r) * front
    col = col * (1 - pup[:, None]) + np.array([0.004, 0.004, 0.005], np.float32) * pup[:, None]
    img = R.to_image(col)
    # conjunctival vessels: random-walk polylines on the eyeball, drawn at their exact UVs
    from scipy.spatial import cKDTree
    kd = cKDTree(P)
    yx = np.stack(R.yx, 1)
    xy_t = np.stack([yx[:, 1] + 0.5, yx[:, 0] + 0.5], 1)
    vimg = np.zeros(img.shape[:2], np.float32)
    rr = 0.0145
    for k in range(70):
        side_s = rng.choice([-1, 1])
        a = np.radians(rng.uniform(55, 95)) * side_s      # start near the canthi (nasal / temporal)
        e = np.radians(rng.uniform(-25, 25))
        d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        step = np.radians(1.2)
        heading = np.array([-side_s * 1.0, rng.normal(0, 0.6), 0.6])
        pts = []
        L = rng.integers(10, 40)
        width = rng.uniform(0.6, 1.8) * res / 2048
        for j in range(L):
            p = d * rr
            pts.append(p)
            heading = heading + rng.normal(0, 0.45, 3)
            heading -= d * (heading @ d)
            heading /= np.linalg.norm(heading)
            d = d + heading * step
            d /= np.linalg.norm(d)
            if np.hypot(d[0], d[1]) * rr < r_iris * 1.25 and d[2] > 0:
                break
        d_kd, idx = kd.query(np.array(pts))
        xy = xy_t[idx]
        jump = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if np.any(jump > res * 0.02) or np.any(d_kd > 0.0004):
            continue
        cv2.polylines(vimg, [np.round(xy * 16).astype(np.int32).reshape(-1, 1, 2)], False,
                      float(rng.uniform(0.35, 0.8)), thickness=max(1, int(round(width))), lineType=cv2.LINE_AA, shift=4)
    vimg = cv2.GaussianBlur(vimg, (0, 0), 0.6 * res / 2048)
    vessel = np.array([0.55, 0.06, 0.05], np.float32)
    img = img * (1 - vimg[..., None] * 0.7) + vessel * vimg[..., None] * 0.7
    img = R.pad(img)
    cv2.imwrite(f"{out_prefix}_albedo.{tile}.png", cv2.cvtColor(lin_to_srgb8(img), cv2.COLOR_RGB2BGR))
    return dict(r_iris_mm=float(r_iris * 1e3), r_pupil_mm=float(r_pup * 1e3))


def make_cornea(tile, res, out_prefix):
    img = np.full((res, res, 3), 255, np.uint8)
    cv2.imwrite(f"{out_prefix}_albedo.{tile}.png", img)


def make_teeth(gnm, res, seed, out_prefix, tooth_srgb=(232, 222, 198), gum_srgb=(196, 98, 96)):
    R = TileRaster(gnm, 1006, res)
    P = R.P.astype(np.float32)
    teeth = R.interp(gnm.group("teeth")).astype(np.float32)
    t = smoothstep(0.3, 0.7, teeth)
    tc = srgb_to_lin(tooth_srgb) * (1 + 0.05 * fbm(P, 0.003, 2, seed))[:, None]
    gc = srgb_to_lin(gum_srgb) * (1 + 0.08 * fbm(P, 0.002, 2, seed + 1))[:, None]
    col = tc * t[:, None] + gc * (1 - t)[:, None]
    img = R.pad(R.to_image(col))
    cv2.imwrite(f"{out_prefix}_albedo.1006.png", cv2.cvtColor(lin_to_srgb8(img), cv2.COLOR_RGB2BGR))


def make_tongue(gnm, res, seed, out_prefix):
    R = TileRaster(gnm, 1007, res)
    P = R.P.astype(np.float32)
    base = srgb_to_lin((190, 92, 96))
    pap = np.clip(fbm(P, 0.0007, 2, seed, ridged=True), 0, 1)
    col = base * (0.85 + 0.25 * pap)[:, None] * (1 + 0.06 * fbm(P, 0.006, 2, seed + 1))[:, None]
    img = R.pad(R.to_image(col))
    cv2.imwrite(f"{out_prefix}_albedo.1007.png", cv2.cvtColor(lin_to_srgb8(img), cv2.COLOR_RGB2BGR))
