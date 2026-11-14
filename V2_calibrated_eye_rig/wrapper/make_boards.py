#!/usr/bin/env python3
"""GazeForge presentation boards.

Generates five scientific figures ("boards") from a GazeForge output package:

  1. sequence_board.png        clockwise 10-frame sequence on a clock layout, with eye-region insets
  2. direction_board.png       5-direction validation (NEUTRAL/LEFT/RIGHT/UP/DOWN/RESET) at true scale
  3. eye_detail.png            native eye crops (x4 Lanczos), neutral + 4 extremes, both eyes
     eye_detail_before_after.png   (only with --before) same crops, older package vs this one
  4. camera_calibration.png    to-scale top/side engineering drawing of eyes, target plane and camera
  5. gaze_rays.png             gaze rays to fixation targets, sequence targets on the fixation plane

Each board is written as PNG (>= 2400 px long side) and PDF.

Usage:
  python3 make_boards.py --pkg <package_dir> --out <output_dir> [--before <older_package_dir>]

Only numpy, Pillow and matplotlib are required. Every landmark drawn comes from the package's
metadata.json / reports/direction_validation.json, or is computed from documented formulas and
labelled as such. The package is opened read-only.
"""
import argparse
import json
import math
import os
import re
import sys
import textwrap

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches, patheffects  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

# --------------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------------
INK = "#2b2b2b"          # main text
INK2 = "#5f6368"         # secondary text
GRID = "#d9dde3"
PANEL_BG = "#ffffff"
FIG_BG = "#ffffff"
ACC = "#1f5a96"          # accent (dark blue)
ACC_MID = "#5b8fc7"
ACC_LIGHT = "#a9c6e6"
ACC_FILL = "#eef4fa"
RED = "#c62828"          # reference / neutral
GREEN = "#2e7d32"        # measured / ground-truth pupil

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.8,
    "savefig.facecolor": FIG_BG,
    "pdf.fonttype": 42,
    "svg.hashsalt": "gazeforge",
})
MONO = "DejaVu Sans Mono"

HALO = [patheffects.withStroke(linewidth=3.0, foreground="white")]
HALO_THIN = [patheffects.withStroke(linewidth=2.0, foreground="white")]
DARK_HALO = [patheffects.withStroke(linewidth=2.5, foreground="#222222")]


# --------------------------------------------------------------------------------------------
# Package loading
# --------------------------------------------------------------------------------------------
class Package:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, "metadata.json")) as fh:
            self.meta = json.load(fh)
        self.meta = sorted(self.meta, key=lambda r: r["frame"])
        dv = os.path.join(self.root, "reports", "direction_validation.json")
        self.dirval = None
        if os.path.isfile(dv):
            with open(dv) as fh:
                self.dirval = json.load(fh)
        self.readme = ""
        rp = os.path.join(self.root, "README.md")
        if os.path.isfile(rp):
            with open(rp, encoding="utf-8", errors="replace") as fh:
                self.readme = fh.read()
        self._parse_constants()
        self._geometry()

    # -- constants that are not in metadata: parsed from README, with documented fallbacks --
    def _parse_constants(self):
        def grab(pattern, default):
            m = re.search(pattern, self.readme)
            return (float(m.group(1)), "README") if m else (default, "default (README not parsed)")

        self.lens_mm, self.lens_src = grab(r"([\d.]+)\s*mm lens", 62.0)
        self.sensor_mm, self.sensor_src = grab(r"([\d.]+)\s*mm sensor", 36.0)
        self.sclera_mm, self.sclera_src = grab(r"sclera sphere r\s*=\s*([\d.]+)\s*mm", 12.5)
        self.iris_mm, _ = grab(r"iris annulus \(r\s*=\s*([\d.]+)\s*mm", 5.9)
        self.pupil_mm, _ = grab(r"pupil disc r\s*=\s*([\d.]+)\s*mm", 2.0)

    def _geometry(self):
        r0 = self.meta[0]
        ex = r0["extra"]
        self.K = np.array(ex["camera_intrinsics_K"], float)
        self.Rt = np.array(ex["camera_extrinsics_Rt_world_to_camera"], float)
        self.P = np.array(r0["camera_matrix"], float)
        self.c2w = np.array(ex["camera_to_world_blender"], float)
        self.W, self.H = [int(v) for v in ex["image_size"]]
        self.CL = np.array(ex["eye_center_3D_L"], float)
        self.CR = np.array(ex["eye_center_3D_R"], float)
        self.M = np.array(ex["eye_midpoint_3D"], float)
        self.cam = self.c2w[:3, 3].copy()
        # Blender camera: local +X = image right, +Y = image up, looks along local -Z.
        self.r = self.c2w[:3, 0] / np.linalg.norm(self.c2w[:3, 0])
        self.u = self.c2w[:3, 1] / np.linalg.norm(self.c2w[:3, 1])
        look = -self.c2w[:3, 2]
        self.f = -look / np.linalg.norm(look)        # head forward = from the face towards camera
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]
        # fixation distance and target bounds derived from metadata targets
        T = np.array([m["extra"]["fixation_target"] for m in self.meta], float)
        h = np.array([m["input"]["h"] for m in self.meta], float)
        v = np.array([m["input"]["v"] for m in self.meta], float)
        self.targets = T
        self.h_in, self.v_in = h, v
        rel = T - self.M
        self.d_fix = float(np.median(rel @ self.f))
        mh, mv = np.abs(h) > 0.3, np.abs(v) > 0.3
        self.bound_h = float(np.median((rel[mh] @ self.r) / h[mh])) if mh.any() else 0.05
        self.bound_v = float(np.median((rel[mv] @ self.u) / v[mv])) if mv.any() else 0.05
        self.F0 = self.M + self.d_fix * self.f
        self.d_cam = float(np.linalg.norm(self.cam - self.M))
        # depth of the eye centres in camera coordinates (for px <-> mm)
        self.Z_eye = float(np.mean([(self.Rt @ np.append(c, 1.0))[2] for c in (self.CL, self.CR)]))
        self.sclera_px = self.fx * self.sclera_mm / 1000.0 / self.Z_eye
        e2L = np.array([m["extra"]["eye_center_2D_L"] for m in self.meta], float)
        e2R = np.array([m["extra"]["eye_center_2D_R"] for m in self.meta], float)
        self.e2L = np.median(e2L, axis=0)
        self.e2R = np.median(e2R, axis=0)

    def project(self, X):
        p = self.P @ np.append(np.asarray(X, float), 1.0)
        return p[:2] / p[2]

    def frame_image(self, rec):
        rel = rec["extra"].get("image") or "renders/frame_%02d.png" % rec["frame"]
        return load_rgb(os.path.join(self.root, rel))

    def dir_image(self, name):
        p = os.path.join(self.root, "reports", "direction_test", name + ".png")
        return load_rgb(p) if os.path.isfile(p) else None

    def coords(self, X):
        """(forward s, lateral x, vertical z) of world point X relative to M, in cm."""
        d = np.asarray(X, float) - self.M
        return 100 * float(d @ self.f), 100 * float(d @ self.r), 100 * float(d @ self.u)


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# --------------------------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------------------------
class Sheet:
    """Figure laid out in inches, origin bottom-left."""

    def __init__(self, w, h, dpi):
        self.w, self.h, self.dpi = w, h, dpi
        self.fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=FIG_BG)

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h], **kw)

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.w, y / self.h, s, **kw)

    def header(self, title, subtitle=None, top=0.35):
        self.text(0.5, self.h - top, title, fontsize=22, fontweight="bold", color=INK,
                  ha="left", va="top")
        if subtitle:
            self.text(0.5, self.h - top - 0.48, subtitle, fontsize=13, color=INK2,
                      ha="left", va="top")
        self.fig.add_artist(plt.Line2D([0.5 / self.w, 1 - 0.5 / self.w],
                                       [(self.h - top - 0.95) / self.h] * 2,
                                       color=GRID, lw=1.2))

    def caption(self, text, y=0.25, width_chars=None, fontsize=10.5):
        if width_chars is None:
            width_chars = int((self.w - 1.0) * 72 / (fontsize * 0.55))
        paras = [textwrap.fill(p, width_chars) for p in text.split("\n")]
        self.text(0.5, y, "\n".join(paras), fontsize=fontsize, color=INK2, ha="left",
                  va="bottom", linespacing=1.35)

    def save(self, out_dir, name):
        png = os.path.join(out_dir, name + ".png")
        pdf = os.path.join(out_dir, name + ".pdf")
        self.fig.savefig(png, dpi=self.dpi, metadata={"Software": None})
        self.fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                        "Creator": None, "Producer": None})
        plt.close(self.fig)
        with Image.open(png) as im:
            size = im.size
        return png, pdf, size


def show_region(ax, img, box, out_w, out_h):
    """Show native-pixel box (x0, y0, x1, y1) of img resampled (Lanczos) to out_w x out_h
    display pixels.  The axes keep IMAGE coordinates (pixel i spans [i, i+1)), so anything
    plotted afterwards in image pixel coordinates lands exactly where it belongs."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, ("crop outside image", box)
    crop = Image.fromarray(img[y0:y1, x0:x1])
    if (out_w, out_h) != crop.size:
        crop = crop.resize((int(out_w), int(out_h)), Image.LANCZOS)
    ax.imshow(np.asarray(crop), extent=(x0, x1, y1, y0), interpolation="nearest",
              resample=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])


def frame_axes(ax, color=INK2, lw=0.8):
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(lw)


def fmt_signed(x, nd=3):
    s = "%+.*f" % (nd, x)
    return s.replace("-", "−")


def sgn(x, nd=3):
    if abs(x) < 0.5 * 10 ** (-nd):
        return "0" if nd == 0 else fmt_signed(0.0, nd)
    return fmt_signed(x, nd)


# --------------------------------------------------------------------------------------------
# Board 1: sequence board
# --------------------------------------------------------------------------------------------
def board_sequence(pkg, out):
    MAG = 3.0                       # inset magnification relative to the face thumbnail
    dpi = 150
    FW, FH = 21.0, 25.4
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — clockwise gaze sequence (10 frames)",
             "Official renders renders/frame_00..09.png, placed at θk clockwise from 12 o'clock; "
             "each tile shows the full frame and a magnified eye region.")

    # eye region (same for every frame: head and camera are locked) from metadata eye centres
    sp = pkg.sclera_px
    cx = 0.5 * (pkg.e2L[0] + pkg.e2R[0])
    cy = 0.5 * (pkg.e2L[1] + pkg.e2R[1])
    rw = int(2 * round((abs(pkg.e2L[0] - pkg.e2R[0]) + 2 * 1.45 * sp) / 2))
    rh = int(2 * round(2.05 * sp / 2))
    x0 = int(round(cx - rw / 2)); y0 = int(round(cy - rh / 2))
    box = (x0, y0, x0 + rw, y0 + rh)

    W = 2.5                                      # face thumbnail (inches)
    Wi = MAG * W * rw / pkg.W                    # inset width so that magnification == MAG
    Hi = Wi * rh / rw
    lab_h, gap, cap_h = 0.55, 0.38, 0.30
    tile_w = max(W, Wi)
    tile_h = lab_h + W + gap + Hi + cap_h
    R = 8.85
    Cx, Cy = FW / 2, 1.55 + 0.2 + (FH - 1.55 - 1.45) / 2 - 0.15
    thumb_px = int(round(W * dpi))
    inset_wpx, inset_hpx = int(round(Wi * dpi)), int(round(Hi * dpi))

    for rec in pkg.meta:
        k = rec["frame"]
        th = rec["theta"]
        tx, ty = Cx + R * math.sin(th), Cy + R * math.cos(th)   # clockwise from 12 o'clock
        left, bottom = tx - tile_w / 2, ty - tile_h / 2
        img = pkg.frame_image(rec)
        # label
        hh, vv = rec["input"]["h"], rec["input"]["v"]
        S.text(tx, bottom + tile_h, "k = %d    θ = %.3f rad" % (k, th), fontsize=14,
               fontweight="bold", ha="center", va="top", color=INK)
        S.text(tx, bottom + tile_h - 0.27, "h = %s    v = %s" % (sgn(hh), sgn(vv)),
               fontsize=13, ha="center", va="top", color=INK)
        # face thumbnail
        fy = bottom + cap_h + Hi + gap
        axf = S.axes(tx - W / 2, fy, W, W)
        show_region(axf, img, (0, 0, pkg.W, pkg.H), thumb_px, thumb_px)
        frame_axes(axf, INK2, 0.6)
        axf.add_patch(patches.Rectangle((box[0], box[1]), rw, rh, fill=False, ec=ACC, lw=1.3))
        # inset
        axi = S.axes(tx - Wi / 2, bottom + cap_h, Wi, Hi)
        show_region(axi, img, box, inset_wpx, inset_hpx)
        frame_axes(axi, ACC, 1.6)
        for key in ("pupil_center_2D_L", "pupil_center_2D_R"):
            px, py = rec[key]
            axi.plot([px], [py], marker="+", ms=9, mew=1.4, color=GREEN, path_effects=HALO_THIN)
        for (xa, xb) in ((box[0], box[0]), (box[2], box[2])):
            con = ConnectionPatch(xyA=(xa, box[3]), coordsA="data", axesA=axf,
                                  xyB=(xb, box[1]), coordsB="data", axesB=axi,
                                  color=ACC, lw=0.9)
            S.fig.add_artist(con)
        S.text(tx, bottom + cap_h - 0.05, "eye region ×%g" % MAG, fontsize=11.5, color=ACC,
               ha="center", va="top")

    # centre: clockwise arrow + equations
    rc = 2.35
    axc = S.axes(Cx - rc - 0.4, Cy - rc - 0.4, 2 * rc + 0.8, 2 * rc + 0.8)
    axc.set_xlim(-rc - 0.4, rc + 0.4); axc.set_ylim(-rc - 0.4, rc + 0.4)
    axc.set_aspect("equal"); axc.axis("off")
    phis = np.radians(np.linspace(8, 352, 300))
    axc.plot(rc * np.sin(phis), rc * np.cos(phis), color=ACC, lw=1.6)
    for phi_deg in (110, 230, 352):
        p1, p0 = math.radians(phi_deg), math.radians(phi_deg - 6)
        axc.annotate("", xy=(rc * math.sin(p1), rc * math.cos(p1)),
                     xytext=(rc * math.sin(p0), rc * math.cos(p0)),
                     arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.6, mutation_scale=22,
                                     shrinkA=0, shrinkB=0))
    for rec in pkg.meta:
        th = rec["theta"]
        sx, sy = math.sin(th), math.cos(th)
        axc.plot([0.93 * rc * sx, 1.07 * rc * sx], [0.93 * rc * sy, 1.07 * rc * sy], color=ACC,
                 lw=1.2)
        axc.text(0.80 * rc * sx, 0.80 * rc * sy, str(rec["frame"]), ha="center", va="center",
                 fontsize=12, color=ACC)
    axc.text(0, 0.62, "clockwise", ha="center", va="center", fontsize=15, fontweight="bold",
             color=INK)
    axc.text(0, 0.05, r"$\theta_k = 2\pi k\,/\,10,\quad k = 0\ldots9$", ha="center",
             va="center", fontsize=15)
    axc.text(0, -0.50, r"$h = \sin\theta_k \qquad v = \cos\theta_k$", ha="center",
             va="center", fontsize=15)
    axc.text(0, -1.0, "k = 0 at 12 o'clock", ha="center", va="center", fontsize=12,
             color=INK2)

    # consistency check of metadata against the sampling equations
    n = len(pkg.meta)
    err = max(max(abs(r["theta"] - 2 * math.pi * r["frame"] / n),
                  abs(r["input"]["h"] - math.sin(r["theta"])),
                  abs(r["input"]["v"] - math.cos(r["theta"]))) for r in pkg.meta)

    # legend (bottom-left)
    lx, ly = 0.6, 2.5
    S.text(lx + 0.5, ly, "Legend", fontsize=12, fontweight="bold", va="center")
    axl = S.axes(lx + 0.45, ly - 0.95, 0.45, 0.7)
    axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    axl.add_patch(patches.Rectangle((0.1, 0.62), 0.8, 0.3, fill=False, ec=ACC, lw=1.4))
    axl.plot([0.5], [0.2], marker="+", ms=11, mew=1.6, color=GREEN)
    S.text(lx + 1.0, ly - 0.42, "eye region magnified in the inset below each face",
           fontsize=11, va="center")
    S.text(lx + 1.0, ly - 0.81, "pupil_center_2D_L / _R from metadata (projected ground truth)",
           fontsize=11, va="center")

    S.caption(
        "Data: renders/frame_00.png…frame_09.png (unmodified pixels; thumbnails and insets are "
        "Lanczos-resampled for display only) and metadata.json. Tile labels k, θ, h, v are read "
        "from metadata (max deviation from θk = 2πk/10, h = sinθk, v = cosθk: %.1e). "
        "The eye region is the same for every frame (head and camera locked): x %d–%d px, "
        "y %d–%d px of the %d×%d image, centred on the metadata eye centres "
        "(extra.eye_center_2D_L/R); ×%g is relative to the face thumbnail (inset scale "
        "%.2f output px per image px at the PNG resolution). Pupil crosses are metadata "
        "pupil_center_2D_L/R plotted in image coordinates (pixel i spans [i, i+1)) through the same "
        "crop/scale transform as the pixels; nothing is repositioned. 'L' = subject's left eye, "
        "image right."
        % (err, box[0], box[2], box[1], box[3], pkg.W, pkg.H, MAG, inset_wpx / rw),
        y=0.3, fontsize=11)
    return S.save(out, "sequence_board")


# --------------------------------------------------------------------------------------------
# Board 2: direction board
# --------------------------------------------------------------------------------------------
def board_direction(pkg, out):
    dv = pkg.dirval
    if dv is None:
        print("  [direction_board] reports/direction_validation.json missing - skipped")
        return None
    MAG = 10
    HS = 24                                       # half-size of the crop window (native px)
    dpi = 200
    cpx = 2 * HS * MAG                            # crop display size (px)
    cin = cpx / dpi                               # crop display size (in)
    FW = 13.6
    row_gap = 0.32
    rows = ["NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN", "RESET"]
    top_h, bot_h = 2.35, 2.25
    FH = top_h + len(rows) * (cin + row_gap) + bot_h
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — five-direction validation",
             "Neutral reference vs. measured pupil centre per direction, both eyes, drawn at true "
             "scale (crops ×%d)." % MAG)
    tests = {t["test"]: t for t in dv["tests"]}
    neu = dv["neutral_pupils"]
    # crop windows centred on the neutral projected pupils (identical in every row)
    win = {}
    for e in ("R", "L"):
        nx, ny = neu[e]["projected"]
        x0, y0 = int(round(nx)) - HS, int(round(ny)) - HS
        win[e] = (x0, y0, x0 + 2 * HS, y0 + 2 * HS)

    xlab, xR = 0.5, 2.35
    xL = xR + cin + 0.25
    xT = xL + cin + 0.45
    yhead = FH - top_h + 0.2
    S.text(xR + cin / 2, yhead + 0.42, "image LEFT side", fontsize=12, color=INK2,
           ha="center", va="bottom")
    S.text(xR + cin / 2, yhead + 0.08, "subject's RIGHT eye (R)", fontsize=13.5,
           fontweight="bold", ha="center", va="bottom")
    S.text(xL + cin / 2, yhead + 0.42, "image RIGHT side", fontsize=12, color=INK2,
           ha="center", va="bottom")
    S.text(xL + cin / 2, yhead + 0.08, "subject's LEFT eye (L)", fontsize=13.5,
           fontweight="bold", ha="center", va="bottom")
    S.text(xT, yhead + 0.08, "displacement vs. neutral (px, 1024² image)", fontsize=13.5,
           fontweight="bold", ha="left", va="bottom")

    key = {"L": "left_eye", "R": "right_eye"}
    expect = {"LEFT": "pupils → image left", "RIGHT": "pupils → image right",
              "UP": "pupils → image up", "DOWN": "pupils → image down",
              "NEUTRAL": "reference pose", "RESET": "back to neutral"}
    n_pass = 0
    for i, name in enumerate(rows):
        yb = FH - top_h - (i + 1) * (cin + row_gap) + row_gap / 2
        t = tests.get(name)
        imgname = name.lower()
        img = pkg.dir_image(imgname)
        # row label
        S.text(xlab, yb + cin - 0.05, name, fontsize=17, fontweight="bold", va="top")
        if name == "NEUTRAL":
            inp = "h = 0, v = 0"
        elif name == "RESET":
            inp = "reset() ×3 after\nmove(0.6, −0.8)"
        else:
            inp = "h = %s, v = %s" % (sgn(t["input"]["h"], 0), sgn(t["input"]["v"], 0))
        S.text(xlab, yb + cin - 0.48, inp, fontsize=12, va="top", color=INK, linespacing=1.3)
        S.text(xlab, yb + cin - (1.15 if name == "RESET" else 0.85), expect[name],
               fontsize=11.5, va="top", color=INK2)
        S.text(xlab, yb + 0.05, "reports/direction_test/\n%s.png" % imgname, fontsize=9,
               va="bottom", color=INK2, family=MONO)
        for e, xe in (("R", xR), ("L", xL)):
            ax = S.axes(xe, yb, cin, cin)
            if img is None:
                ax.text(0.5, 0.5, "image missing", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            show_region(ax, img, win[e], cpx, cpx)
            frame_axes(ax, INK2, 0.8)
            ref = np.array(neu[e]["projected"], float)
            ref_det = np.array(neu[e]["detected"], float)
            ax.plot(*ref, marker="o", ms=10, mfc="none", mec=RED, mew=1.8, path_effects=HALO_THIN)
            if name in ("LEFT", "RIGHT", "UP", "DOWN"):
                pj, dt = t[key[e]]["projected"], t[key[e]]["detected"]
                mp = ref + np.array([pj["dx"], pj["dy"]])
                md = ref_det + np.array([dt["dx"], dt["dy"]])
                ax.annotate("", xy=tuple(mp), xytext=tuple(ref),
                            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0,
                                            mutation_scale=13, shrinkA=0, shrinkB=0,
                                            path_effects=HALO))
                ax.plot(*md, marker="x", ms=7, mew=1.6, color=GREEN, path_effects=HALO_THIN)
                ax.plot(*mp, marker="+", ms=12, mew=1.6, color=GREEN, path_effects=HALO_THIN)
            else:
                ax.plot(*ref_det, marker="x", ms=7, mew=1.6, color=GREEN, path_effects=HALO_THIN)
            if i == 0:
                # 10 px scale bar
                bx, by = win[e][0] + 3, win[e][3] - 4
                ax.plot([bx, bx + 10], [by, by], color="white", lw=4.5, solid_capstyle="butt")
                ax.plot([bx, bx + 10], [by, by], color=INK, lw=2.5, solid_capstyle="butt")
                ax.text(bx + 5, by - 1.5, "10 px", ha="center", va="bottom", fontsize=10,
                        color=INK, path_effects=HALO_THIN)
        # numbers
        mono = dict(family=MONO, fontsize=11.5, va="top", ha="left")
        yt = yb + cin - 0.05
        if name in ("LEFT", "RIGHT", "UP", "DOWN"):
            lines = ["              R (img-left)     L (img-right)",
                     "              dx      dy       dx      dy"]
            for kind in ("projected", "detected"):
                vals = []
                for e in ("R", "L"):
                    d = t[key[e]][kind]
                    vals += [sgn(d["dx"]), sgn(d["dy"])]
                lines.append("%-10s %7s %7s  %7s %7s" % (kind, *vals))
            lines.append("")
            lines.append("along/cross (projected):")
            lines.append("  R %.3f / %.3f    L %.3f / %.3f" % (
                t["right_eye"]["projected"]["along_px"], t["right_eye"]["projected"]["cross_px"],
                t["left_eye"]["projected"]["along_px"], t["left_eye"]["projected"]["cross_px"]))
            ok = t["status"] == "PASS"
        elif name == "NEUTRAL":
            lines = ["absolute pupil centre (x, y):",
                     "  R projected %8.2f %8.2f" % tuple(neu["R"]["projected"]),
                     "  R detected  %8.2f %8.2f" % tuple(neu["R"]["detected"]),
                     "  L projected %8.2f %8.2f" % tuple(neu["L"]["projected"]),
                     "  L detected  %8.2f %8.2f" % tuple(neu["L"]["detected"]),
                     "",
                     "reference for all rows below"]
            ok = None
        else:  # RESET
            lr, ll = t["right_eye"], t["left_eye"]
            idiff = t.get("image_diff_vs_neutral_8bit", {})
            lines = ["shift vs neutral (px):",
                     "  R projected %.3f  detected %.3f" % (lr["projected_shift_px"],
                                                            lr["detected_shift_px"]),
                     "  L projected %.3f  detected %.3f" % (ll["projected_shift_px"],
                                                            ll["detected_shift_px"]),
                     "eye quaternion max diff: %g" % t.get(
                         "pose_quaternion_max_diff_over_3_resets", float("nan")),
                     "image diff vs neutral (8-bit):",
                     "  max %g, mean %g" % (idiff.get("max", float("nan")),
                                            idiff.get("mean", float("nan")))]
            ok = t["status"] == "PASS"
        S.text(xT, yt, "\n".join(lines), linespacing=1.35, **mono)
        if ok is not None:
            n_pass += int(ok)
            S.text(FW - 0.5, yb + 0.08, "PASS" if ok else "FAIL", fontsize=19,
                   fontweight="bold", color=GREEN if ok else RED, ha="right", va="bottom")
        if i < len(rows) - 1:
            yl = yb - row_gap / 2
            S.fig.add_artist(plt.Line2D([0.5 / FW, 1 - 0.5 / FW], [yl / FH] * 2, color=GRID,
                                        lw=1.0))

    # legend
    ly = bot_h - 0.12
    axl = S.axes(0.5, ly - 0.42, 12.6, 0.42)
    axl.axis("off"); axl.set_xlim(0, 12.6); axl.set_ylim(0, 0.42)
    y = 0.21
    axl.plot(0.15, y, marker="o", ms=10, mfc="none", mec=RED, mew=1.8)
    axl.text(0.4, y, "neutral reference (projected)", va="center", fontsize=11.5)
    axl.plot(3.45, y, marker="+", ms=11, mew=1.8, color=GREEN)
    axl.text(3.65, y, "measured: projected GT", va="center", fontsize=11.5)
    axl.plot(6.25, y, marker="x", ms=7, mew=1.8, color=GREEN)
    axl.text(6.45, y, "measured: detected in PNG", va="center", fontsize=11.5)
    axl.annotate("", xy=(9.6, y), xytext=(9.1, y),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0, mutation_scale=13,
                                 shrinkA=0, shrinkB=0))
    axl.text(9.75, y, "reference → projected (true scale)", va="center", fontsize=11.5)

    crit = dv.get("criteria", {})
    S.caption(
        "Data: reports/direction_validation.json (%d/%d PASS; criteria: shift ≥ %s px along the "
        "expected direction, cross/along ≤ %s; %s) and the renders reports/direction_test/"
        "{neutral,left,right,up,down,reset}.png. Crops: %d×%d native px centred on each eye's "
        "neutral projected pupil (same window in every row), Lanczos ×%d, i.e. 1 image px = %d "
        "output px at the PNG resolution; arrows are the projected displacement at that same "
        "scale (no exaggeration). Projected positions = neutral projected + projected (dx, dy); "
        "detected positions = neutral detected + detected (dx, dy), both recomputed from the JSON. "
        "Neutral/RESET rows show the neutral detected centre as ×. Image y points down, so "
        "UP gives dy < 0."
        % (dv.get("passed", n_pass), dv.get("total", 5), crit.get("min_shift_px", "?"),
           crit.get("max_cross_ratio", "?"), crit.get("measurements", ""), 2 * HS, 2 * HS, MAG,
           MAG),
        y=0.25, fontsize=11)
    return S.save(out, "direction_board")


# --------------------------------------------------------------------------------------------
# Board 3: eye detail (+ before/after)
# --------------------------------------------------------------------------------------------
EYE_CONDS = [("neutral", "NEUTRAL", "h = 0, v = 0"), ("left", "LEFT", "h = −1, v = 0"),
             ("right", "RIGHT", "h = +1, v = 0"), ("up", "UP", "h = 0, v = +1"),
             ("down", "DOWN", "h = 0, v = −1")]


def eye_windows(pkg, half_w=72, half_h=48):
    win = {}
    for e, c in (("R", pkg.e2R), ("L", pkg.e2L)):
        x0, y0 = int(round(c[0])) - half_w, int(round(c[1])) - half_h
        win[e] = (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h)
    return win


def cond_image(pkg, cond):
    img = pkg.dir_image(cond)
    src = "reports/direction_test/%s.png" % cond
    return img, src


def ext_scale_bar(S, x, y, mag, pkg, n=10):
    """Scale bar drawn OUTSIDE the image, at exactly the crop's display scale."""
    w_in = n * mag / S.dpi
    ax = S.axes(x, y, w_in, 0.12)
    ax.axis("off")
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.fill_between([0, n], 0.2, 0.8, color=INK, lw=0)
    mm = n * pkg.Z_eye / pkg.fx * 1000
    S.text(x + w_in + 0.08, y + 0.06, "%d px \u2248 %.2f mm" % (n, mm), fontsize=10.5,
           va="center", ha="left", color=INK)


def board_eye_detail(pkg, out):
    MAG = 4
    dpi = 200
    win = eye_windows(pkg)
    cw = (win["R"][2] - win["R"][0]) * MAG
    ch = (win["R"][3] - win["R"][1]) * MAG
    wi, hi = cw / dpi, ch / dpi
    lab_w, gap, rgap = 2.1, 0.22, 0.30
    top_h, bot_h = 2.25, 1.55
    FW = 0.5 + lab_w + 2 * wi + gap + 0.5
    FH = top_h + len(EYE_CONDS) * (hi + rgap) + bot_h
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — binocular eye detail",
             "Native eye crops, Lanczos ×%d, neutral and four extremes" % MAG)
    xR = 0.5 + lab_w
    xL = xR + wi + gap
    yh = FH - top_h + 0.2
    for xe, a, b in ((xR, "subject's RIGHT eye (R)", "image left"),
                     (xL, "subject's LEFT eye (L)", "image right")):
        S.text(xe + wi / 2, yh + 0.36, b, fontsize=11.5, color=INK2, ha="center", va="bottom")
        S.text(xe + wi / 2, yh + 0.04, a, fontsize=13, fontweight="bold", ha="center",
               va="bottom")
    for i, (cond, name, inp) in enumerate(EYE_CONDS):
        yb = FH - top_h - (i + 1) * (hi + rgap) + rgap / 2
        img, src = cond_image(pkg, cond)
        S.text(0.5, yb + hi - 0.05, name, fontsize=16, fontweight="bold", va="top")
        S.text(0.5, yb + hi - 0.45, inp, fontsize=12, va="top")
        S.text(0.5, yb + 0.05, src.replace("direction_test/", "direction_test/\n"), fontsize=8.5,
               family=MONO, color=INK2, va="bottom")
        for e, xe in (("R", xR), ("L", xL)):
            ax = S.axes(xe, yb, wi, hi)
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            show_region(ax, img, win[e], cw, ch)
            frame_axes(ax, INK2, 0.8)
        if i == 0:
            ext_scale_bar(S, 0.5, yb + 0.62, MAG, pkg)
    mm10 = 10 * pkg.Z_eye / pkg.fx * 1000
    S.caption(
        "Data: reports/direction_test/{neutral,left,right,up,down}.png (h, v = ±1 are the "
        "extreme inputs). Windows %d×%d px centred on the metadata eye centres "
        "extra.eye_center_2D (R: x %d–%d, L: x %d–%d; y %d–%d), identical in every "
        "row; Lanczos ×%d (1 image px = %d output px at PNG resolution). Scale bar: 10 px "
        "≈ %.2f mm at the eye-centre depth Z = %.1f cm (mm = px·Z/fx, fx = %.2f px from "
        "metadata K), drawn beside the crops at the same scale. No overlays on the pixels."
        % (win["R"][2] - win["R"][0], win["R"][3] - win["R"][1], win["R"][0], win["R"][2],
           win["L"][0], win["L"][2], win["R"][1], win["R"][3], MAG, MAG, mm10,
           pkg.Z_eye * 100, pkg.fx),
        y=0.25, fontsize=10.5)
    return S.save(out, "eye_detail")


def board_eye_before_after(pkg, before, out):
    MAG = 4
    dpi = 200
    win = eye_windows(pkg)                    # same windows (current package) for both versions
    cw = (win["R"][2] - win["R"][0]) * MAG
    ch = (win["R"][3] - win["R"][1]) * MAG
    wi, hi = cw / dpi, ch / dpi
    lab_w, gap, sub_gap, grp_gap = 2.1, 0.22, 0.10, 0.42
    stat_w = 2.9
    top_h, bot_h = 2.25, 1.55
    FW = 0.5 + lab_w + 2 * wi + gap + 0.3 + stat_w + 0.4
    FH = top_h + len(EYE_CONDS) * (2 * hi + sub_gap + grp_gap) + bot_h
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — eye detail, before vs. after",
             "Per condition: top = BEFORE (older package), bottom = AFTER (this package); "
             "Lanczos ×%d, identical windows" % MAG)
    xR = 0.5 + lab_w
    xL = xR + wi + gap
    xS = xL + wi + 0.3
    yh = FH - top_h + 0.2
    for xe, a, b in ((xR, "subject's RIGHT eye (R)", "image left"),
                     (xL, "subject's LEFT eye (L)", "image right")):
        S.text(xe + wi / 2, yh + 0.36, b, fontsize=11.5, color=INK2, ha="center", va="bottom")
        S.text(xe + wi / 2, yh + 0.04, a, fontsize=13, fontweight="bold", ha="center",
               va="bottom")
    S.text(xS, yh + 0.04, "|after − before|\n(8-bit RGB, crop)", fontsize=12,
           fontweight="bold", ha="left", va="bottom")
    y = FH - top_h
    for i, (cond, name, inp) in enumerate(EYE_CONDS):
        imgs = {"BEFORE": cond_image(before, cond)[0], "AFTER": cond_image(pkg, cond)[0]}
        y_grp_top = y
        for j, tag in enumerate(("BEFORE", "AFTER")):
            yb = y - hi
            img = imgs[tag]
            S.text(xR - 0.12, yb + hi / 2, tag, fontsize=11, fontweight="bold",
                   color=INK2 if tag == "BEFORE" else ACC, ha="right", va="center")
            for e, xe in (("R", xR), ("L", xL)):
                ax = S.axes(xe, yb, wi, hi)
                if img is None:
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.5, 0.5, "image missing", ha="center", va="center",
                            transform=ax.transAxes, color=INK2)
                    continue
                show_region(ax, img, win[e], cw, ch)
                frame_axes(ax, ACC if tag == "AFTER" else INK2, 1.2 if tag == "AFTER" else 0.8)
            if i == 0 and j == 1:
                ext_scale_bar(S, 0.5, yb + 0.15, MAG, pkg)
            y = yb - (sub_gap if j == 0 else 0)
        S.text(0.5, y_grp_top - 0.05, name, fontsize=16, fontweight="bold", va="top")
        S.text(0.5, y_grp_top - 0.45, inp, fontsize=12, va="top")
        lines = []
        for e in ("R", "L"):
            if imgs["BEFORE"] is None or imgs["AFTER"] is None:
                lines.append("%s: n/a" % e)
                continue
            x0, y0, x1, y1 = win[e]
            d = np.abs(imgs["AFTER"][y0:y1, x0:x1].astype(int) -
                       imgs["BEFORE"][y0:y1, x0:x1].astype(int))
            lines.append("%s: max %3d  mean %.2f" % (e, d.max(), d.mean()))
        S.text(xS, y_grp_top - 0.1, "\n".join(lines), family=MONO, fontsize=11, va="top",
               linespacing=1.5)
        y -= grp_gap
        if i < len(EYE_CONDS) - 1:
            yl = y + grp_gap / 2
            S.fig.add_artist(plt.Line2D([0.5 / FW, 1 - 0.5 / FW], [yl / FH] * 2, color=GRID,
                                        lw=1.0))
    S.caption(
        "BEFORE: %s\nAFTER: %s\nImages: reports/direction_test/{neutral,left,right,up,down}.png "
        "of each package. Crop windows are taken from the AFTER package's metadata eye centres "
        "(extra.eye_center_2D) and applied unchanged to both versions, so pixels are directly "
        "comparable. Difference statistics are computed on the native 8-bit crops (no "
        "resampling). Scale bar: 10 px ≈ %.2f mm at the eye-centre depth."
        % (before.root, pkg.root, 10 * pkg.Z_eye / pkg.fx * 1000), y=0.25, fontsize=10.5)
    return S.save(out, "eye_detail_before_after")


# --------------------------------------------------------------------------------------------
# Geometry drawing helpers
# --------------------------------------------------------------------------------------------
def dim_line(ax, p1, p2, off, text, fs=11, color=INK, text_off=0.0, ext=True, rot=None):
    """Engineering dimension between p1 and p2 (data coords), offset by vector off."""
    p1, p2, off = np.asarray(p1, float), np.asarray(p2, float), np.asarray(off, float)
    a, b = p1 + off, p2 + off
    if ext:
        n = off / (np.linalg.norm(off) + 1e-12)
        for p, q in ((p1, a), (p2, b)):
            ax.plot([p[0] + 0.15 * n[0], q[0] + 0.25 * n[0]],
                    [p[1] + 0.15 * n[1], q[1] + 0.25 * n[1]], color=INK2, lw=0.6)
    ax.annotate("", xy=tuple(b), xytext=tuple(a),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.9, mutation_scale=10,
                                shrinkA=0, shrinkB=0))
    m = (a + b) / 2
    d = b - a
    if rot is None:
        rot = math.degrees(math.atan2(d[1], d[0]))
        if rot > 90: rot -= 180
        if rot < -90: rot += 180
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    if np.dot(nrm, off) < 0:
        nrm = -nrm
    pos = m + nrm * text_off
    ax.text(pos[0], pos[1], text, ha="center", va="center", fontsize=fs, color=color,
            rotation=rot, rotation_mode="anchor",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def hdim(ax, x1, x2, y, y1_from, y2_from, text, fs=11, color=INK):
    """Horizontal dimension between x1 and x2 at height y; extension lines from y*_from."""
    for x, yf in ((x1, y1_from), (x2, y2_from)):
        ax.plot([x, x], [yf, y + 0.35 * np.sign(y - yf)], color=INK2, lw=0.6, zorder=6)
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.9, mutation_scale=10,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.text((x1 + x2) / 2, y, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=7)


def setup_view(ax, xlabel, ylabel):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def camera_glyph(ax, s, y, size=1.0):
    body = patches.Rectangle((s + 0.35 * size, y - 0.9 * size), 2.6 * size, 1.8 * size,
                             fc="#e6e9ee", ec=INK, lw=0.9, zorder=5)
    lens = patches.Polygon([[s, y], [s + 0.35 * size, y - 0.55 * size],
                            [s + 0.35 * size, y + 0.55 * size]], closed=True, fc="#c9ced6",
                           ec=INK, lw=0.9, zorder=5)
    ax.add_patch(body); ax.add_patch(lens)
    ax.plot([s], [y], marker="o", ms=4, color=INK, zorder=6)


def angle_arc(ax, center, r, a0, a1, text, color, fs=10.5, tr=None, ha="left"):
    a0, a1 = sorted((a0, a1))
    t = np.radians(np.linspace(a0, a1, 50))
    ax.plot(center[0] + r * np.cos(t), center[1] + r * np.sin(t), color=color, lw=1.0)
    am = math.radians((a0 + a1) / 2)
    tr = tr or r * 1.12
    ax.text(center[0] + tr * math.cos(am), center[1] + tr * math.sin(am), text, fontsize=fs,
            color=color, ha=ha, va="center", path_effects=HALO_THIN)


# --------------------------------------------------------------------------------------------
# Board 4: camera calibration
# --------------------------------------------------------------------------------------------
def board_camera(pkg, out):
    dpi = 180
    FW, FH = 18.0, 17.8
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — camera & eye-rig calibration (to scale)",
             "Top view (world X–Y) and side view (world Y–Z), centimetres, relative to "
             "the eye midpoint M; all values computed from metadata.json")
    sL, xL_, zL = pkg.coords(pkg.CL)
    sR, xR_, zR = pkg.coords(pkg.CR)
    sF, xF, zF = pkg.coords(pkg.F0)
    sC, xC, zC = pkg.coords(pkg.cam)
    hfov = math.degrees(math.atan(pkg.cx / pkg.fx))
    vfov = math.degrees(math.atan(pkg.cy / pkg.fy))
    bh, bv = 100 * pkg.bound_h, 100 * pkg.bound_v
    rs = pkg.sclera_mm / 10.0
    ipd = 100 * float(np.linalg.norm(pkg.CL - pkg.CR))
    smin, smax = -6.5, 46.0
    lim = 13.6
    zs, zx = (-2.4, 3.2), (-4.6, 4.6)          # detail A window (top view)

    def draw_view(ax, side):
        if side == "top":
            eyes = [(sL, xL_), (sR, xR_)]
            half, cam_y, F_y, b = hfov, xC, xF, bh
        else:
            eyes = [(sL, zL)]
            half, cam_y, F_y, b = vfov, zC, zF, bv
        ax.set_xlim(smin, smax); ax.set_ylim(-lim, lim)
        ta = math.tan(math.radians(half))
        s_end = smin
        wedge = [[sC, cam_y], [s_end, cam_y + (sC - s_end) * ta],
                 [s_end, cam_y - (sC - s_end) * ta]]
        ax.add_patch(patches.Polygon(wedge, closed=True, fc=ACC_FILL, ec="none", zorder=0))
        for sg in (1, -1):
            ax.plot([sC, s_end], [cam_y, cam_y + sg * (sC - s_end) * ta], color=ACC_MID,
                    lw=1.0, ls="--", zorder=1)
        ax.plot([smin, sC], [cam_y, cam_y], color=ACC, lw=0.9, ls=(0, (8, 3, 2, 3)), zorder=2)
        for (se, ye) in eyes:
            ax.add_patch(patches.Circle((se, ye), rs, fc="white", ec=INK, lw=1.0, zorder=3))
            ax.plot([se], [ye], marker="o", ms=3, color=INK, zorder=4)
        ax.plot([0], [0], marker="+", ms=12, mew=1.6, color=RED, zorder=6)
        ax.plot([sF, sF], [F_y - b, F_y + b], color=ACC, lw=3.2, solid_capstyle="butt", zorder=4)
        ax.plot([sF], [F_y], marker="o", ms=6, mfc="white", mec=RED, mew=1.8, zorder=6)
        camera_glyph(ax, sC, cam_y, 1.0)
        # labels
        ax.text(sF - 0.6, F_y - b, "target bound ±%.1f cm" % b, fontsize=10.5,
                color=ACC, ha="right", va="bottom")
        ax.text(sF - 0.7, F_y - 0.9, "F$_0$", fontsize=13, color=RED, ha="right", va="top",
                path_effects=HALO_THIN)
        ax.text(sC + 0.2, cam_y + 1.4, "camera C", fontsize=11.5, ha="left", va="bottom")
        if side == "top":
            ax.text(0.9, -0.5, "M", fontsize=13, color=RED, ha="left", va="top",
                    path_effects=HALO_THIN)
            ax.text(sL - rs - 0.2, xL_, "C$_L$", fontsize=11, ha="right", va="center")
            ax.text(sR - rs - 0.2, xR_, "C$_R$", fontsize=11, ha="right", va="center")
        else:
            ax.text(sL + 0.4, zL - rs - 0.3, "M, C$_L$, C$_R$ (coincide in this view)",
                    fontsize=10.5, color=INK, ha="left", va="top", path_effects=HALO_THIN)
        s_lab = 22.0
        y_lab = cam_y + (sC - s_lab) * ta
        ax.text(s_lab, y_lab + 0.45, "FOV edge, half-angle atan(c/f) = %.2f°" % half,
                fontsize=10, color=ACC_MID, rotation=-half, rotation_mode="anchor",
                ha="center", va="bottom")
        ax.text(14.0, 0.35, "optical axis", fontsize=10, color=ACC, ha="center", va="bottom")
        # dimensions (extension lines start clear of the eyes / target)
        top = lim - 1.0
        clear = (abs(xL_) + rs + 0.3) if side == "top" else rs + 0.3
        hdim(ax, 0, sC, top, clear, cam_y + 1.2, "M → camera C  %.2f cm" % sC)
        hdim(ax, 0, sF, top - 2.3, clear, F_y + b + 0.3, "M → F$_0$  %.2f cm" % sF)
        hdim(ax, sF, sC, -top, F_y - b - 0.3, cam_y - 1.2, "%.2f cm" % (sC - sF), fs=10.5)
        wh = sC * ta
        dim_line(ax, (0, -wh), (0, wh), (-4.3, 0), "%.1f cm FOV at M" % (2 * wh), fs=10.5)
        if side == "top":
            ax.add_patch(patches.Rectangle((zs[0], zx[0]), zs[1] - zs[0], zx[1] - zx[0],
                                           fill=False, ec=INK2, lw=0.8, ls=":", zorder=7))
            ax.text(zs[1] + 0.15, zx[0] + 0.1, "A", fontsize=11, fontweight="bold", color=INK2,
                    va="bottom", ha="left")

    lw_ = 11.0
    hview = lw_ * (2 * lim) / (smax - smin)
    yT = FH - 2.1 - hview
    axT = S.axes(1.2, yT, lw_, hview)
    draw_view(axT, "top")
    setup_view(axT, "forward distance from M along f = −Y (cm)",
               "X − X$_M$ (cm)  [+X = subject's left]")
    axT.set_title("TOP VIEW (looking down −Z; +X up = subject's left = image right)",
                  fontsize=12.5, loc="left", color=INK, fontweight="bold")
    yS = yT - 1.45 - hview
    axS = S.axes(1.2, yS, lw_, hview)
    draw_view(axS, "side")
    setup_view(axS, "forward distance from M along f = −Y (cm)", "Z − Z$_M$ (cm)  [up]")
    axS.set_title("SIDE VIEW (from the subject's left, looking along −X)", fontsize=12.5,
                  loc="left", color=INK, fontweight="bold")

    # detail A: zoom on eyes (top view)
    xa = 1.2 + lw_ + 1.25
    wa = 3.4
    ha_ = wa * (zx[1] - zx[0]) / (zs[1] - zs[0])
    yA = FH - 2.1 - ha_
    axA = S.axes(xa, yA, wa, ha_)
    axA.set_xlim(*zs); axA.set_ylim(*zx)
    for (se, xe, lab, va) in ((sL, xL_, "C$_L$", "bottom"), (sR, xR_, "C$_R$", "top")):
        axA.add_patch(patches.Circle((se, xe), rs, fc="white", ec=INK, lw=1.1))
        axA.plot([se], [xe], marker="o", ms=4, color=INK)
        axA.text(se - 0.25, xe + (0.3 if va == "bottom" else -0.3), lab, fontsize=12,
                 ha="right", va=va)
        dim_line(axA, (se, xe), (se + rs, xe), (0, 0), "r %.2f" % rs, fs=10, ext=False,
                 text_off=0.0)
    axA.plot([zs[0], zs[1]], [0, 0], color=ACC, lw=0.9, ls=(0, (8, 3, 2, 3)))
    axA.plot([0], [0], marker="+", ms=14, mew=1.8, color=RED)
    axA.text(0.15, -0.25, "M", fontsize=13, color=RED, va="top", path_effects=HALO_THIN)
    dim_line(axA, (sR + rs * 0.3, xR_), (sL + rs * 0.3, xL_), (2.05 - rs * 0.3, 0),
             "IPD %.2f cm" % ipd, fs=11, text_off=0.0)
    setup_view(axA, "forward (cm)", "X − X$_M$ (cm)")
    zoom = (wa / (zs[1] - zs[0])) / (lw_ / (smax - smin))
    axA.set_title("Detail A: eyes, top view (×%.1f)" % zoom, fontsize=12, loc="left",
                  fontweight="bold")

    # table
    Mw = 100 * pkg.M
    Cw = 100 * pkg.cam
    mpix = pkg.project(pkg.M)
    look = -pkg.c2w[:3, 2] / np.linalg.norm(pkg.c2w[:3, 2])
    miss = 1000 * float(np.linalg.norm(np.cross(pkg.M - pkg.cam, look)))
    f_mm = pkg.fx * pkg.sensor_mm / pkg.W
    rows = [
        ("IPD |C$_L$ − C$_R$|", "%.2f mm" % (10 * ipd)),
        ("M (world)", "(%.2f, %.2f, %.2f) cm" % tuple(Mw)),
        ("camera C (world)", "(%.2f, %.2f, %.2f) cm" % tuple(Cw)),
        ("M → F$_0$ (fixation)", "%.2f cm" % sF),
        ("M → C (camera)", "%.2f cm" % sC),
        ("F$_0$ → C", "%.2f cm" % (sC - sF)),
        ("target bound h / v", "±%.2f / ±%.2f cm" % (bh, bv)),
        ("f$_x$, f$_y$", "%.2f, %.2f px" % (pkg.fx, pkg.fy)),
        ("c$_x$, c$_y$", "%.1f, %.1f px" % (pkg.cx, pkg.cy)),
        ("focal length f$_x$·sensor/W", "%.2f mm" % f_mm),
        ("   sensor / lens (%s)" % pkg.sensor_src, "%g / %g mm" % (pkg.sensor_mm, pkg.lens_mm)),
        ("FOV h × v, 2·atan(c/f)", "%.2f° × %.2f°" % (2 * hfov, 2 * vfov)),
        ("image size", "%d × %d px" % (pkg.W, pkg.H)),
        ("M projects to (P·M)", "(%.3f, %.3f) px" % tuple(mpix)),
        ("M off optical axis", "%.4f mm" % miss),
        ("eye-centre depth Z", "%.2f cm" % (100 * pkg.Z_eye)),
        ("1 px at eye depth", "%.3f mm" % (pkg.Z_eye / pkg.fx * 1000)),
        ("sclera radius (%s)" % pkg.sclera_src, "%.1f mm" % pkg.sclera_mm),
    ]
    ty = yA - 1.05
    S.text(xa, ty, "Calibration values", fontsize=13, fontweight="bold", va="top")
    ty -= 0.45
    for a, b in rows:
        S.text(xa, ty, a, fontsize=10.5, va="top", color=INK2 if a.startswith("   ") else INK)
        S.text(FW - 0.45, ty, b, fontsize=10.5, va="top", ha="right", family=MONO)
        ty -= 0.34

    # legend
    ly = yS - 0.95
    axl = S.axes(1.2, ly - 0.4, FW - 1.7, 0.4)
    axl.axis("off"); axl.set_xlim(0, FW - 1.7); axl.set_ylim(0, 0.4)
    yy = 0.2
    axl.add_patch(patches.Ellipse((0.2, yy), 0.26, 0.26 * 0.4 / 0.4 * (FW - 1.7) / (FW - 1.7),
                                  fc="white", ec=INK, lw=1.0))
    axl.text(0.45, yy, "eyeball (sclera r) & centre", va="center", fontsize=10.5)
    axl.plot(3.25, yy, marker="+", ms=12, mew=1.6, color=RED)
    axl.text(3.45, yy, "eye midpoint M", va="center", fontsize=10.5)
    axl.plot(5.3, yy, marker="o", ms=6, mfc="white", mec=RED, mew=1.8)
    axl.text(5.5, yy, "F$_0$ nominal fixation", va="center", fontsize=10.5)
    axl.plot([7.8, 8.3], [yy, yy], color=ACC, lw=3.2)
    axl.text(8.45, yy, "± target bound", va="center", fontsize=10.5)
    axl.plot([10.2, 10.7], [yy, yy], color=ACC_MID, lw=1.0, ls="--")
    axl.text(10.85, yy, "FOV edge (shaded: FOV)", va="center", fontsize=10.5)
    axl.plot([13.2, 13.7], [yy, yy], color=ACC, lw=0.9, ls=(0, (8, 3, 2, 3)))
    axl.text(13.85, yy, "optical axis", va="center", fontsize=10.5)

    S.caption(
        "Data: metadata.json frame 0 (extra.eye_center_3D_L/R, eye_midpoint_3D, "
        "camera_to_world_blender, camera_intrinsics_K, camera_extrinsics_Rt_world_to_camera, "
        "camera_matrix). Head axes: f (forward) = camera local +Z, r = camera +X, u = camera +Y "
        "(from camera_to_world_blender). Fixation distance and ± target bounds are derived "
        "from the 10 metadata fixation targets ((T−M)·f, (T−M)·r / h, (T−M)·u / v), not "
        "hard-coded. FOV half-angle = atan(c/f) from K. Sensor width, lens and sclera radius are "
        "not in metadata (source: %s). Both views share one scale (equal aspect); Detail A "
        "enlarges the dotted box A." % pkg.sensor_src, y=0.25, fontsize=10.5)
    return S.save(out, "camera_calibration")


# --------------------------------------------------------------------------------------------
# Board 5: gaze rays
# --------------------------------------------------------------------------------------------
def yaw_pitch(eye, T, pkg):
    g = (np.asarray(T, float) - eye)
    g = g / np.linalg.norm(g)
    yaw = math.degrees(math.atan2(g @ pkg.r, g @ pkg.f))
    pitch = math.degrees(math.atan2(g @ pkg.u, math.hypot(g @ pkg.r, g @ pkg.f)))
    return yaw, pitch


def board_gaze(pkg, out):
    dpi = 180
    FW, FH = 19.3, 14.5
    S = Sheet(FW, FH, dpi)
    S.header("GazeForge — fixation targets and gaze rays (to scale)",
             "Rays from each eye's rotation centre to the common fixation target; centimetres "
             "relative to the eye midpoint M")
    d = pkg.d_fix
    T_of = lambda h, v: pkg.M + d * pkg.f + h * pkg.bound_h * pkg.r + v * pkg.bound_v * pkg.u
    shades = {-1: (ACC_LIGHT, (0, (6, 3))), 0: (ACC_MID, "-"), 1: (ACC, (0, (8, 3, 2, 3)))}
    shades_lw = {-1: 1.6, 0: 1.4, 1: 1.6}
    sL, xL_, zL = pkg.coords(pkg.CL)
    sR, xR_, zR = pkg.coords(pkg.CR)
    rs = pkg.sclera_mm / 10
    seqc = [pkg.coords(T) for T in pkg.targets]
    smin, smax = -3.0, 39.0

    lw_ = 8.6
    ylo, yhi = -8.0, 13.0
    vh = lw_ * (yhi - ylo) / (smax - smin)
    y_top = FH - 2.0 - vh
    vgap = 1.25
    axT = S.axes(1.15, y_top, lw_, vh)
    axS = S.axes(1.15, y_top - vh - vgap, lw_, vh)
    yaw_txt, pit_txt = [], []
    for ax, view in ((axT, "top"), (axS, "side")):
        ax.set_xlim(smin, smax); ax.set_ylim(ylo, yhi)
        ax.axvline(100 * d, color=GRID, lw=1.0, zorder=0)
        ax.text(100 * d - 0.3, ylo + 0.3, "fixation plane", fontsize=10, color=INK2,
                va="bottom", ha="right")
        eyes = [(pkg.CL, sL, xL_ if view == "top" else zL),
                (pkg.CR, sR, xR_ if view == "top" else zR)]
        for val in (-1, 0, 1):
            T = T_of(val, 0) if view == "top" else T_of(0, val)
            sT, xT_, zT = pkg.coords(T)
            yT = xT_ if view == "top" else zT
            col, ls = shades[val]
            for (C, se, ye) in eyes:
                ax.plot([se, sT], [ye, yT], color=col, ls=ls, lw=shades_lw[val], zorder=3)
            ax.plot([sT], [yT], marker="s", ms=7, mfc="white", mec=col, mew=1.8, zorder=5)
            yl, pl = yaw_pitch(pkg.CL, T, pkg)
            yr, pr = yaw_pitch(pkg.CR, T, pkg)
            if view == "top":
                yaw_txt.append((val, yl, yr, col))
                ax.text(sT + 0.6, yT, "h = %s" % sgn(val, 0), fontsize=11, va="center",
                        color=INK)
            else:
                pit_txt.append((val, pl, pr, col))
                ax.text(sT + 0.6, yT, "v = %s" % sgn(val, 0), fontsize=11, va="center",
                        color=INK)
        # 10 sequence targets (projected onto this view)
        for (s_, x_, z_) in seqc:
            ax.plot([s_], [x_ if view == "top" else z_], marker="o", ms=4.5, color=GREEN,
                    zorder=6)
        for (C, se, ye) in eyes:
            ax.add_patch(patches.Circle((se, ye), rs, fc="white", ec=INK, lw=1.0, zorder=4))
            ax.plot([se], [ye], marker="o", ms=2.5, color=INK, zorder=5)
        ax.plot([0], [0], marker="+", ms=10, mew=1.5, color=RED, zorder=6)
        setup_view(ax, "forward distance from M along f = −Y (cm)",
                   ("X − X$_M$ (cm)  [+X = subject's left]" if view == "top"
                    else "Z − Z$_M$ (cm)  [up]"))
    axT.set_title("TOP VIEW — h = −1, 0, +1 (v = 0)", fontsize=12.5, loc="left",
                  fontweight="bold")
    axS.set_title("SIDE VIEW — v = −1, 0, +1 (h = 0); both eyes coincide in projection",
                  fontsize=12.5, loc="left", fontweight="bold")
    # angle tables inside the views
    verg0 = [b - a for v, a, b, c in yaw_txt if v == 0][0]
    tx = "computed yaw (°, + towards +X)\n" + "\n".join(
        "h = %2s:  L %s   R %s" % (sgn(v, 0), sgn(a, 2), sgn(b, 2)) for v, a, b, c in yaw_txt)
    tx += "\nvergence at h = v = 0: %.2f°" % verg0
    axT.text(-2.4, yhi - 0.5, tx, fontsize=10.5, family=MONO, va="top", ha="left",
             bbox=dict(fc="white", ec=GRID, boxstyle="square,pad=0.4"), zorder=8)
    tx = "computed pitch (°, + up)\n" + "\n".join(
        "v = %2s:  L %s   R %s" % (sgn(v, 0), sgn(a, 2), sgn(b, 2)) for v, a, b, c in pit_txt)
    axS.text(-2.4, yhi - 0.5, tx, fontsize=10.5, family=MONO, va="top", ha="left",
             bbox=dict(fc="white", ec=GRID, boxstyle="square,pad=0.4"), zorder=8)

    # zoom insets near the eyes
    xz = 1.15 + lw_ + 0.85
    zs, zy = (-1.5, 5.5), (-4.3, 4.3)
    wz = 3.0
    hz = wz * (zy[1] - zy[0]) / (zs[1] - zs[0])
    zoom_mag = (wz / (zs[1] - zs[0])) / (lw_ / (smax - smin))
    for ax_src, view, yb in ((axT, "top", y_top + vh - hz),
                             (axS, "side", y_top - vgap - hz)):
        ax_src.add_patch(patches.Rectangle((zs[0], zy[0]), zs[1] - zs[0], zy[1] - zy[0],
                                           fill=False, ec=INK2, lw=0.8, ls=":", zorder=7))
        az = S.axes(xz, yb, wz, hz)
        az.set_xlim(*zs); az.set_ylim(*zy)
        eyes = [(pkg.CL, sL, xL_ if view == "top" else zL, "L"),
                (pkg.CR, sR, xR_ if view == "top" else zR, "R")]
        if view == "side":
            eyes = eyes[:1]
        for val in (-1, 0, 1):
            T = T_of(val, 0) if view == "top" else T_of(0, val)
            sT, xT_, zT = pkg.coords(T)
            yT = xT_ if view == "top" else zT
            col, ls = shades[val]
            for (C, se, ye, lab) in eyes:
                az.plot([se, sT], [ye, yT], color=col, ls=ls, lw=shades_lw[val])
        for (C, se, ye, lab) in eyes:
            az.add_patch(patches.Circle((se, ye), rs, fc="white", ec=INK, lw=1.0, zorder=3,
                                        alpha=1.0))
            az.plot([se, zs[1]], [ye, ye], color=INK2, lw=0.8, ls=":", zorder=4)
            az.plot([se], [ye], marker="o", ms=3, color=INK, zorder=5)
        az.plot([0], [0], marker="+", ms=11, mew=1.5, color=RED, zorder=6)
        # angle arcs for the v=0/h=0 ray
        T0 = T_of(0, 0) if view == "top" else T_of(0, 1)
        for (C, se, ye, lab) in eyes:
            sT, xT_, zT = pkg.coords(T0)
            yT = xT_ if view == "top" else zT
            ang = math.degrees(math.atan2(yT - ye, sT - se))
            yaw, pit = yaw_pitch(C, T0, pkg)
            val = yaw if view == "top" else pit
            r_arc = 2.6
            az.plot([se, se + 3.9 * math.cos(math.radians(ang))],
                    [ye, ye + 3.9 * math.sin(math.radians(ang))], color=ACC_MID, lw=1.4, zorder=5)
            t = np.radians(np.linspace(min(0, ang), max(0, ang), 40))
            az.plot(se + r_arc * np.cos(t), ye + r_arc * np.sin(t), color=INK, lw=1.0, zorder=5)
            lbl = ("%s yaw %s°" % (lab, sgn(val, 2)) if view == "top"
                   else "pitch %s° (L = R)" % sgn(val, 2))
            off = (0.55 if ang >= 0 else -0.55) if view == "top" else 0.6
            az.text(se + r_arc + 0.15, ye + off + (0.6 if view == "side" else 0), lbl,
                    fontsize=10, ha="left", va="center", path_effects=HALO_THIN, zorder=8)
        setup_view(az, "forward (cm)", "X − X$_M$ (cm)" if view == "top" else "Z − Z$_M$ (cm)")
        az.tick_params(labelsize=9)
        az.set_title("Detail near the eyes (×%.1f)" % zoom_mag, fontsize=11.5, loc="left",
                     fontweight="bold")

    # front view of the fixation plane
    xf = xz + wz + 1.0
    wf = FW - xf - 0.4
    axF = S.axes(xf, y_top + vh - wf, wf, wf)
    lim = 100 * max(pkg.bound_h, pkg.bound_v) + 2.0
    axF.set_xlim(-lim, lim); axF.set_ylim(-lim, lim)
    bh, bv = 100 * pkg.bound_h, 100 * pkg.bound_v
    axF.add_patch(patches.Rectangle((-bh, -bv), 2 * bh, 2 * bv, fill=False, ec=ACC, lw=1.6))
    tt = np.linspace(0, 2 * math.pi, 200)
    axF.plot(bh * np.sin(tt), bv * np.cos(tt), color=ACC_LIGHT, lw=1.0, ls="--")
    axF.plot([0], [0], marker="o", ms=6, mfc="white", mec=RED, mew=1.8)
    axF.text(0.3, -0.35, "F$_0$", fontsize=11, color=RED, va="top")
    for val in (-1, 1):
        for (h, v) in ((val, 0), (0, val)):
            _, x_, z_ = pkg.coords(T_of(h, v))
            axF.plot([x_], [z_], marker="s", ms=7, mfc="white", mec=ACC, mew=1.6)
    for rec, (s_, x_, z_) in zip(pkg.meta, seqc):
        axF.plot([x_], [z_], marker="o", ms=6, color=GREEN, zorder=5)
        rr = math.hypot(x_, z_) + 1e-9
        axF.text(x_ + 1.0 * x_ / rr, z_ + 1.0 * z_ / rr, str(rec["frame"]), fontsize=11,
                 ha="center", va="center", color=INK)
    setup_view(axF, "X − X$_M$ (cm)  [image right]", "Z − Z$_M$ (cm)  [up]")
    axF.set_title("FIXATION PLANE at %.0f cm (camera view)" % (100 * d), fontsize=12,
                  loc="left", fontweight="bold")

    # metadata table of angles
    ytab = y_top + vh - wf - 0.95
    S.text(xf, ytab, "Per-frame angles from metadata (°)", fontsize=12, fontweight="bold",
           va="top")
    lines = [" k   yaw L   yaw R   pitch   verg."]
    for rec in pkg.meta:
        e = rec["extra"]
        lines.append("%2d %7s %7s %7s %6.2f" % (rec["frame"], sgn(e["yaw_deg_L"], 2),
                                                sgn(e["yaw_deg_R"], 2), sgn(e["pitch_deg_L"], 2),
                                                e["vergence_deg"]))
    S.text(xf, ytab - 0.4, "\n".join(lines), family=MONO, fontsize=10.5, va="top",
           linespacing=1.3)

    # consistency: recompute yaw/pitch from metadata targets & centres
    dev = 0.0
    for rec in pkg.meta:
        e = rec["extra"]
        for C, s_ in ((pkg.CL, "L"), (pkg.CR, "R")):
            y_, p_ = yaw_pitch(C, e["fixation_target"], pkg)
            dev = max(dev, abs(y_ - e["yaw_deg_" + s_]), abs(p_ - e["pitch_deg_" + s_]))

    # legend
    ly = y_top - vh - vgap - 0.85
    axl = S.axes(1.15, ly - 0.4, FW - 1.6, 0.42)
    axl.axis("off"); axl.set_xlim(0, FW - 1.6); axl.set_ylim(0, 0.42)
    yy = 0.21
    x = 0.0
    for val in (-1, 0, 1):
        col, ls = shades[val]
        axl.plot([x, x + 0.6], [yy, yy], color=col, ls=ls, lw=shades_lw[val])
        axl.text(x + 0.72, yy, "ray, h or v = %s" % sgn(val, 0), va="center",
                 fontsize=10.5)
        x += 2.55
    axl.plot(x + 0.15, yy, marker="s", ms=7, mfc="white", mec=ACC, mew=1.6)
    axl.text(x + 0.35, yy, "±1 target, computed from calibration", va="center",
             fontsize=10.5)
    x += 3.85
    axl.plot(x + 0.15, yy, marker="o", ms=5, color=GREEN)
    axl.text(x + 0.35, yy, "sequence target k, from metadata", va="center", fontsize=10.5)
    x += 3.35
    axl.plot(x + 0.15, yy, marker="+", ms=10, mew=1.5, color=RED)
    axl.text(x + 0.35, yy, "M", va="center", fontsize=10.5)

    S.caption(
        "Sequence targets (green, labelled k) are extra.fixation_target from metadata.json (in the top/side views "
        "they appear projected onto the fixation plane line); eye "
        "centres C_L/C_R and M are extra.eye_center_3D_L/R and eye_midpoint_3D. The ±1 "
        "targets (squares) are computed from calibration: T = M + %.3f m·f + h·%.3f "
        "m·r + v·%.3f m·u, with f = −Y (head forward), r = +X, u = +Z taken from "
        "camera_to_world_blender and the distances derived from the metadata targets. Yaw = "
        "atan2(g·r, g·f), pitch = atan2(g·u, |g_horizontal|), g = unit(T − C). "
        "Recomputing these for the 10 metadata targets reproduces extra.yaw_deg/pitch_deg to "
        "%.1e°. Dotted lines in the details: primary (straight-ahead) direction. Detail "
        "panels are enlargements of the dotted boxes; all panels are equal-aspect (to scale)."
        % (pkg.d_fix, pkg.bound_h, pkg.bound_v, dev), y=0.25, fontsize=10.5)
    return S.save(out, "gaze_rays")


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg", required=True, help="package directory (read-only)")
    ap.add_argument("--out", required=True, help="output directory for the boards")
    ap.add_argument("--before", help="older package directory for eye_detail_before_after")
    ap.add_argument("--contact-sheet", help="also write the sequence board to this path "
                    "(e.g. renders/contact_sheet.png); the only file allowed under renders/")
    a = ap.parse_args(argv)
    pkg_abs, out_abs = os.path.abspath(a.pkg), os.path.abspath(a.out)
    # boards may go to <pkg>/reports/... but never next to the official renders / metadata
    if out_abs.startswith(pkg_abs + os.sep) and not out_abs.startswith(os.path.join(pkg_abs, "reports") + os.sep):
        print("refusing to write inside the package outside reports/", file=sys.stderr)
        return 2
    if a.contact_sheet and os.path.basename(a.contact_sheet) != "contact_sheet.png":
        print("--contact-sheet must be a file named contact_sheet.png", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    pkg = Package(a.pkg)
    results = []
    for fn in (board_sequence, board_direction, board_eye_detail, board_camera, board_gaze):
        r = fn(pkg, a.out)
        if r:
            results.append(r)
    if a.before:
        results.append(board_eye_before_after(pkg, Package(a.before), a.out))
    for png, pdf, size in results:
        print("%-60s %dx%d  (+ %s)" % (png, size[0], size[1], os.path.basename(pdf)))
    if a.contact_sheet:
        import shutil
        shutil.copyfile(os.path.join(a.out, "sequence_board.png"), a.contact_sheet)
        print("contact sheet ->", a.contact_sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
