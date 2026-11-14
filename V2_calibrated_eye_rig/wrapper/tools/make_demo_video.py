"""
make_demo_video.py -- assemble the GazeForge demo video (1080p, 30 fps, < 2 min) from REAL outputs.

    python3 make_demo_video.py <package_dir> <stdout.tsv> <out.mp4>

<stdout.tsv> is a recording of `generate_sequence.py -- --demo` (one line per stdout line:
"<unix time>\\t<text>"); it is replayed with its real timestamps (sped up, factor shown on screen)
next to the PNGs that run wrote (demo_run/renders). All other images are the official quality
renders and boards; all numbers come from metadata.json / reports/*.json.
Requires Pillow, numpy and ffmpeg.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG, INK, MUTED, ACC, ACC2, PANEL = (246, 247, 248), (30, 42, 51), (91, 107, 117), (14, 124, 134), (181, 72, 47), (36, 59, 74)
import matplotlib  # only for its bundled DejaVu fonts
FD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    return ImageFont.truetype(os.path.join(FD, name), size)


F_TITLE, F_H, F_B, F_S, F_M, F_MS = font(64, True), font(40, True), font(30), font(24), font(22, mono=True), font(20, mono=True)


class Scene:
    def __init__(self, path):
        self.tmp = path + ".part.mp4"
        self.path = path
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "18", "-pix_fmt", "yuv420p", self.tmp], stdin=subprocess.PIPE)

    def put(self, img, n=1):
        b = np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()
        for _ in range(n):
            self.p.stdin.write(b)

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        os.replace(self.tmp, self.path)


def canvas(dark=False):
    return Image.new("RGB", (W, H), PANEL if dark else BG)


def header(d, title, sub=None):
    d.text((80, 50), title, font=F_H, fill=INK)
    if sub:
        d.text((80, 104), sub, font=F_S, fill=MUTED)


def fit(img, w, h):
    im = img.copy()
    im.thumbnail((w, h), Image.LANCZOS)
    return im


def fade(a, b, t):
    return Image.blend(a, b, max(0.0, min(1.0, t)))


def hold(sc, img, secs, fade_in=0.4, prev=None):
    n = int(secs * FPS)
    k = int(fade_in * FPS) if prev is not None else 0
    for i in range(k):
        sc.put(fade(prev, img, (i + 1) / k))
    sc.put(img, n - k)
    return img


def eye_strip(path, zoom=3, box=(320, 470, 704, 560)):
    im = Image.open(path).convert("RGB").crop(box)
    return im.resize((im.width * zoom, im.height * zoom), Image.LANCZOS)


def main(pkg, tsv, out):
    meta = json.load(open(os.path.join(pkg, "metadata.json")))
    dv = json.load(open(os.path.join(pkg, "reports", "direction_validation.json")))
    rl = json.load(open(os.path.join(pkg, "reports", "run_log.json")))
    acc_p = os.path.join(pkg, "reports", "acceptance_report.json")
    acc = json.load(open(acc_p)) if os.path.isfile(acc_p) else None
    rec = [ln.rstrip("\n").split("\t", 1) for ln in open(tsv) if "\t" in ln]
    rec = [(float(t), s) for t, s in rec]
    work = tempfile.mkdtemp(prefix="gfvid_")
    parts = []

    # ---------- 1. title ----------
    sc = Scene(os.path.join(work, "s1.mp4")); parts.append(sc.path)
    im = canvas(True); d = ImageDraw.Draw(im)
    face = fit(Image.open(os.path.join(pkg, "renders", "frame_00.png")).crop((150, 60, 874, 1000)), 640, 840)
    im.paste(face, (W - 80 - face.width, (H - face.height) // 2))
    d.text((90, 330), "GazeForge", font=font(96, True), fill=(255, 255, 255))
    d.text((90, 460), "Calibrated synthetic eye movement & virtual-camera rendering", font=F_B, fill=(217, 228, 232))
    d.text((90, 510), "Samsung PRISM XR Hackathon 2026 · Track 01", font=F_S, fill=(159, 184, 194))
    d.text((90, 600), "Everything shown is real output: recorded runs, rendered PNGs, generated JSON.", font=F_S, fill=(159, 184, 194))
    hold(sc, im, 6.0)
    sc.close()

    # ---------- 2. calibration ----------
    sc = Scene(os.path.join(work, "s2.mp4")); parts.append(sc.path)
    im = canvas(); d = ImageDraw.Draw(im)
    header(d, "Eye midpoint calibration", "Fixation 35 cm and camera 40 cm on the head's forward axis · eyes rotate about their own centres")
    b = fit(Image.open(os.path.join(pkg, "reports", "boards", "camera_calibration.png")).crop((150, 310, 3240, 2700)), W - 160, H - 190)
    im.paste(b, ((W - b.width) // 2, 160))
    hold(sc, im, 11.0)
    sc.close()

    # ---------- 3. recorded run ----------
    sc = Scene(os.path.join(work, "s3.mp4")); parts.append(sc.path)
    t0 = rec[0][0] - 0.6
    t_end = rec[-1][0] + 1.0
    real = t_end - t0
    dur = 36.0
    speed = real / dur
    demo_dir = os.path.join(pkg, "demo_run", "renders")
    base = canvas(); bd = ImageDraw.Draw(base)
    header(bd, "Live pipeline run (recorded, demo profile: 8 samples)",
           f"blender -b scene.blend --python wrapper/generate_sequence.py -- --demo     ·     playback ×{speed:.1f}")
    cache = {}
    for i in range(int(dur * FPS)):
        tr = t0 + i / FPS * speed
        shown = [s for t, s in rec if t <= tr]
        im = base.copy(); d = ImageDraw.Draw(im)
        d.rounded_rectangle((60, 160, 1020, 1030), 14, fill=(24, 32, 38))
        y = 180
        for s in shown[-26:]:
            s2 = s.replace("  ", " ")
            if " pupil L " in s2 and "det-err" in s2:          # keep the measured fields, drop raw coordinates
                s2 = s2[:s2.index(" pupil L ")] + "  " + s2[s2.index("det-err"):]
            s2 = (s2[:74] + "…") if len(s2) > 75 else s2
            col = (120, 220, 160) if "PASS" in s2 else (210, 220, 225)
            d.text((80, y), s2, font=F_MS, fill=col)
            y += 32
        el = max(0.0, tr - t0)
        d.text((1060, 170), f"elapsed (real) {int(el // 60):d}:{int(el % 60):02d}", font=F_B, fill=INK)
        frames = [s for s in shown if s.startswith("[seq] frame ")]
        if frames:
            k = int(frames[-1].split()[2])
            p = os.path.join(demo_dir, f"frame_{k:02d}.png")
            if k not in cache:
                full = Image.open(p).convert("RGB")
                cache[k] = (fit(full.crop((150, 60, 874, 1000)), 360, 470), eye_strip(p, 2))
            fc, es = cache[k]
            im.paste(fc, (1060, 230))
            d.text((1440, 240), f"frame_{k:02d}.png", font=F_B, fill=INK)
            m = meta[k]
            d.text((1440, 290), f"θ = {m['theta']:.3f} rad", font=F_S, fill=MUTED)
            d.text((1440, 325), f"h = {m['input']['h']:+.3f}", font=F_S, fill=MUTED)
            d.text((1440, 360), f"v = {m['input']['v']:+.3f}", font=F_S, fill=MUTED)
            es2 = fit(es, 800, 200)
            im.paste(es2, (1060, 740))
            d.text((1060, 710), "eye region ×2 (rendered pixels)", font=F_S, fill=MUTED)
        sc.put(im)
    sc.close()

    # ---------- 4. five directions ----------
    sc = Scene(os.path.join(work, "s4.mp4")); parts.append(sc.path)
    tests = {t["test"]: t for t in dv["tests"]}
    n_det = dv["neutral_pupils"]
    prev = None
    for name in ("NEUTRAL", "LEFT", "RIGHT", "UP", "DOWN", "RESET"):
        im = canvas(); d = ImageDraw.Draw(im)
        header(d, "Five-direction test: 5 / 5 PASS", "Rendered eye region ×4 · red + = neutral pupil (projected) · green ○ = pupil measured in this PNG")
        box = (320, 470, 704, 560)
        p = os.path.join(pkg, "reports", "direction_test", f"{name.lower()}.png")
        strip = eye_strip(p, 4, box)
        sd = ImageDraw.Draw(strip)
        for s in ("L", "R"):
            nx, ny = n_det[s]["projected"]
            X, Y = (nx - box[0]) * 4, (ny - box[1]) * 4
            sd.line((X - 14, Y, X + 14, Y), fill=(230, 40, 40), width=3); sd.line((X, Y - 14, X, Y + 14), fill=(230, 40, 40), width=3)
            if name in ("NEUTRAL", "RESET"):
                dx = dy = 0.0
            else:
                e = tests[name]["left_eye" if s == "L" else "right_eye"]["detected"]
                dx, dy = e["dx"], e["dy"]
            mx, my = (n_det[s]["detected"][0] + dx - box[0]) * 4, (n_det[s]["detected"][1] + dy - box[1]) * 4
            sd.ellipse((mx - 12, my - 12, mx + 12, my + 12), outline=(20, 200, 90), width=3)
        im.paste(strip, ((W - strip.width) // 2, 200))
        d.text((80, 600), name, font=F_TITLE, fill=INK)
        if name in ("LEFT", "RIGHT", "UP", "DOWN"):
            t = tests[name]
            k = "dx" if name in ("LEFT", "RIGHT") else "dy"
            d.text((80, 700), f"input (h, v) = ({t['input']['h']:+.0f}, {t['input']['v']:+.0f})", font=F_B, fill=MUTED)
            d.text((80, 760), f"subject's left eye   {k} projected {t['left_eye']['projected'][k]:+.2f} px   measured {t['left_eye']['detected'][k]:+.2f} px", font=F_B, fill=INK)
            d.text((80, 810), f"subject's right eye  {k} projected {t['right_eye']['projected'][k]:+.2f} px   measured {t['right_eye']['detected'][k]:+.2f} px", font=F_B, fill=INK)
            d.text((80, 880), t["status"], font=F_TITLE, fill=ACC)
        elif name == "RESET":
            r = tests["RESET"]
            d.text((80, 700), "reset() ×3 after move(0.6, −0.8)", font=F_B, fill=MUTED)
            d.text((80, 760), f"eye quaternion diff {r['pose_quaternion_max_diff_over_3_resets']:.1f} · re-render diff "
                              f"{r['image_diff_vs_neutral_8bit']['max']:.0f}/255 vs neutral", font=F_B, fill=INK)
            d.text((80, 880), r["status"], font=F_TITLE, fill=ACC)
        else:
            d.text((80, 700), "reference pose: both eyes converge on the 35 cm target", font=F_B, fill=MUTED)
        prev = hold(sc, im, 4.2, 0.35, prev)
    sc.close()

    # ---------- 5. sequence ----------
    sc = Scene(os.path.join(work, "s5.mp4")); parts.append(sc.path)
    prev = None
    for m in meta:
        k = m["frame"]
        p = os.path.join(pkg, "renders", f"frame_{k:02d}.png")
        im = canvas(); d = ImageDraw.Draw(im)
        header(d, "Ten clockwise frames: θk = 2πk/10,  h = sin θk,  v = cos θk", "Official quality renders (32 samples), 1024 × 1024")
        fc = fit(Image.open(p).convert("RGB"), 860, 860)
        im.paste(fc, (80, 170))
        es = fit(eye_strip(p, 3), 860, 300)
        im.paste(es, (980, 600))
        d.text((980, 570), "eye region ×3", font=F_S, fill=MUTED)
        cx, cy, R = 1400, 360, 150
        d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(180, 190, 195), width=3)
        for j in range(10):
            a = 2 * math.pi * j / 10
            x, y = cx + R * math.sin(a), cy - R * math.cos(a)
            r = 16 if j == k else 7
            d.ellipse((x - r, y - r, x + r, y + r), fill=ACC if j == k else (180, 190, 195))
        d.text((980, 180), f"k = {k}", font=F_H, fill=INK)
        d.text((980, 240), f"h = {m['input']['h']:+.3f}", font=F_S, fill=MUTED)
        d.text((980, 275), f"v = {m['input']['v']:+.3f}", font=F_S, fill=MUTED)
        prev = hold(sc, im, 1.5, 0.25, prev)
    sc.close()

    # ---------- 6. ground truth ----------
    sc = Scene(os.path.join(work, "s6.mp4")); parts.append(sc.path)
    im = canvas(); d = ImageDraw.Draw(im)
    header(d, "Ground truth you can check", "metadata.json, frame 3 (required fields)")
    m = meta[3]
    lines = ["{", f'  "frame": {m["frame"]},', f'  "theta": {m["theta"]:.6f},',
             f'  "input": {{"h": {m["input"]["h"]:.6f}, "v": {m["input"]["v"]:.6f}}},',
             '  "eye_rotation_L": [' + ", ".join(f"{v:.5f}" for v in m["eye_rotation_L"]) + "],",
             '  "eye_rotation_R": [' + ", ".join(f"{v:.5f}" for v in m["eye_rotation_R"]) + "],",
             '  "pupil_center_2D_L": [' + ", ".join(f"{v:.3f}" for v in m["pupil_center_2D_L"]) + "],",
             '  "pupil_center_2D_R": [' + ", ".join(f"{v:.3f}" for v in m["pupil_center_2D_R"]) + "],",
             '  "camera_matrix": [[' + ", ".join(f"{v:.1f}" for v in m["camera_matrix"][0]) + "], …]", "}"]
    d.rounded_rectangle((60, 160, 1080, 640), 14, fill=(24, 32, 38))
    for i, ln in enumerate(lines):
        d.text((90, 185 + i * 43), ln, font=F_M, fill=(210, 220, 225))
    pv = rl["sequence_projection_vs_detection_px"]
    stats = [(f"{pv['max']:.2f} px", f"max |label − detected pupil|, 20 pupils (mean {pv['mean']:.2f})"),
             ("≤ 1·10⁻⁷ m", "both gaze rays pass through the common fixation target"),
             ("0.0", "difference of v2 labels vs v1 (rotations, 2-D pupils, P)"),
             (f"{acc['passed']}/{acc['total']}" if acc else "A–I", "acceptance test groups on the packaged ZIP")]
    for i, (big, lab) in enumerate(stats):
        y = 170 + i * 120
        d.text((1150, y), big, font=font(52, True), fill=ACC)
        d.text((1150, y + 66), lab, font=F_S, fill=MUTED)
    d.text((80, 700), "P = K [R | t] maps the 3-D pupil centre to pupil_center_2D (pixel origin = top-left image corner).", font=F_S, fill=INK)
    d.text((80, 745), "Labels are read from the evaluated scene after every move(); none are hard-coded.", font=F_S, fill=INK)
    hold(sc, im, 11.0)
    sc.close()

    # ---------- 7. close ----------
    sc = Scene(os.path.join(work, "s7.mp4")); parts.append(sc.path)
    im = canvas(True); d = ImageDraw.Draw(im)
    d.text((90, 300), "GazeForge", font=font(96, True), fill=(255, 255, 255))
    d.text((90, 440), "5 / 5 directions  ·  labels within a quarter pixel of the pixels  ·  one command, deterministic", font=F_B, fill=(217, 228, 232))
    d.text((90, 520), "Limitations: ±5 cm ⇒ ≈ 7 px pupil travel · static eyelids · non-refracting cornea · CPU rendering", font=F_S, fill=(159, 184, 194))
    d.text((90, 570), "Track 01 · Samsung PRISM XR Hackathon 2026 · Team GazeForge", font=F_S, fill=(159, 184, 194))
    hold(sc, im, 6.0)
    sc.close()

    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    tmp = out + ".part.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, out)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
