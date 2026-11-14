"""
imgutil.py -- tiny dependency-free image helpers (numpy + bpy only; Blender ships no PIL).

Images are numpy float arrays (H, W, 3) in [0, 1], row 0 = TOP of the image.
"""
import os

import bpy
import numpy as np


def load_rgb(path):
    img = bpy.data.images.load(os.path.abspath(path), check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1, :, :3].copy()


def save_rgb(path, arr):
    arr = np.clip(np.asarray(arr, dtype=np.float32), 0, 1)
    h, w, _ = arr.shape
    img = bpy.data.images.new("_imgutil_tmp", width=w, height=h, alpha=False)
    try:
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[..., :3] = arr[::-1]
        img.pixels.foreach_set(rgba.ravel())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        img.filepath_raw = os.path.abspath(path)
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)
    return os.path.abspath(path)


def downscale(arr, factor):
    """Box-filter downscale by an integer factor."""
    h, w, c = arr.shape
    h2, w2 = h // factor, w // factor
    return arr[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor, c).mean(axis=(1, 3))


def upscale_nearest(arr, factor):
    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


# ---------------------------------------------------------------- 5x7 bitmap font
_GLYPHS = {
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
    ".": ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "],
    ",": ["     ", "     ", "     ", "     ", " ##  ", "  #  ", " #   "],
    "+": ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
    "-": ["     ", "     ", "     ", "#####", "     ", "     ", "     "],
    "=": ["     ", "     ", "#####", "     ", "#####", "     ", "     "],
    ":": ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "],
    "/": ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
    "(": ["   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "],
    ")": [" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "],
    ">": [" #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   "],
    "Θ": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", " ### "],
    "π": ["     ", "#####", " # # ", " # # ", " # # ", " # # ", " # # "],
    "Π": ["     ", "#####", " # # ", " # # ", " # # ", " # # ", " # # "],   # str.upper() maps π -> Π
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "J": ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
    "Y": ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
}


def text_width(text, scale=2):
    return len(text) * 6 * scale


def draw_text(canvas, text, x, y, scale=2, color=(0.1, 0.1, 0.1)):
    """Draw upper-case text with the top-left corner at (x, y)."""
    H, W, _ = canvas.shape
    cx = x
    for ch in text.upper():
        g = _GLYPHS.get(ch)
        if g is not None:
            for r, row in enumerate(g):
                for c, bit in enumerate(row):
                    if bit == "#":
                        y0, x0 = y + r * scale, cx + c * scale
                        if 0 <= y0 < H and 0 <= x0 < W:
                            canvas[y0:y0 + scale, x0:x0 + scale] = color
        cx += 6 * scale
    return canvas


def draw_cross(canvas, x, y, size=6, color=(1, 0, 0), thick=1):
    H, W, _ = canvas.shape
    xi, yi = int(round(x)), int(round(y))
    for d in range(-size, size + 1):
        for t in range(-(thick // 2), thick // 2 + 1):
            if 0 <= yi + t < H and 0 <= xi + d < W:
                canvas[yi + t, xi + d] = color
            if 0 <= yi + d < H and 0 <= xi + t < W:
                canvas[yi + d, xi + t] = color
    return canvas


def draw_circle(canvas, x, y, r, color=(0, 0.7, 0), thick=1.0):
    H, W, _ = canvas.shape
    x0, x1 = max(0, int(x - r - 2)), min(W, int(x + r + 3))
    y0, y1 = max(0, int(y - r - 2)), min(H, int(y + r + 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.abs(np.hypot(xx + 0.5 - x, yy + 0.5 - y) - r)
    m = d <= thick * 0.5 + 0.25
    canvas[y0:y1, x0:x1][m] = color
    return canvas


def draw_line(canvas, x0, y0, x1, y1, color=(0.3, 0.3, 0.3), thick=2):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    H, W, _ = canvas.shape
    for t in np.linspace(0, 1, n):
        x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
        canvas[max(0, y - thick // 2):min(H, y + thick // 2 + 1), max(0, x - thick // 2):min(W, x + thick // 2 + 1)] = color
    return canvas


def draw_rect(canvas, x0, y0, x1, y1, color=(0.6, 0.6, 0.6), thick=1):
    for t in range(thick):
        canvas[y0 + t, x0:x1] = color
        canvas[y1 - 1 - t, x0:x1] = color
        canvas[y0:y1, x0 + t] = color
        canvas[y0:y1, x1 - 1 - t] = color
    return canvas
