"""Contact sheets of every texture set: skin UV tile (1001) + left/right iris tiles (1002/1004) + UV wireframe
overlay, for slides and quick review. Usage: python tools/make_previews.py textures/ GNM_NPZ out_dir"""
import glob
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gazeforge_gnm.gnm_model import GNMHead

root, gnm_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
g = GNMHead(gnm_path)
font = ImageFont.load_default()
cells = []
for d in sorted(glob.glob(os.path.join(root, "set*_*"))):
    sid = os.path.basename(d)[3:5]
    skin = Image.open(f"{d}/set{sid}_albedo.1001.png").convert("RGB").resize((1024, 1024), Image.LANCZOS)
    # UV wireframe of tile 1001 (quads), half-transparent
    wire = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    dr = ImageDraw.Draw(wire)
    sel = g.quad_tiles == 1001
    for q in g.quad_uvs[sel][::1]:
        pts = [(u * 1024, (1 - v) * 1024) for u, v in q]
        dr.line(pts + [pts[0]], fill=(255, 255, 255, 38), width=1)
    skin_w = Image.alpha_composite(skin.convert("RGBA"), wire).convert("RGB")
    def crop_iris(t):
        im = Image.open(f"{d}/set{sid}_albedo.{t}.png").convert("RGB")
        w = im.width
        return im.crop((int(w * 0.30), int(w * 0.30), int(w * 0.70), int(w * 0.70))).resize((512, 512), Image.LANCZOS)
    sheet = Image.new("RGB", (2048 + 512, 1024), (18, 20, 26))
    sheet.paste(skin, (0, 0)); sheet.paste(skin_w, (1024, 0))
    sheet.paste(crop_iris(1002), (2048, 0)); sheet.paste(crop_iris(1004), (2048, 512))
    dd = ImageDraw.Draw(sheet)
    for x, y, txt in ((8, 8, f"set {sid}  tile 1001 skin albedo (4096 px)"), (1032, 8, "GNM UV layout overlay"),
                      (2056, 8, "tile 1002 left iris"), (2056, 520, "tile 1004 right iris")):
        dd.rectangle((x - 4, y - 4, x + 6 * len(txt) + 4, y + 12), fill=(0, 0, 0))
        dd.text((x, y), txt, fill=(255, 255, 255), font=font)
    sheet.save(os.path.join(out, f"set{sid}_uv_preview.jpg"), quality=90)
    cells.append((sid, skin, crop_iris(1002)))
# overview grid of all sets
W = Image.new("RGB", (4 * 512, 2 * 640), (18, 20, 26))
for i, (sid, skin, iris) in enumerate(cells):
    x, y = (i % 4) * 512, (i // 4) * 640
    W.paste(skin.resize((512, 512)), (x, y)); W.paste(iris.resize((128, 128)), (x + 384, y + 512))
    ImageDraw.Draw(W).text((x + 8, y + 520), f"set {sid}", fill=(255, 255, 255), font=font)
W.save(os.path.join(out, "all_sets_overview.jpg"), quality=90)
print("previews:", len(cells))
