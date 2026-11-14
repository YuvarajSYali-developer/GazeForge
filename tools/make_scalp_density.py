"""Per-vertex scalp-hair density (GNM vertex order) from the same hairline field used by the textures.
Usage: python tools/make_scalp_density.py GNM_NPZ LANDMARKS_TXT assets/gnm_scalp_density.npy"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gazeforge_gnm.gnm_model import GNMHead
from gazeforge_gnm.textures.skin import SkinContext, vertex_normals

g = GNMHead(sys.argv[1])
Lt = np.loadtxt(sys.argv[2])
LM = (g.template[Lt[:, 0::2].astype(int)] * Lt[:, 1::2][..., None]).sum(1)
ctx = SkinContext.__new__(SkinContext)
ctx.g, ctx.LM = g, LM
ctx.V = g.template.astype(np.float32)
ctx.vn = vertex_normals(g.template, g.triangles).astype(np.float32)
ctx.eyeL, ctx.eyeR = g.joints0[2], g.joints0[3]
ctx.brow_y = float(LM[17:27, 1].mean())
ctx.c_cheekL = (LM[13] * 0.35 + LM[45] * 0.25 + LM[54] * 0.4)
ctx.c_cheekR = (LM[3] * 0.35 + LM[36] * 0.25 + LM[48] * 0.4)
f = ctx.vertex_fields()
scalp = f["scalp"] * (g.group("skin") > 0.5)
np.save(sys.argv[3], scalp.astype(np.float32))
print("vertices with hair density > 0.5:", int((scalp > 0.5).sum()), "of", len(scalp))
