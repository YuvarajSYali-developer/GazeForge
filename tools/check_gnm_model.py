"""Check gazeforge_gnm.gnm_model against the official google/GNM numpy implementation."""
import sys
import numpy as np
sys.path[:0] = [sys.argv[2], sys.argv[3]]
from gnm.shape import gnm_numpy
from gazeforge_gnm.gnm_model import GNMHead
path = sys.argv[1]
ref = gnm_numpy.GNM.from_custom_file(path)
mine = GNMHead(path)
rng = np.random.default_rng(0)
worst = 0.0
for k in range(5):
    i = rng.normal(size=253); e = rng.normal(size=383) * 0.5
    r = rng.normal(size=(4, 3)) * 0.2; t = rng.normal(size=3) * 0.01
    a = np.asarray(ref(i.astype(np.float32), e.astype(np.float32), r.astype(np.float32), t.astype(np.float32)))
    b = mine(i, e, r, t)
    worst = max(worst, np.abs(a - b).max())
uv_ok = np.allclose(mine.uvs_udim(mine.quad_uvs, mine.quad_tiles), ref.quad_uvs_udim, atol=1e-6) and \
        np.array_equal(mine.quad_tiles, ref.quad_udim_tiles)
print(f"max |vertex difference| over 5 random parameter sets: {worst:.3e} m")
print("UDIM tiles and coordinates identical to official:", uv_ok)
assert worst < 1e-5 and uv_ok
