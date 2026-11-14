"""Rasterise GNM triangles into a UDIM tile: per-texel triangle id + barycentrics,
so any per-vertex quantity (3D position, normal, vertex-group weight) can be
interpolated into texture space. Textures built from 3D positions are seamless
across UV seams by construction."""
import numpy as np
from scipy import ndimage


def rasterize(tri_uv, res):
    """tri_uv: (T,3,2) in [0,1]. Returns tid (res,res) int32 (-1 empty), bary (res,res,3) float32.
    Image row 0 = v=1 (top), as Blender/OpenGL images are stored bottom-up but displayed top-down."""
    tid = np.full((res, res), -1, np.int32)
    bary = np.zeros((res, res, 3), np.float32)
    P = tri_uv * res
    P[..., 1] = res - P[..., 1]           # to image rows
    for t in range(len(P)):
        a, b, c = P[t]
        x0 = int(max(np.floor(min(a[0], b[0], c[0]) - 0.5), 0)); x1 = int(min(np.ceil(max(a[0], b[0], c[0]) + 0.5), res - 1))
        y0 = int(max(np.floor(min(a[1], b[1], c[1]) - 0.5), 0)); y1 = int(min(np.ceil(max(a[1], b[1], c[1]) + 0.5), res - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        l0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        l1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        yy, xx = np.nonzero(inside)
        tid[y0 + yy, x0 + xx] = t
        bary[y0 + yy, x0 + xx] = np.stack([l0[inside], l1[inside], l2[inside]], -1)
    return tid, bary


class TileRaster:
    def __init__(self, gnm, tile, res, positions=None):
        sel = np.nonzero(gnm.triangle_tiles == tile)[0]
        uv = gnm.triangle_uvs[sel]                     # per-region coords already in [0,1]
        self.tris = gnm.triangles[sel]
        self.res = res
        self.tid, self.bary = rasterize(uv, res)
        self.mask = self.tid >= 0
        self.yx = tuple(a.astype(np.int32) for a in np.nonzero(self.mask))
        t = self.tid[self.yx]
        self.vidx = self.tris[t].astype(np.int32)       # (N,3)
        self.w = self.bary[self.yx]                     # (N,3)
        del self.bary, self.tid
        self.n = len(self.w)
        self.P = None if res > 2048 else self.interp(gnm.template if positions is None else positions)
        self._fill = None

    def interp(self, per_vertex, sl=slice(None)):
        a = np.asarray(per_vertex, np.float32)
        vidx, w = self.vidx[sl], self.w[sl]
        out = None
        for k in range(3):
            term = a[vidx[:, k]]
            term *= (w[:, k] if a.ndim == 1 else w[:, k:k + 1])
            out = term if out is None else (out.__iadd__(term))
        return out

    def to_image(self, vals, fill=0.0):
        vals = np.asarray(vals)
        shape = (self.res, self.res) + vals.shape[1:]
        img = np.full(shape, fill, np.float32)
        img[self.yx] = vals
        return img

    def put(self, img, sl, vals):
        img[self.yx[0][sl], self.yx[1][sl]] = vals

    def chunks(self, size=1_500_000):
        for s in range(0, self.n, size):
            yield slice(s, min(s + size, self.n))

    def pad(self, img):
        """Fill every empty texel with its nearest valid texel (no seams from filtering/mipmaps)."""
        if self._fill is None:
            _, idx = ndimage.distance_transform_edt(~self.mask, return_indices=True)
            self._fill = idx.astype(np.int32)
            del idx
        iy, ix = self._fill
        return img[iy, ix]
