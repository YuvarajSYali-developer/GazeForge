"""Seeded 3D gradient (Perlin) noise in NumPy, evaluated at surface positions.
Evaluating in 3D object space (not UV space) makes every texture seamless across
GNM's UV seams: two texels on either side of a seam map to the same 3D point."""
import numpy as np

_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                  [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], np.float32)


class Perlin3:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int32)
        self.perm = np.concatenate([p, p])
        self.offset = rng.uniform(-1000, 1000, 3).astype(np.float32)

    def _h(self, ix, iy, iz):
        P = self.perm
        return P[(P[(P[ix & 255] + iy) & 255] + iz) & 255] % 12

    def __call__(self, pts, chunk=1_000_000):
        out = np.empty(len(pts), np.float32)
        for s in range(0, len(pts), chunk):
            p = pts[s:s + chunk].astype(np.float32) + self.offset
            i = np.floor(p).astype(np.int32)
            f = p - i
            u = f * f * f * (f * (f * 6 - 15) + 10)
            res = 0
            acc = []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        g = _GRAD[self._h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)]
                        acc.append((g * (f - np.array([dx, dy, dz], np.float32))).sum(1))
            x00 = acc[0] + u[:, 2] * (acc[1] - acc[0]); x01 = acc[2] + u[:, 2] * (acc[3] - acc[2])
            x10 = acc[4] + u[:, 2] * (acc[5] - acc[4]); x11 = acc[6] + u[:, 2] * (acc[7] - acc[6])
            y0 = x00 + u[:, 1] * (x01 - x00); y1 = x10 + u[:, 1] * (x11 - x10)
            out[s:s + chunk] = y0 + u[:, 0] * (y1 - y0)
            del res
        return out


def fbm(pts, scale_m, octaves, seed, gain=0.5, lac=2.0, ridged=False):
    """Fractal noise; scale_m = feature size of the first octave in metres. Output ~[-1,1]."""
    total = np.zeros(len(pts), np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0 / scale_m
    for o in range(octaves):
        n = Perlin3(seed * 101 + o)(pts * freq)
        if ridged:
            n = 1.0 - 2.0 * np.abs(n)
        total += amp * n
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm
