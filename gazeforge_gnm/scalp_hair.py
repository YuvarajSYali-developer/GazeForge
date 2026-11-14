"""Short scalp hair as explicit 3D strands (NumPy), grown along a combing field over the
GNM scalp and kept just above the surface. Rendered as Blender hair curves, rigidly attached to
the head bone (the scalp does not deform with facial expressions)."""
import numpy as np


class _NN:
    """Brute-force nearest neighbour (NumPy only, so it runs inside Blender's Python)."""

    def __init__(self, P):
        self.P = P.astype(np.float32)
        self.P2 = (self.P ** 2).sum(1)

    def query(self, Q, chunk=4000):
        Q = Q.astype(np.float32)
        idx = np.empty(len(Q), np.int64)
        for s in range(0, len(Q), chunk):
            q = Q[s:s + chunk]
            d = self.P2[None, :] - 2 * q @ self.P.T
            idx[s:s + chunk] = np.argmin(d, axis=1)
        return None, idx


def vertex_normals(V, tris):
    n = np.zeros_like(V)
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def grow(V, tris, density, n_strands=60000, length=0.016, seed=1, steps=6, female=False):
    """V: (N,3) GNM bind-pose vertices, tris: (T,3), density: (N,) in [0,1]. Returns (n_strands, steps+1, 3)."""
    rng = np.random.default_rng(seed)
    vn = vertex_normals(V, tris)
    a = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    p = a * density[tris].mean(1)
    p /= p.sum()
    t = rng.choice(len(tris), size=n_strands, p=p)
    r1, r2 = rng.random(n_strands), rng.random(n_strands)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    root = (V[tris[t]] * b[..., None]).sum(1)
    nrm = (vn[tris[t]] * b[..., None]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    dens_root = (density[tris[t]] * b).sum(1)
    sel = np.nonzero(density > 0.02)[0]
    kd = _NN(V[sel])
    # combing field: hair flows back from the hairline over the crown and down the back/sides,
    # with a slight side parting; females slightly longer and swept back
    crown = np.array([0.0, 0.37, -0.03])
    L = length * (0.55 + 0.45 * dens_root) * rng.uniform(0.8, 1.2, n_strands)
    pts = [root + nrm * 0.0002]
    pos = root.copy()
    seg = L / steps
    for k in range(steps):
        _, i = kd.query(pos)
        sp = V[sel][i]; sn = vn[sel][i]
        away = pos - crown
        comb = np.stack([0.35 * away[:, 0] / 0.08, -0.6 * np.ones(len(pos)), -1.0 * np.ones(len(pos))], 1)
        front = pos[:, 2] > 0.02
        comb[front] += np.array([0.25, 0.35, -0.6])      # front: swept back and slightly to the side
        comb += rng.normal(0, 0.15, comb.shape)
        d = comb - sn * (comb * sn).sum(1, keepdims=True)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        lift = (0.25 if female else 0.45) * (1 - k / steps)
        d = d + sn * lift
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = pos + d * seg[:, None]
        # stay above the scalp: height grows slightly along the strand (layered hair volume)
        _, i = kd.query(pos)
        sp = V[sel][i]; sn = vn[sel][i]
        h = ((pos - sp) * sn).sum(1)
        hmin = 0.0004 + 0.0018 * (k + 1) / steps * (0.6 if female else 1.0)
        pos = pos + sn * np.maximum(hmin - h, 0)[:, None]
        pts.append(pos.copy())
    return np.stack(pts, 1)
