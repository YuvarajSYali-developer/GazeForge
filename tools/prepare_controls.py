"""Derive named facial controls (GNM expression coefficient vectors) and identities.

Uses GNM's own semantic decoders (class means, z = 0) as the source of
anatomically plausible motion, then isolates single actions (blink, brow raise,
lid raise, ...) per eye with a masked ridge least-squares fit in GNM's own
100-component eye-region expression basis. Output: assets/gnm_controls.npz
(coefficient vectors only; vertices are always produced by the GNM basis).
"""
import json
import sys
import numpy as np
root = sys.argv[1]
sys.path.insert(0, root)
from gazeforge_gnm.gnm_model import GNMHead
from gazeforge_gnm.semantic import Semantic

gnm = GNMHead(sys.argv[2])
sem = Semantic(sys.argv[3])
V = gnm.template
B = gnm.expr_basis.astype(np.float64)
L = np.loadtxt(f"{sys.argv[3]}/../landmarks/head_sparse_68.txt")
lm_idx = L[:, 0::2].astype(int); lm_w = L[:, 1::2]


def lms(v):
    return (v[lm_idx] * lm_w[..., None]).sum(1)


def disp(e):
    return np.tensordot(e, B, axes=(0, 0))


def side_slices(side):
    return slice(0, 100) if side == "L" else slice(100, 200)


g = gnm.group
eye_j = {"L": gnm.joints0[2], "R": gnm.joints0[3]}
sx = {"L": 1.0, "R": -1.0}


def masks(side):
    s = sx[side]
    side_m = (V[:, 0] * s) > -0.004
    brow = (g("left_brow_region" if side == "L" else "right_brow_region") +
            g("middle_brow_region") * side_m + g("forehead_region") * side_m)
    brow = np.clip(brow, 0, 1)
    orb = np.clip(g("left_orbital_region" if side == "L" else "right_orbital_region") +
                  g("eye_sockets") * side_m, 0, 1) * (1 - brow)
    upper = orb * (V[:, 1] > eye_j[side][1] - 0.0015)
    lower = orb * (V[:, 1] <= eye_j[side][1] - 0.0015)
    return brow, orb, upper, lower


def fit(side, target, weight, ridge=1e-4):
    """min_c sum_v weight_v |B_side^T c - target_v|^2 + ridge |c|^2"""
    sl = side_slices(side)
    Bs = B[sl]                                   # (100, V, 3)
    sup = (np.abs(Bs).max(axis=(0, 2)) > 1e-7) | (weight > 0)
    w = np.sqrt(np.repeat(weight[sup], 3))
    A = Bs[:, sup, :].reshape(100, -1).T * w[:, None]
    b = target[sup].reshape(-1) * w
    AtA = A.T @ A
    c = np.linalg.solve(AtA + ridge * np.trace(AtA) / 100 * np.eye(100), A.T @ b)
    e = np.zeros(383); e[sl] = c
    return e


def aperture(v, side):
    l = lms(v)
    if side == "L":
        return ((l[43, 1] - l[47, 1]) + (l[44, 1] - l[46, 1])) / 2   # signed: <0 = lids crossed
    return ((l[37, 1] - l[41, 1]) + (l[38, 1] - l[40, 1])) / 2


def brow_y(v, side):
    l = lms(v)
    return l[22:27, 1].mean() if side == "L" else l[17:22, 1].mean()


src = {n: disp(sem.expression(n)) for n in
       ["wink_left", "wink_right", "surprise", "squint", "disgust", "happy",
        "compress_face", "snarl", "corners_down"]}
report = {"neutral_aperture_mm": {s: aperture(V, s) * 1e3 for s in "LR"}}
for n, D in src.items():
    report[n] = {s: {"aperture_mm": aperture(V + D, s) * 1e3,
                     "brow_dy_mm": (brow_y(V + D, s) - brow_y(V, s)) * 1e3} for s in "LR"}

controls = {}
others = lambda side: np.ones(len(V))  # noqa: E731
for side in "LR":
    brow, orb, upper, lower = masks(side)
    rest = np.clip(1 - brow - orb, 0, 1)
    wink = src["wink_left" if side == "L" else "wink_right"]
    # blink: lids follow the wink, brow/forehead stay still
    tgt = wink * orb[:, None]
    e = fit(side, tgt, orb + 4 * brow + rest)
    a0 = aperture(V, side)
    lo, hi = 0.5, 2.0
    for _ in range(40):  # scale so the lids just meet (aperture ~0.15 mm)
        m = (lo + hi) / 2
        if aperture(V + disp(e * m), side) > 0.00015:
            lo = m
        else:
            hi = m
    controls[f"blink_{side}"] = e * hi
    # upper lid down only (lid follows downward gaze); lower lid fixed
    e = fit(side, disp(controls[f"blink_{side}"]) * upper[:, None],
            upper + 3 * lower + 4 * brow + rest)
    controls[f"upper_lid_down_{side}"] = e
    # upper lid raise (eyes widen), from 'surprise' upper-lid motion x2
    # 'surprise' widens mainly via the lower lid, so the upper-lid lift uses a geometric
    # target (margin up 2 mm, Gaussian falloff 5 mm) projected into GNM's eye basis.
    l = lms(V)
    margin = l[[43, 44]] if side == "L" else l[[37, 38]]
    dist = np.min(np.linalg.norm(V[:, None, :] - margin[None], axis=2), axis=1)
    tgt = np.zeros_like(V)
    tgt[:, 1] = 0.002 * np.exp(-(dist / 0.005) ** 2) * upper
    e = fit(side, tgt, upper + 2 * lower + 2 * brow + rest)
    controls[f"upper_lid_raise_{side}"] = e
    # lower lid raise (squint / Duchenne), from 'squint' lower-lid motion
    e = fit(side, src["squint"] * lower[:, None], lower + 2 * upper + 2 * brow + rest)
    controls[f"lower_lid_raise_{side}"] = e
    # brow raise: 'surprise' brow motion scaled to ~6 mm
    D = src["surprise"] * brow[:, None]
    dy = brow_y(V + D, side) - brow_y(V, side)
    e = fit(side, D * (0.006 / max(dy, 1e-4)), brow + 0.5 * orb + rest)
    controls[f"brow_raise_{side}"] = e
    # brow lower (corrugator / concentration), from 'disgust' brow motion scaled to ~3 mm down
    D = src["disgust"] * brow[:, None]
    dy = brow_y(V + D, side) - brow_y(V, side)
    k = -0.003 / dy if dy < -1e-5 else 1.0
    e = fit(side, D * k, brow + 0.5 * orb + rest)
    controls[f"brow_lower_{side}"] = e

controls["pupil_dilate"] = np.eye(383)[382]
hap = sem.expression("happy")
sm = np.zeros(383); sm[200:350] = hap[200:350]
controls["smile"] = sm * 0.6

check = {}
for k, e in controls.items():
    v = V + disp(e)
    check[k] = {s: {"aperture_mm": round(aperture(v, s) * 1e3, 3),
                    "brow_dy_mm": round((brow_y(v, s) - brow_y(V, s)) * 1e3, 3)} for s in "LR"}

# identities: GNM semantic identity decoder, class means + small seeded latent
rng = np.random.default_rng(7)
ids = {}
for gender in ["female", "male"]:
    for eth in ["middle_eastern", "asian", "white", "black"]:
        ids[f"{gender}_{eth}"] = sem.identity(gender, eth, z=rng.normal(size=64) * 0.35)

np.savez(f"{root}/assets/gnm_controls.npz",
         control_names=np.array(list(controls)), controls=np.stack(list(controls.values())),
         identity_names=np.array(list(ids)), identities=np.stack(list(ids.values())))
json.dump({"semantic_sources": report, "controls": check}, open(f"{root}/assets/gnm_controls_report.json", "w"),
          indent=1, default=float)
for k, v in check.items():
    print(f"{k:22s}", v)
