"""
prepare_basemesh.py  --  one-off asset preparation (provenance script).

Source asset : MakeHuman / MPFB2 base mesh `base.obj` + macro targets (CC0 1.0)
               https://github.com/makehumancommunity/mpfb2  (src/mpfb/data)
Output       : wrapper/assets/head_basemesh.npz

What it does
  1. Parses the hm08 base mesh (vertices, UVs, quad faces, OBJ groups).
  2. Applies the MakeHuman macro targets for a young adult male with the
     default 1/3 : 1/3 : 1/3 ancestry blend (exactly how MakeHuman composes
     its default "male, 25 y" character).
  3. Converts MakeHuman units/axes (decimetres, Y-up, facing +Z) to Blender
     metres, Z-up, facing -Y:  (x, y, z)_mh  ->  0.1 * (x, -z, y).
  4. Least-squares-fits a sphere to each eye helper (`helper-l-eye`,
     `helper-r-eye`) -> anatomical eyeball centres + radii.
  5. Keeps only the `body` group faces of the head/neck region (no helpers,
     no joint cubes) to keep the render light.

Run (plain CPython + numpy):
    python wrapper/tools/prepare_basemesh.py <path/to/mpfb2/src/mpfb/data> wrapper/assets/head_basemesh.npz
This script is NOT needed to run the wrapper; the produced .npz is shipped.
"""
import gzip
import os
import sys

import numpy as np

TARGETS = {  # weight per macro target file (male, young, 1/3 ancestry each)
    "macrodetails/caucasian-male-young.target.gz": 1.0 / 3.0,
    "macrodetails/african-male-young.target.gz": 1.0 / 3.0,
    "macrodetails/asian-male-young.target.gz": 1.0 / 3.0,
    # open the palpebral fissure (default MakeHuman lids cover half the pupil): upper lid margin
    # 0.8 mm -> ~3.4 mm above the eye centre (0.8 weight): the lid overlaps the top of the iris as in a real eye,
    # so the pupil stays fully visible across the +/-8.1 deg vertical gaze range
    "eyes/l-eye-height1-incr.target.gz": 0.8,
    "eyes/l-eye-height2-incr.target.gz": 0.8,
    "eyes/r-eye-height1-incr.target.gz": 0.8,
    "eyes/r-eye-height2-incr.target.gz": 0.8,
}


def parse_obj(path):
    verts, uvs, faces, fuvs, fgroup = [], [], [], [], []
    group = None
    groups = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(t) for t in line.split()[1:3]])
            elif line.startswith("g "):
                group = line.split(None, 1)[1].strip()
                if group not in groups:
                    groups.append(group)
            elif line.startswith("f "):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    vi.append(int(parts[0]) - 1)
                    ti.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
                faces.append(vi)
                fuvs.append(ti)
                fgroup.append(groups.index(group))
    return np.array(verts), np.array(uvs), faces, fuvs, np.array(fgroup), groups


def apply_target(verts, path, weight):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.split()
            if len(p) != 4 or p[0].startswith("#"):
                continue
            verts[int(p[0])] += weight * np.array([float(p[1]), float(p[2]), float(p[3])])


def fit_sphere(pts):
    """Algebraic least-squares sphere fit: |p|^2 = 2 c.p + (r^2 - |c|^2)."""
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = np.sqrt(sol[3] + c @ c)
    resid = np.abs(np.linalg.norm(pts - c, axis=1) - r)
    return c, r, resid.max()


def main(data_dir, out_path):
    verts, uvs, faces, fuvs, fgroup, groups = parse_obj(os.path.join(data_dir, "3dobjs", "base.obj"))
    print(f"base.obj: {len(verts)} verts, {len(uvs)} uvs, {len(faces)} faces, {len(groups)} groups")
    for rel, w in TARGETS.items():
        apply_target(verts, os.path.join(data_dir, "targets", rel), w)

    # MakeHuman (dm, Y-up, facing +Z) -> Blender (m, Z-up, facing -Y)
    vb = 0.1 * np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    def group_verts(name):
        gi = groups.index(name)
        idx = sorted({v for f, g in zip(faces, fgroup) if g == gi for v in f})
        return vb[idx]

    eyes = {}
    for side, g in (("L", "helper-l-eye"), ("R", "helper-r-eye")):
        c, r, err = fit_sphere(group_verts(g))
        eyes[side] = (c, r)
        print(f"eye {side}: centre {np.round(c, 5)} m, radius {r*1000:.2f} mm, fit max residual {err*1000:.3f} mm")

    # ---- head / neck region of the skin mesh only -------------------------
    gi_body = groups.index("body")
    eye_mid_z = 0.5 * (eyes["L"][0][2] + eyes["R"][0][2])
    z_cut = eye_mid_z - 0.20          # keep 20 cm below the eye line (chin + neck stump)
    keep = [i for i, (f, g) in enumerate(zip(faces, fgroup))
            if g == gi_body and min(vb[f][:, 2]) > z_cut]
    used = sorted({v for i in keep for v in faces[i]})
    remap = {v: k for k, v in enumerate(used)}
    out_faces = np.array([[remap[v] for v in faces[i]] for i in keep], dtype=np.int32)
    out_fuv = np.array([fuvs[i] for i in keep], dtype=np.int32)
    assert out_faces.shape[1] == 4, "hm08 base mesh is all-quads"
    out_verts = vb[used].astype(np.float64)
    print(f"head region: {len(out_verts)} verts, {len(out_faces)} quads")

    # MPFB region masks (CC0), stored at 1024^2 in the original MakeHuman UV layout
    masks = {}
    try:
        from PIL import Image
        for key, fname in (("mask_lips", "mpfb_lips.jpg"), ("mask_eyelids", "mpfb_eyelids.jpg")):
            im = Image.open(os.path.join(data_dir, "textures", fname)).convert("L").resize((1024, 1024), Image.BILINEAR)
            masks[key] = np.asarray(im, dtype=np.uint8)
    except ImportError:
        print("PIL not available - masks skipped")

    np.savez_compressed(
        out_path, **masks,
        verts=out_verts, faces=out_faces, face_uv_idx=out_fuv, uvs=uvs.astype(np.float64),
        eye_center_L=eyes["L"][0], eye_radius_L=eyes["L"][1],
        eye_center_R=eyes["R"][0], eye_radius_R=eyes["R"][1],
        source=np.array("MakeHuman/MPFB2 hm08 base mesh, CC0 1.0; macro: male, young, ancestry 1/3 each; eye height1/height2 incr 0.8"),
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
