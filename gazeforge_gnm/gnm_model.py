"""Minimal, dependency-free (NumPy only) evaluator for the GNM v3 head model.

Re-implements the GNM forward pass (google/GNM, Apache-2.0): bind-pose vertices
= template + identity basis + expression basis; joints = template joints +
joint identity basis; posing = linear blend skinning about the bind-pose joints
with axis-angle joint rotations, parented neck -> head -> eyes.
Checked against the official gnm_numpy implementation (tools/check_gnm_model.py).
"""
import numpy as np

UDIM_TILES = {"skin": 1001, "left_eye_interior": 1002, "left_eye_exterior": 1003,
              "right_eye_interior": 1004, "right_eye_exterior": 1005,
              "teeth_and_gums": 1006, "tongue": 1007}
_TILE_REGIONS = [  # (tile, union groups, intersect group)
    (1001, ("skin",), None),
    (1002, ("left_eye",), "eye_interiors"),
    (1003, ("left_eye",), "eye_exteriors"),
    (1004, ("right_eye",), "eye_interiors"),
    (1005, ("right_eye",), "eye_exteriors"),
    (1006, ("upper_teeth_and_gums", "lower_teeth_and_gums"), None),
    (1007, ("tongue",), None),
]


def axis_angle_to_matrix(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class GNMHead:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.d = d
        self.template = d["template_vertex_positions"].astype(np.float64)
        self.id_basis = d["vertex_identity_basis"]
        self.expr_basis = d["expression_basis"]
        self.joints0 = d["template_joint_positions"].astype(np.float64)
        self.joint_id_basis = d["joint_identity_basis"].astype(np.float64)
        self.joint_names = [str(s) for s in d["joint_names"]]
        self.parents = d["joint_parent_indices"].tolist()
        self.weights = d["skinning_weights"].astype(np.float64)
        self.quads = d["quads"]
        self.quad_uvs = d["quad_uvs"].astype(np.float64)
        self.triangles = d["triangles"]
        self.triangle_uvs = d["triangle_uvs"].astype(np.float64)
        self.group_names = [str(s) for s in d["vertex_group_names"]]
        self.groups = d["vertex_groups"]
        self.expression_names = [str(s) for s in d["expression_names"]]
        self.quad_tiles = self._tiles(self.quads)
        self.triangle_tiles = self._tiles(self.triangles)

    # ---- groups / UDIM ---------------------------------------------------
    def group(self, name):
        return self.groups[self.group_names.index(name)]

    def _tiles(self, faces):
        tiles = np.zeros(len(faces), np.int32)
        for tile, union, inter in _TILE_REGIONS:
            m = np.zeros(self.template.shape[0], bool)
            for g in union:
                m |= self.group(g) > 0.5
            if inter:
                m &= self.group(inter) > 0.5
            sel = np.all(m[faces], axis=1)
            assert not np.any(tiles[sel]), "overlapping UDIM regions"
            tiles[sel] = tile
        assert np.all(tiles), "face without UDIM tile"
        return tiles

    def uvs_udim(self, uvs, tiles):
        off = np.stack([(tiles - 1001) % 10, (tiles - 1001) // 10], -1).astype(np.float64)
        return uvs + off[:, None, :]

    # ---- forward model ---------------------------------------------------
    def bind_vertices(self, identity=None, expression=None):
        v = self.template.copy()
        if identity is not None:
            v += np.tensordot(identity, self.id_basis, axes=(0, 0))
        if expression is not None:
            v += np.tensordot(expression, self.expr_basis, axes=(0, 0))
        return v

    def bind_joints(self, identity=None):
        j = self.joints0.copy()
        if identity is not None:
            j += np.tensordot(identity, self.joint_id_basis, axes=(0, 0))
        return j

    def __call__(self, identity=None, expression=None, rotations=None, translation=None):
        v = self.bind_vertices(identity, expression)
        j = self.bind_joints(identity)
        nj = len(self.parents)
        rotations = np.zeros((nj, 3)) if rotations is None else np.asarray(rotations, float)
        translation = np.zeros(3) if translation is None else np.asarray(translation, float)
        G = []  # world transforms of joints (4x4), relative to bind
        for i in range(nj):
            T = np.eye(4)
            T[:3, :3] = axis_angle_to_matrix(rotations[i])
            p = self.parents[i]
            T[:3, 3] = j[i] - (j[p] if p >= 0 else 0)
            if p < 0:
                T[:3, 3] += translation
            G.append(T if p < 0 else G[p] @ T)
        A = []
        for i in range(nj):  # remove bind pose
            Bi = np.eye(4)
            Bi[:3, 3] = -j[i]
            A.append(G[i] @ Bi)
        A = np.stack(A)
        M = np.einsum("jv,jab->vab", self.weights, A)
        vh = np.concatenate([v, np.ones((len(v), 1))], 1)
        return np.einsum("vab,vb->va", M, vh)[:, :3]
