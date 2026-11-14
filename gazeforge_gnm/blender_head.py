"""Build the GNM head in Blender (bpy): mesh with GNM quads + UDIM UVs, armature
(neck, head, eye_L, eye_R) with GNM skinning weights, identity and expression
shape keys computed from the GNM bases. Coordinates: the armature object rotates
GNM space (Y up, face looking +Z) into Blender space (Z up, face looking -Y).
"""
import math
import os
import bpy
import numpy as np
from mathutils import Matrix, Vector

from .gnm_model import GNMHead

TILE_MATERIAL = {1001: "GF_Skin", 1002: "GF_EyeInterior_L", 1003: "GF_Cornea",
                 1004: "GF_EyeInterior_R", 1005: "GF_Cornea", 1006: "GF_Teeth", 1007: "GF_Tongue"}
MATERIAL_ORDER = ["GF_Skin", "GF_EyeInterior_L", "GF_EyeInterior_R", "GF_Cornea", "GF_Teeth", "GF_Tongue"]
BONES = {"neck": "neck", "head": "head", "left_eye": "eye_L", "right_eye": "eye_R"}
GNM_TO_BLENDER = Matrix.Rotation(math.radians(90.0), 4, "X")


def build_head(gnm_path, controls_path, identity_name, name="GNM_Head"):
    gnm = GNMHead(gnm_path)
    c = np.load(controls_path)
    id_names = [str(s) for s in c["identity_names"]]
    ids = c["identities"]
    ctrl_names = [str(s) for s in c["control_names"]]
    ctrls = c["controls"]

    V = gnm.template
    eye_v = (gnm.group("left_eye") > 0.5) | (gnm.group("right_eye") > 0.5)
    parts = {name: ~eye_v, name.replace("Head", "Eyes"): eye_v}
    objs = {}
    for pname, vmask in parts.items():
        objs[pname] = _make_part(gnm, pname, vmask, id_names, ids, ctrl_names, ctrls,
                                 with_controls=(pname == name), assets_dir=os.path.dirname(controls_path))
    obj, eyes = objs[name], objs[name.replace("Head", "Eyes")]

    # ---------------- armature ----------------
    arm_data = bpy.data.armatures.new(name + "_Rig")
    arm = bpy.data.objects.new(name + "_Rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    arm.matrix_world = GNM_TO_BLENDER
    for o in (obj, eyes):
        o.parent = arm
        mod = o.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
    obj["gnm_identity_names"] = id_names
    arm["gnm_joints_template"] = gnm.joints0.ravel().tolist()
    arm["gnm_joint_id_basis_shape"] = list(gnm.joint_id_basis.shape)
    np_ids = {n: np.tensordot(e, gnm.joint_id_basis, axes=(0, 0)) for n, e in zip(id_names, ids)}
    arm["gnm_joint_offsets"] = {n: v.ravel().tolist() for n, v in np_ids.items()}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for j, jn in enumerate(gnm.joint_names):
        b = arm_data.edit_bones.new(BONES[jn])
        b.head = Vector(gnm.joints0[j])
        b.tail = b.head + Vector((0, 0.02 if jn in ("neck", "head") else 0.012, 0))
        b.roll = 0.0
        p = gnm.parents[j]
        if p >= 0:
            b.parent = arm_data.edit_bones[BONES[gnm.joint_names[p]]]
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    set_identity((obj, eyes), arm, identity_name)
    return obj, eyes, arm


def set_identity(objs, arm, identity_name):
    """Switch identity: identity shape key on, bones moved to the GNM joint positions."""
    for obj in objs:
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name.startswith("ID_"):
                kb.value = 1.0 if kb.name == "ID_" + identity_name else 0.0
        obj["gnm_identity"] = identity_name
    J0 = np.array(arm["gnm_joints_template"]).reshape(-1, 3)
    off = np.array(arm["gnm_joint_offsets"][identity_name]).reshape(-1, 3)
    J = J0 + off
    prev = bpy.context.view_layer.objects.active
    mode_obj = bpy.context.mode
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for j, bn in enumerate(["neck", "head", "eye_L", "eye_R"]):
        b = arm.data.edit_bones[bn]
        L = (b.tail - b.head).length
        b.head = Vector(J[j])
        b.tail = b.head + Vector((0, L, 0))
        b.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = prev
    arm["gnm_identity"] = identity_name


def _make_part(gnm, pname, vmask, id_names, ids, ctrl_names, ctrls, with_controls, assets_dir=""):
    V = gnm.template
    vidx = np.nonzero(vmask)[0]
    remap = -np.ones(len(V), np.int64)
    remap[vidx] = np.arange(len(vidx))
    qsel = np.nonzero(np.all(vmask[gnm.quads], axis=1))[0]
    quads = remap[gnm.quads[qsel]]
    me = bpy.data.meshes.new(pname)
    me.from_pydata(V[vidx].tolist(), [], quads.tolist())
    me.update()
    uv = me.uv_layers.new(name="UVMap_UDIM")
    uvs = gnm.uvs_udim(gnm.quad_uvs, gnm.quad_tiles)[qsel].reshape(-1, 2)
    uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    uv2 = me.uv_layers.new(name="UVMap_GNM_per_region")
    uv2.data.foreach_set("uv", gnm.quad_uvs[qsel].reshape(-1, 2).astype(np.float32).ravel())
    me.uv_layers.active = uv
    uv.active_render = True
    tiles = gnm.quad_tiles[qsel]
    used = [m for m in MATERIAL_ORDER if any(TILE_MATERIAL[t] == m for t in set(tiles.tolist()))]
    for m in used:
        mat = bpy.data.materials.get(m) or bpy.data.materials.new(m)
        me.materials.append(mat)
    mi = np.array([used.index(TILE_MATERIAL[t]) for t in tiles], np.int32)
    me.polygons.foreach_set("material_index", mi)
    me.polygons.foreach_set("use_smooth", np.ones(len(quads), bool))
    obj = bpy.data.objects.new(pname, me)
    bpy.context.scene.collection.objects.link(obj)
    obj["gnm_vertex_indices_count"] = int(len(vidx))
    for j, jn in enumerate(gnm.joint_names):
        vg = obj.vertex_groups.new(name=BONES[jn])
        w = gnm.weights[j][vidx]
        for i in np.nonzero(w > 1e-6)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    if with_controls:
        for gname in ["eye_sockets", "left_brow_region", "right_brow_region", "middle_brow_region",
                      "forehead_region", "left_orbital_region", "right_orbital_region"]:
            vg = obj.vertex_groups.new(name="gnm_" + gname)
            w = gnm.group(gname)[vidx]
            for i in np.nonzero(w > 1e-6)[0]:
                vg.add([int(i)], float(w[i]), "REPLACE")
        sp = os.path.join(assets_dir, "gnm_scalp_density.npy")
        if os.path.exists(sp):   # scalp-hair density (same hairline field as the textures)
            w = np.load(sp)[vidx]
            vg = obj.vertex_groups.new(name="gf_scalp")
            for i in np.nonzero(w > 1e-4)[0]:
                vg.add([int(i)], float(w[i]), "REPLACE")
    obj.shape_key_add(name="Basis", from_mix=False)
    Vp = V[vidx]

    def add_key(kname, delta, smin=0.0, smax=1.0):
        k = obj.shape_key_add(name=kname, from_mix=False)
        k.data.foreach_set("co", (Vp + delta[vidx]).astype(np.float32).ravel())
        k.slider_min, k.slider_max = smin, smax
        k.value = 0.0

    for n, e in zip(id_names, ids):
        add_key("ID_" + n, np.tensordot(e, gnm.id_basis, axes=(0, 0)))
    for n, e in zip(ctrl_names, ctrls):
        if with_controls != (n == "pupil_dilate"):
            add_key(n, np.tensordot(e, gnm.expr_basis, axes=(0, 0)), -1.0, 1.5)
    return obj
