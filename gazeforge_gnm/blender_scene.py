"""Materials (UDIM textures), hair strands, lights, camera and render settings for the GNM head."""
import math
import os
import bpy
import numpy as np
from mathutils import Vector

GF_COLLECTION = "GazeForge"
SKIN_SSS_METHOD = "BURLEY"   # "RANDOM_WALK_SKIN" for stills (3x slower)


def _img(path, colorspace):
    """Load a UDIM tile set given the path of any tile with <UDIM> token."""
    name = os.path.basename(path)
    img = bpy.data.images.get(name)
    if img is None:
        first = path.replace("<UDIM>", "1001")
        img = bpy.data.images.load(first, check_existing=True)
        img.name = name
        img.source = "TILED"
        img.filepath = path
        img.reload()
    _drop_missing_tiles(img, path)
    img.colorspace_settings.name = colorspace
    return img


def _drop_missing_tiles(img, path):
    """Keep only UDIM tiles whose files exist (GNM cornea tiles 1003/1005 need no texture)."""
    if img.source != "TILED":
        return
    for t in list(img.tiles):
        if not os.path.exists(bpy.path.abspath(path.replace("<UDIM>", str(t.number)))):
            img.tiles.remove(t)


def _set_tiled(img_node, img):
    img_node.image = img
    img_node.extension = "EXTEND"
    img_node.interpolation = "Cubic"


def texture_paths(tex_dir):
    sid = os.path.basename(tex_dir).split("_")[0].replace("set", "")
    pre = os.path.join(tex_dir, f"set{sid}")
    return {"albedo": pre + "_albedo.<UDIM>.png", "height": pre + "_height.<UDIM>.png",
            "rough": pre + "_roughness.<UDIM>.png", "hairmask": pre + "_hairmask.<UDIM>.png"}


def _uv_node(nt, x=-1100):
    n = nt.nodes.new("ShaderNodeUVMap")
    n.uv_map = "UVMap_UDIM"
    n.location = (x, 0)
    return n


def build_materials(tex_dir):
    paths = texture_paths(tex_dir)
    alb = _img(paths["albedo"], "sRGB")
    hgt = bpy.data.images.load(paths["height"].replace("<UDIM>", "1001"), check_existing=True)
    hgt.colorspace_settings.name = "Non-Color"
    rgh = bpy.data.images.load(paths["rough"].replace("<UDIM>", "1001"), check_existing=True)
    rgh.colorspace_settings.name = "Non-Color"

    # ---------------- skin ----------------
    m = bpy.data.materials["GF_Skin"]
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled"); b.location = (150, 0)
    uv = _uv_node(nt)
    ta = nt.nodes.new("ShaderNodeTexImage"); ta.location = (-700, 300); _set_tiled(ta, alb); ta.name = "albedo"
    th = nt.nodes.new("ShaderNodeTexImage"); th.location = (-700, -100); th.image = hgt; th.interpolation = "Cubic"; th.name = "height"
    tr = nt.nodes.new("ShaderNodeTexImage"); tr.location = (-700, -400); tr.image = rgh; tr.name = "roughness"
    for t in (ta, th, tr):
        nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (-250, -150)
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.00012
    nt.links.new(th.outputs["Color"], bump.inputs["Height"])
    rmul = nt.nodes.new("ShaderNodeMath"); rmul.operation = "MULTIPLY"; rmul.location = (-250, -400)
    rmul.inputs[1].default_value = 0.95
    nt.links.new(tr.outputs["Color"], rmul.inputs[0])
    nt.links.new(ta.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(ta.outputs["Color"], b.inputs["Subsurface Radius"]) if False else None
    nt.links.new(rmul.outputs[0], b.inputs["Roughness"])
    hm_path = paths["hairmask"].replace("<UDIM>", "1001")
    if os.path.exists(hm_path):
        hmi = bpy.data.images.load(hm_path, check_existing=True); hmi.colorspace_settings.name = "Non-Color"
        tm = nt.nodes.new("ShaderNodeTexImage"); tm.location = (-700, -650); tm.image = hmi; tm.name = "hairmask"
        nt.links.new(uv.outputs["UV"], tm.inputs["Vector"])
        inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.location = (-250, -650)
        inv.inputs[0].default_value = 1.0
        nt.links.new(tm.outputs["Color"], inv.inputs[1])
        nt.links.new(inv.outputs[0], b.inputs["Subsurface Weight"])
        sh = nt.nodes.new("ShaderNodeMath"); sh.operation = "MULTIPLY"; sh.location = (-250, -800)
        sh.inputs[1].default_value = 0.12
        nt.links.new(inv.outputs[0], sh.inputs[0])
        nt.links.new(sh.outputs[0], b.inputs["Sheen Weight"])
        sp = nt.nodes.new("ShaderNodeMath"); sp.operation = "MULTIPLY"; sp.location = (-250, -950)
        sp.inputs[1].default_value = 0.5
        nt.links.new(inv.outputs[0], sp.inputs[0])
        nt.links.new(sp.outputs[0], b.inputs["Specular IOR Level"])
        rmx = nt.nodes.new("ShaderNodeMix"); rmx.data_type = "FLOAT"; rmx.location = (-50, -400)
        nt.links.new(tm.outputs["Color"], rmx.inputs["Factor"])
        nt.links.new(rmul.outputs[0], rmx.inputs["A"])
        rmx.inputs["B"].default_value = 0.75
        nt.links.new(rmx.outputs["Result"], b.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.subsurface_method = SKIN_SSS_METHOD
    b.inputs["Subsurface Weight"].default_value = 1.0
    b.inputs["Subsurface Radius"].default_value = (3.67, 1.37, 0.68)
    b.inputs["Subsurface Scale"].default_value = 0.0009
    for k, v in (("Subsurface IOR", 1.40), ("Subsurface Anisotropy", 0.8)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["IOR"].default_value = 1.44
    b.inputs["Sheen Weight"].default_value = 0.12
    b.inputs["Sheen Roughness"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.06        # thin sebum layer: sharper highlights
    b.inputs["Coat Roughness"].default_value = 0.35
    # neck / upper chest a touch darker than the face (it sits in the chin's shadow in real portraits)
    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-1100, 500)
    sxyz = nt.nodes.new("ShaderNodeSeparateXYZ"); sxyz.location = (-900, 500)
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.location = (-700, 600)
    mr.inputs["From Min"].default_value = 0.11; mr.inputs["From Max"].default_value = 0.18
    mr.inputs["To Min"].default_value = 0.86; mr.inputs["To Max"].default_value = 1.0
    nt.links.new(tc.outputs["Object"], sxyz.inputs[0]); nt.links.new(sxyz.outputs["Y"], mr.inputs["Value"])
    mulc = nt.nodes.new("ShaderNodeMix"); mulc.data_type = "RGBA"; mulc.blend_type = "MULTIPLY"; mulc.location = (-200, 400)
    mulc.inputs["Factor"].default_value = 1.0
    nt.links.new(ta.outputs["Color"], mulc.inputs["A"])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ"); cmb.location = (-450, 600)
    for k in ("X", "Y", "Z"):
        nt.links.new(mr.outputs["Result"], cmb.inputs[k])
    nt.links.new(cmb.outputs[0], mulc.inputs["B"])
    nt.links.new(mulc.outputs["Result"], b.inputs["Base Color"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])

    # ---------------- eye interior (iris / sclera / pupil), per eye ----------------
    for side in ("L", "R"):
        m = bpy.data.materials[f"GF_EyeInterior_{side}"]
        m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled"); b.location = (150, 0)
        uv = _uv_node(nt)
        ta = nt.nodes.new("ShaderNodeTexImage"); ta.location = (-700, 200); _set_tiled(ta, alb); ta.name = "albedo"
        nt.links.new(uv.outputs["UV"], ta.inputs["Vector"])
        nt.links.new(ta.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.45
        b.inputs["Subsurface Weight"].default_value = 0.12
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.5)
        b.inputs["Subsurface Scale"].default_value = 0.0004
        b.inputs["Specular IOR Level"].default_value = 0.0   # under the aqueous/cornea: no dry specular
        # micro bump on the iris from the albedo luminance (fibre relief)
        bump = nt.nodes.new("ShaderNodeBump"); bump.location = (-200, -200)
        bump.inputs["Distance"].default_value = 0.00004
        nt.links.new(ta.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])

    # ---------------- cornea: clear refracting shell, transparent to shadow rays ----------------
    m = bpy.data.materials["GF_Cornea"]
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    g = nt.nodes.new("ShaderNodeBsdfPrincipled"); g.location = (0, 150)
    g.inputs["Base Color"].default_value = (1, 1, 1, 1)
    g.inputs["Transmission Weight"].default_value = 1.0
    g.inputs["IOR"].default_value = 1.376
    g.inputs["Roughness"].default_value = 0.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.location = (0, -150)
    lp = nt.nodes.new("ShaderNodeLightPath"); lp.location = (-300, 300)
    mx = nt.nodes.new("ShaderNodeMixShader"); mx.location = (350, 0)
    nt.links.new(lp.outputs["Is Shadow Ray"], mx.inputs[0])
    nt.links.new(g.outputs["BSDF"], mx.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mx.inputs[2])
    nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    m.blend_method = "HASHED" if hasattr(m, "blend_method") else None

    # ---------------- teeth / tongue ----------------
    for name, rough, sss in (("GF_Teeth", 0.25, 0.3), ("GF_Tongue", 0.35, 0.6)):
        m = bpy.data.materials[name]
        m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        uv = _uv_node(nt)
        ta = nt.nodes.new("ShaderNodeTexImage"); ta.location = (-700, 200); _set_tiled(ta, alb)
        nt.links.new(uv.outputs["UV"], ta.inputs["Vector"])
        nt.links.new(ta.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Scale"].default_value = 0.001
        nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return alb


def swap_textures(tex_dir):
    """Point all GF materials at another texture set (same UDIM layout)."""
    paths = texture_paths(tex_dir)
    for img in list(bpy.data.images):
        if img.users == 0:
            continue
    alb = _img(paths["albedo"], "sRGB")
    hgt = bpy.data.images.load(paths["height"].replace("<UDIM>", "1001"), check_existing=True)
    hgt.colorspace_settings.name = "Non-Color"
    rgh = bpy.data.images.load(paths["rough"].replace("<UDIM>", "1001"), check_existing=True)
    rgh.colorspace_settings.name = "Non-Color"
    for m in bpy.data.materials:
        if not m.name.startswith("GF_") or not m.use_nodes:
            continue
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE":
                if n.name == "height":
                    n.image = hgt
                elif n.name == "roughness":
                    n.image = rgh
                elif n.name == "hairmask":
                    hm = bpy.data.images.load(paths["hairmask"].replace("<UDIM>", "1001"), check_existing=True)
                    hm.colorspace_settings.name = "Non-Color"
                    n.image = hm
                else:
                    n.image = alb


# ---------------------------------------------------------------------------
def strands_mesh(name, strands, root_r, tip_r, sides=3):
    """Tapered tube mesh for a list of (n,3) polylines given in GNM coordinates."""
    verts, faces = [], []
    for s in strands:
        s = np.asarray(s, float)
        n = len(s)
        t = np.gradient(s, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        a = np.cross(t[0], [0.3, 0.9, 0.2]); a /= np.linalg.norm(a)
        base = len(verts)
        for i in range(n):
            u = a - t[i] * (a @ t[i]); u /= np.linalg.norm(u)
            v = np.cross(t[i], u)
            r = root_r + (tip_r - root_r) * (i / (n - 1)) ** 0.8
            for k in range(sides):
                ang = 2 * math.pi * k / sides
                verts.append(s[i] + r * (math.cos(ang) * u + math.sin(ang) * v))
        for i in range(n - 1):
            for k in range(sides):
                a0 = base + i * sides + k; a1 = base + i * sides + (k + 1) % sides
                faces.append((a0, a1, a1 + sides, a0 + sides))
        tip = len(verts); verts.append(s[-1] + t[-1] * tip_r)
        for k in range(sides):
            faces.append((base + (n - 1) * sides + k, base + (n - 1) * sides + (k + 1) % sides, tip))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    return me


def hair_material(name, srgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    c = np.array(srgb) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    b.inputs["Base Color"].default_value = (*lin, 1)
    b.inputs["Roughness"].default_value = 0.38
    b.inputs["Coat Weight"].default_value = 0.2
    b.inputs["Subsurface Weight"].default_value = 0.0
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return m


def add_hair(head, arm, tex_dir, hair_srgb):
    """Eyebrow + eyelash strand meshes bound to the head surface (Surface Deform), so they follow
    blinks, brow motion and identity changes."""
    d = np.load(os.path.join(tex_dir, "hair_strands.npz"), allow_pickle=True)
    out = []
    for name, strands, r0, r1, col in (("GF_Brows", list(d["brows"])[::2], 0.000042, 0.00001, hair_srgb),
                                       ("GF_Lashes", list(d["lashes"]), 0.00008, 0.00002, (10, 8, 7))):
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        me = strands_mesh(name, strands, r0, r1)
        me.materials.append(hair_material(name + "_Mat", col))
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.parent = arm
        ob.matrix_parent_inverse.identity()
        sd = ob.modifiers.new("SurfaceDeform", "SURFACE_DEFORM")
        sd.target = head
        sd.falloff = 4.0
        out.append(ob)
    return out


def _srgb_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _set(obj, **kw):
    for k, v in kw.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def add_scalp_hair(head, spec, arm=None, n_strands=60000):
    """Short scalp hair: explicit strands (gazeforge_gnm.scalp_hair) as a hair-curves object,
    parented to the head bone so it follows head motion; regenerated per identity."""
    from .scalp_hair import grow
    if "gf_scalp" not in head.vertex_groups:
        return None
    arm = arm or head.parent
    me = head.data
    nv = len(me.vertices)
    # bind-pose vertices for the current identity (identity key only, no expressions)
    kb = me.shape_keys.key_blocks
    V = np.empty(nv * 3, np.float32); kb["Basis"].data.foreach_get("co", V); V = V.reshape(-1, 3).astype(np.float64)
    for k in kb:
        if k.name.startswith("ID_") and k.value > 0:
            D = np.empty(nv * 3, np.float32); k.data.foreach_get("co", D)
            V += (D.reshape(-1, 3) - _basis(me)) * k.value
    dens = np.zeros(nv)
    gi = head.vertex_groups["gf_scalp"].index
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi:
                dens[v.index] = g.weight
    tris = []
    for poly in me.polygons:
        vs = list(poly.vertices)
        for j in range(1, len(vs) - 1):
            tris.append((vs[0], vs[j], vs[j + 1]))
    tris = np.array(tris)
    female = spec["identity"].startswith("female")
    strands = grow(V, tris, dens, n_strands=n_strands, length=0.030 if female else 0.017,
                   seed=int(spec.get("seed", 1)), female=female)
    old = bpy.data.objects.get("GF_ScalpHair")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    cv = bpy.data.hair_curves.new("GF_ScalpHair")
    n, m, _ = strands.shape
    cv.add_curves([m] * n)
    cv.position_data.foreach_set("vector", strands.astype(np.float32).ravel())
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    r = np.linspace(0.000045, 0.000012, m, dtype=np.float32)
    rad.data.foreach_set("value", np.tile(r, n))
    ob = bpy.data.objects.new("GF_ScalpHair", cv)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.get("GF_ScalpHair") or bpy.data.materials.new("GF_ScalpHair")
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (300, 0)
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled"); hb.name = "hair"
    hb.parametrization = "MELANIN"
    nt.links.new(hb.outputs[0], out.inputs["Surface"])
    cv.materials.append(mat)
    # rigid attachment to the head bone, rest pose
    pose = {pb.name: pb.rotation_quaternion.copy() for pb in arm.pose.bones}
    for pb in arm.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = "head"
    bpy.context.view_layer.update()
    ob.matrix_world = arm.matrix_world.copy()
    for pb in arm.pose.bones:
        pb.rotation_quaternion = pose[pb.name]
    bpy.context.view_layer.update()
    style_scalp_hair(head, spec)
    return ob


def _basis(me):
    nv = len(me.vertices)
    B = np.empty(nv * 3, np.float32)
    me.shape_keys.key_blocks["Basis"].data.foreach_get("co", B)
    return B.reshape(-1, 3)


def style_scalp_hair(head, spec):
    """Per-identity hair colour (melanin from the set's hair colour) and length."""
    if "GF_ScalpHair" not in bpy.data.materials:
        return
    lin = _srgb_lin(spec["hair_srgb"])
    lum = float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
    rg = float(lin[0] / max(lin[1], 1e-4))
    if lum < 0.02:
        melanin, red = 0.95, 0.15          # black / very dark brown
    elif rg > 2.2:
        melanin, red = 0.55, 0.80          # auburn
    else:
        melanin, red = float(np.clip(1.0 - lum * 4.0, 0.6, 0.85)), 0.38   # brown
    hb = bpy.data.materials["GF_ScalpHair"].node_tree.nodes["hair"]
    hb.inputs["Melanin"].default_value = melanin
    hb.inputs["Melanin Redness"].default_value = red
    hb.inputs["Roughness"].default_value = 0.40
    hb.inputs["Radial Roughness"].default_value = 0.45
    hb.inputs["Coat"].default_value = 0.08
    hb.inputs["Random Color"].default_value = 0.08
    hb.inputs["Random Roughness"].default_value = 0.15


def bind_hair(hair_objs, head):
    """Bind with the head at its rest shape (all shape keys 0, rest pose)."""
    kb = head.data.shape_keys.key_blocks
    saved = {k.name: k.value for k in kb}
    for k in kb:
        k.value = 0.0
    arm = head.parent
    pose = {}
    if arm is not None and arm.type == "ARMATURE":   # bind in the rest pose
        pose = {pb.name: pb.rotation_quaternion.copy() for pb in arm.pose.bones}
        for pb in arm.pose.bones:
            pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    for ob in hair_objs:
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.object.surfacedeform_bind(modifier="SurfaceDeform")
    for k in kb:
        k.value = saved[k.name]
    for n, q in pose.items():
        arm.pose.bones[n].rotation_quaternion = q
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
def enable_gpu(scene):
    """Use an NVIDIA (OptiX/CUDA), AMD (HIP), Intel (oneAPI) or Apple (Metal) GPU if present."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def setup_render(scene, res=(1280, 720), samples=64, engine="CYCLES"):
    scene.render.engine = engine
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 6
    scene.render.use_persistent_data = True
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 1.0
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Base Contrast"
    except TypeError:
        pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 24


def setup_lights_camera(scene, eye_height, framing="face"):
    """Soft studio portrait lighting (key softbox, fill, rim, hair light) + 85 mm portrait camera.
    Blender space: the face looks along -Y; eyes at z = eye_height."""
    for o in list(scene.objects):
        if o.type in ("LIGHT", "CAMERA"):
            bpy.data.objects.remove(o, do_unlink=True)

    def area(name, loc, size, energy, color=(1, 1, 1), size_y=None):
        L = bpy.data.lights.new(name, "AREA")
        L.energy = energy
        L.color = color
        if size_y:
            L.shape = "RECTANGLE"; L.size = size; L.size_y = size_y
        else:
            L.size = size
        o = bpy.data.objects.new(name, L)
        scene.collection.objects.link(o)
        o.location = loc
        target = Vector((0, 0.0, eye_height - 0.02))
        o.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        return o
    area("Key", (-0.55, -0.85, eye_height + 0.45), 0.9, 28, (1.0, 0.97, 0.93), size_y=1.2)
    area("Fill", (0.85, -0.7, eye_height + 0.05), 1.4, 9, (0.93, 0.96, 1.0))
    area("Rim_L", (-0.45, 0.6, eye_height + 0.25), 0.4, 14, (1.0, 0.95, 0.9))
    area("Rim_R", (0.5, 0.55, eye_height + 0.2), 0.4, 11, (0.95, 0.97, 1.0))
    area("Top", (0.0, 0.05, eye_height + 0.8), 0.8, 8)
    w = scene.world or bpy.data.worlds.new("World")
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.045, 0.048, 0.055, 1)
    bg.inputs["Strength"].default_value = 1.0
    cd = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    set_framing(cam, eye_height, framing)
    return cam


FRAMINGS = {
    # name: (lens mm, distance m, z offset from eye height, sensor)
    "face": (85.0, 0.74, -0.035),
    "portrait": (85.0, 0.66, -0.032),
    "eyes": (135.0, 0.50, -0.003),
}


def set_framing(cam, eye_height, framing):
    lens, dist, dz = FRAMINGS[framing]
    cam.data.lens = lens
    cam.data.sensor_width = 36
    cam.data.dof.use_dof = framing == "eyes"
    cam.data.dof.focus_distance = dist - 0.105
    cam.data.dof.aperture_fstop = 16.0
    cam.location = (0.0, -dist, eye_height + dz)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def apply_texture_set(tex_dir, head=None, eyes=None, arm=None):
    """Switch an existing GazeForge scene to another texture set: textures, GNM identity
    (shape keys + joints), eyebrow/eyelash strands and scalp hair. Needs no GNM model file."""
    import json
    from .blender_head import set_identity
    head = head or bpy.data.objects["GNM_Head"]
    eyes = eyes or bpy.data.objects["GNM_Eyes"]
    arm = arm or bpy.data.objects["GNM_Head_Rig"]
    spec = json.load(open(os.path.join(tex_dir, "texture_set.json")))
    swap_textures(tex_dir)
    set_identity((head, eyes), arm, spec["identity"])
    hair = add_hair(head, arm, tex_dir, spec["hair_srgb"])
    bind_hair(hair, head)
    add_scalp_hair(head, spec, arm)
    bpy.context.scene["gf_texture_set"] = os.path.basename(tex_dir)
    return spec
