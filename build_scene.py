"""Build GazeForge_GNM.blend from the GNM v3 head model.

  blender -b --python build_scene.py -- --gnm gnm_head.npz [--set 01] [--out GazeForge_GNM.blend]
  (or: python build_scene.py ... with the `bpy` module installed)

Creates: GNM head + eyes (GNM topology, UDIM UVs, identity + expression shape keys), armature
(neck/head/eye_L/eye_R with GNM skinning weights), UDIM-textured materials, eyebrow/eyelash strands
bound to the surface, studio lights, portrait camera, Cycles settings.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from gazeforge_gnm.blender_head import build_head  # noqa: E402
from gazeforge_gnm import blender_scene as bs  # noqa: E402
import json  # noqa: E402


def find_set(tex_root, sid):
    m = glob.glob(os.path.join(tex_root, f"set{sid}_*"))
    if not m:
        raise SystemExit(f"texture set {sid} not found in {tex_root}")
    return m[0]


def build(gnm, controls, tex_root, sid, framing="face", res=(1280, 720), samples=64):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    tex_dir = find_set(tex_root, sid)
    spec = json.load(open(os.path.join(tex_dir, "texture_set.json")))
    head, eyes, arm = build_head(gnm, controls, spec["identity"])
    for o in (head, eyes):
        sub = o.modifiers.new("Subdivision", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 1
        sub.uv_smooth = "PRESERVE_BOUNDARIES"
    bs.build_materials(tex_dir)
    hair = bs.add_hair(head, arm, tex_dir, spec["hair_srgb"])
    bs.bind_hair(hair, head)
    bs.add_scalp_hair(head, spec, arm)
    sc = bpy.context.scene
    bs.setup_render(sc, res=res, samples=samples)
    eye_h = arm.matrix_world @ arm.pose.bones["eye_L"].bone.head_local
    bs.setup_lights_camera(sc, eye_h.z, framing)
    sc["gf_texture_set"] = os.path.basename(tex_dir)
    sc["gf_texture_root"] = tex_root
    return head, eyes, arm


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnm", required=True, help="path to gnm_head.npz (GNM v3, google/gnm-v3 on Hugging Face)")
    ap.add_argument("--controls", default=os.path.join(HERE, "assets", "gnm_controls.npz"))
    ap.add_argument("--textures", default=os.path.join(HERE, "textures"))
    ap.add_argument("--set", default="01")
    ap.add_argument("--out", default=os.path.join(HERE, "GazeForge_GNM.blend"))
    a = ap.parse_args(argv)
    build(a.gnm, a.controls, os.path.abspath(a.textures), a.set)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    print("saved", a.out)
