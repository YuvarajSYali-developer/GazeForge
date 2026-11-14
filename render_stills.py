"""Render one photoreal portrait per texture set (renders/), each a different GNM identity + texture.

  blender -b --python render_stills.py -- [--sets 01,02] [--samples 64] [--res 1080x1350] [--gpu]
  (uses GazeForge_GNM.blend; pass --gnm gnm_head.npz to rebuild the scene from the GNM model instead)
Stills use the random-walk (skin) subsurface model; videos use Christensen-Burley for speed.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import build_scene  # noqa: E402
from gazeforge_gnm import blender_scene as bs  # noqa: E402
from gazeforge_gnm.rig import Rig  # noqa: E402

NEUTRAL = dict(gaze_yaw=0, gaze_pitch=0, head_yaw=0, head_pitch=0, head_roll=0, blink=0, brow_raise=0,
               brow_lower=0, lid_raise=0, squint=0, pupil=0, smile=0, vergence_m=0.8)
POSES = {  # a natural, different moment for each identity
    "01": dict(gaze_yaw=-4, head_yaw=8, head_roll=-2, smile=0.25, squint=0.15),
    "02": dict(gaze_yaw=2, head_yaw=-10, head_pitch=-1, brow_raise=0.15),
    "03": dict(gaze_yaw=-2, head_yaw=12, smile=0.38, squint=0.3, head_roll=3),
    "04": dict(gaze_yaw=1, head_yaw=-6, head_pitch=2, brow_lower=0.1),
    "05": dict(gaze_yaw=-3, head_yaw=9, smile=0.42, squint=0.35, head_roll=-2),
    "06": dict(gaze_yaw=14, gaze_pitch=-2, head_yaw=-12, brow_raise=0.2),
    "07": dict(gaze_yaw=0, head_yaw=7, head_pitch=-2, smile=0.2, pupil=0.3),
    "08": dict(gaze_yaw=-12, gaze_pitch=4, head_yaw=10, brow_raise=0.1),
}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnm", default=None, help="gnm_head.npz: rebuild the scene per set (else use --blend)")
    ap.add_argument("--blend", default=os.path.join(HERE, "GazeForge_GNM.blend"))
    ap.add_argument("--controls", default=os.path.join(HERE, "assets", "gnm_controls.npz"))
    ap.add_argument("--textures", default=os.path.join(HERE, "textures"))
    ap.add_argument("--sets", default="")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1080x1350")
    ap.add_argument("--framing", default="portrait")
    ap.add_argument("--out", default=os.path.join(HERE, "renders"))
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    w, h = (int(v) for v in a.res.split("x"))
    sets = sorted(os.path.basename(p)[3:5] for p in glob.glob(os.path.join(a.textures, "set*_*")))
    if a.sets:
        sets = [s for s in sets if s in a.sets.split(",")]
    bs.SKIN_SSS_METHOD = "RANDOM_WALK_SKIN"
    for sid in sets:
        t = time.time()
        tex_dir = glob.glob(os.path.join(os.path.abspath(a.textures), f"set{sid}_*"))[0]
        if a.gnm:
            head, eyes, arm = build_scene.build(a.gnm, a.controls, os.path.abspath(a.textures), sid, framing=a.framing,
                                                res=(w, h), samples=a.samples)
        else:
            if not bpy.data.objects.get("GNM_Head"):
                bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
            head, eyes, arm = (bpy.data.objects[n] for n in ("GNM_Head", "GNM_Eyes", "GNM_Head_Rig"))
            bs.apply_texture_set(tex_dir, head, eyes, arm)
            sc = bpy.context.scene
            bs.setup_render(sc, res=(w, h), samples=a.samples)
            eye_h = (arm.matrix_world @ arm.pose.bones["eye_L"].bone.head_local).z
            bs.set_framing(sc.camera, eye_h, a.framing)
            for m in bpy.data.materials:   # stills: random-walk skin SSS
                if m.name == "GF_Skin":
                    for nd in m.node_tree.nodes:
                        if nd.type == "BSDF_PRINCIPLED":
                            nd.subsurface_method = "RANDOM_WALK_SKIN"
        sc = bpy.context.scene
        sc.cycles.adaptive_threshold = 0.01
        if a.gpu:
            print("[stills] device:", bs.enable_gpu(sc), flush=True)
        f = dict(NEUTRAL); f.update(POSES.get(sid, {}))
        Rig(head, eyes, arm).apply(f)
        name = os.path.basename(glob.glob(os.path.join(a.textures, f"set{sid}_*"))[0])
        sc.render.filepath = os.path.join(a.out, f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[stills] {name}: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
