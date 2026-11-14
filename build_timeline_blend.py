"""Build GazeForge_GNM_Timeline.blend: the GNM head with an expression animated on the Blender Timeline.

Open it in Blender and press Space (or scrub the Timeline): ~7 s at 24 fps, sparse Bezier keyframes on
  * eye bones (iris / gaze direction, both eyes, vergent) and head/neck bones,
  * eyelid shape keys (blink, upper lid down/raise, lower lid raise),
  * eyebrow shape keys (raise, lower), pupil dilation (iris), smile.
Timeline markers name each phase. Usage:
  blender -b --python build_timeline_blend.py -- --gnm gnm_head.npz [--set 03]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import build_scene  # noqa: E402
from gazeforge_gnm.rig import Rig, action_fcurves  # noqa: E402

FPS = 24
NEUTRAL = dict(gaze_yaw=0, gaze_pitch=0, head_yaw=0, head_pitch=0, head_roll=0, blink=0, brow_raise=0,
               brow_lower=0, lid_raise=0, squint=0, pupil=0, smile=0, vergence_m=0.8)
# (frame, marker or None, pose overrides). Fast eye movements are 2-3 frames apart (saccades),
# blinks close in 2 frames and open in 4 (75/150 ms), expressions ease over 6-12 frames.
KEYS = [
    (1, "Neutral: looking at the viewer", {}),
    (12, None, {}),
    (14, "Blink", {"blink": 1.0}),
    (15, None, {"blink": 1.0}),
    (19, None, {}),
    (30, None, {}),
    (32, "Curious glance left", {"gaze_yaw": -20, "gaze_pitch": 2, "brow_raise": 0.1}),
    (40, None, {"gaze_yaw": -20, "gaze_pitch": 2, "head_yaw": -4, "brow_raise": 0.25}),
    (52, None, {"gaze_yaw": -20, "gaze_pitch": 2, "head_yaw": -4, "brow_raise": 0.25}),
    (54, "Glance right", {"gaze_yaw": 18, "gaze_pitch": 0, "head_yaw": -2, "brow_raise": 0.2}),
    (64, None, {"gaze_yaw": 18, "gaze_pitch": 0, "head_yaw": 3, "brow_raise": 0.1}),
    (74, None, {"gaze_yaw": 18, "gaze_pitch": 0, "head_yaw": 3}),
    (76, "Surprise: brows up, lids wide, pupils dilate", {"gaze_yaw": 0, "gaze_pitch": 3, "head_yaw": 1}),
    (86, None, {"gaze_pitch": 4, "brow_raise": 1.0, "lid_raise": 0.8, "pupil": 0.8, "head_pitch": 3}),
    (100, None, {"gaze_pitch": 4, "brow_raise": 1.0, "lid_raise": 0.8, "pupil": 0.8, "head_pitch": 3}),
    (110, "Eye roll upward", {"gaze_yaw": -15, "gaze_pitch": 8, "brow_raise": 0.7, "lid_raise": 0.5, "pupil": 0.5}),
    (118, None, {"gaze_yaw": -6, "gaze_pitch": 22, "brow_raise": 0.9, "lid_raise": 0.6, "head_roll": 3, "pupil": 0.3}),
    (124, None, {"gaze_yaw": 6, "gaze_pitch": 23, "brow_raise": 0.9, "lid_raise": 0.6, "head_roll": 4}),
    (132, None, {"gaze_yaw": 15, "gaze_pitch": 4, "brow_raise": 0.5, "head_roll": 2}),
    (135, "Blink", {"gaze_yaw": 2, "gaze_pitch": 0, "blink": 1.0, "brow_raise": 0.2}),
    (136, None, {"gaze_yaw": 0, "gaze_pitch": 0, "blink": 1.0}),
    (140, None, {}),
    (146, "Look down (lids follow), slight frown", {"gaze_pitch": -20, "gaze_yaw": -3, "brow_lower": 0.35, "pupil": -0.3}),
    (160, None, {"gaze_pitch": -21, "gaze_yaw": 3, "brow_lower": 0.4, "head_pitch": -2, "pupil": -0.3}),
    (163, "Back to the viewer: warm smile", {"gaze_pitch": 0, "gaze_yaw": 0, "brow_lower": 0.1}),
    (176, None, {"smile": 0.42, "squint": 0.4, "brow_raise": 0.1}),
    (186, None, {"smile": 0.45, "squint": 0.42, "brow_raise": 0.1}),
    (188, "Blink", {"smile": 0.45, "squint": 0.42, "blink": 1.0}),
    (189, None, {"smile": 0.45, "squint": 0.42, "blink": 1.0}),
    (193, None, {"smile": 0.45, "squint": 0.42}),
    (168 + 36, None, {"smile": 0.4, "squint": 0.38}),
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnm", required=True)
    ap.add_argument("--controls", default=os.path.join(HERE, "assets", "gnm_controls.npz"))
    ap.add_argument("--textures", default=os.path.join(HERE, "textures"))
    ap.add_argument("--set", default="03")
    ap.add_argument("--out", default=os.path.join(HERE, "GazeForge_GNM_Timeline.blend"))
    a = ap.parse_args(argv)
    head, eyes, arm = build_scene.build(a.gnm, a.controls, os.path.abspath(a.textures), a.set,
                                        res=(1280, 720), samples=32)
    sc = bpy.context.scene
    rig = Rig(head, eyes, arm)
    rig.clear_animation()
    for frame, marker, pose in KEYS:
        f = dict(NEUTRAL); f.update(pose)
        rig.apply(f, key_frame=frame)
        if marker:
            sc.timeline_markers.new(marker, frame=frame)
    # Bezier with automatic-clamped handles: smooth eases, no overshoot
    for o in (arm, head.data.shape_keys, eyes.data.shape_keys):
        if o.animation_data and o.animation_data.action:
            for fc in action_fcurves(o.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
                fc.update()
    sc.frame_start, sc.frame_end = 1, KEYS[-1][0]
    sc.render.fps = FPS
    sc.frame_set(1)
    sc.render.use_motion_blur = True
    sc["gf_note"] = ("GazeForge GNM timeline: press Space to play. Animated: eye_L/eye_R bones (iris direction), "
                     "neck/head bones, shape keys blink_*, upper_lid_*, lower_lid_raise_*, brow_raise_*, brow_lower_*, "
                     "smile, and pupil_dilate on GNM_Eyes. Edit keys in the Timeline / Dope Sheet.")
    txt = bpy.data.texts.new("README_GazeForge")
    txt.write(sc["gf_note"] + "\n\nMarkers:\n" + "\n".join(f"  frame {f}: {m}" for f, m, _ in KEYS if m))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    print("saved", a.out)


if __name__ == "__main__":
    main()
