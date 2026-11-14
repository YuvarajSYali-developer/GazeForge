"""GazeForge GNM -- render natural eye-movement videos of the textured GNM head.

Takes ONE of three inputs and renders the frames + an MP4 (each video <= 15 s):
  Left            both eyes move toward the left of the image (natural saccade, head follows a little,
                  blinks, slight brow/lid expression)
  Right           the same toward the right
  Generate Motion a natural expressive sequence: scanning left/right, an upward eye roll with raised
                  brows and lids, a look down with the lids following, blinks, a soft smile

Usage (Blender 4.2+):
  blender -b GazeForge_GNM.blend --python gnm_eyes.py -- --input left
  blender -b GazeForge_GNM.blend --python gnm_eyes.py -- --input "Generate Motion" --texture 05
  python gnm_eyes.py --blend GazeForge_GNM.blend --input right            (with `pip install bpy`)
If --input is omitted the script asks for it. Options: --texture 01..08, --fps, --res 960x540,
--samples, --framing face|portrait|eyes, --side-convention image|subject, --out, --keep-frames, --resume.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INPUTS = {"left": "left", "l": "left", "right": "right", "r": "right",
          "generate motion": "generate_motion", "generate_motion": "generate_motion", "generate": "generate_motion",
          "g": "generate_motion", "motion": "generate_motion"}
MAX_SECONDS = 15.0


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="Left | Right | Generate Motion")
    ap.add_argument("--blend", default=os.path.join(HERE, "GazeForge_GNM.blend"))
    ap.add_argument("--texture", default=None, help="texture set id 01..08 (default: the one saved in the .blend)")
    ap.add_argument("--textures-dir", default=os.path.join(HERE, "textures"))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--framing", default="face", choices=["face", "portrait", "eyes"])
    ap.add_argument("--side-convention", default="image", choices=["image", "subject"],
                    help="'image' (default): Left = toward the left of the video frame; "
                         "'subject': Left = the person's own left")
    ap.add_argument("--out", default=None, help="output .mp4 (default videos/<input>.mp4)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip frames already rendered in the frames folder")
    ap.add_argument("--gpu", action="store_true", help="render on the GPU (OptiX/CUDA/HIP/oneAPI/Metal) if available")
    ap.add_argument("--frame-step", type=int, default=1, help="render every n-th frame (quick previews)")
    a = ap.parse_args(argv)
    if not a.input:
        a.input = input("Enter input (Left / Right / Generate Motion): ")
    key = a.input.strip().lower().replace("-", " ")
    if key not in INPUTS:
        raise SystemExit(f"unknown input {a.input!r}; use Left, Right or Generate Motion")
    a.mode = INPUTS[key]
    return a


def ensure_scene(a):
    import bpy
    if not bpy.data.filepath or os.path.abspath(bpy.data.filepath) != os.path.abspath(a.blend):
        if not any(o.name == "GNM_Head" for o in bpy.data.objects):
            bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
    return bpy


def main():
    a = parse()
    bpy = ensure_scene(a)
    from gazeforge_gnm import blender_scene as bs
    from gazeforge_gnm.motion import PROGRAMMES
    from gazeforge_gnm.rig import Rig

    sc = bpy.context.scene
    head, eyes, arm = bpy.data.objects["GNM_Head"], bpy.data.objects["GNM_Eyes"], bpy.data.objects["GNM_Head_Rig"]
    if a.texture:
        tex = glob.glob(os.path.join(a.textures_dir, f"set{a.texture}_*"))
        if not tex:
            raise SystemExit(f"texture set {a.texture} not found under {a.textures_dir}")
        bs.apply_texture_set(tex[0], head, eyes, arm)
    rig = Rig(head, eyes, arm)
    rig.refresh_joints()
    frames = PROGRAMMES[a.mode](fps=a.fps)
    if a.side_convention == "subject":       # person's own left = image right
        for f in frames:
            f["gaze_yaw"] = -f["gaze_yaw"]; f["head_yaw"] = -f["head_yaw"]
    seconds = (len(frames) - 1) / a.fps
    assert seconds <= MAX_SECONDS, f"programme is {seconds:.2f} s > {MAX_SECONDS} s"
    end = rig.keyframe_programme(frames, start=1)
    sc.frame_start, sc.frame_end, sc.frame_step = 1, end, a.frame_step
    sc.render.fps = a.fps
    w, h = (int(v) for v in a.res.lower().split("x"))
    bs.setup_render(sc, res=(w, h), samples=a.samples)
    sc.render.fps = a.fps
    if a.gpu:
        print("[gazeforge] device:", bs.enable_gpu(sc), flush=True)
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    eye_h = (arm.matrix_world @ arm.pose.bones["eye_L"].bone.head_local).z
    bs.set_framing(sc.camera, eye_h, a.framing)

    out = a.out or os.path.join(HERE, "videos", f"{a.mode}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fdir = os.path.splitext(out)[0] + "_frames"
    os.makedirs(fdir, exist_ok=True)
    sc.render.filepath = os.path.join(fdir, "frame_")
    sc.render.image_settings.file_format = "PNG"
    sc.render.use_overwrite = not a.resume
    t0 = time.time()
    print(f"[gazeforge] input={a.mode} frames={end} ({seconds:.2f} s at {a.fps} fps) texture={sc.get('gf_texture_set')}",
          flush=True)
    bpy.ops.render.render(animation=True)
    print(f"[gazeforge] rendered in {time.time() - t0:.0f} s", flush=True)
    json.dump({"input": a.mode, "fps": a.fps, "seconds": seconds, "texture_set": sc.get("gf_texture_set"),
               "side_convention": a.side_convention, "channels_per_frame": frames},
              open(os.path.splitext(out)[0] + "_motion.json", "w"))
    ff = shutil.which("ffmpeg")
    if ff is None:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            print("[gazeforge] ffmpeg not found: frames are in", fdir)
            return
    rate = a.fps / a.frame_step
    frames_png = sorted(glob.glob(os.path.join(fdir, "frame_*.png")))
    lst = os.path.join(fdir, "frames.txt")          # concat list: works on Windows/macOS/Linux ffmpeg builds
    with open(lst, "w") as fh:
        for p in frames_png:
            fh.write("file '%s'\nduration %.6f\n" % (os.path.abspath(p).replace("\\", "/").replace("'", "'\\''"), 1.0 / rate))
        if frames_png:
            fh.write("file '%s'\n" % os.path.abspath(frames_png[-1]).replace("\\", "/"))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-r", f"{rate}", "-frames:v", str(len(frames_png)),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    if not a.keep_frames:
        shutil.rmtree(fdir, ignore_errors=True)
    print("[gazeforge] video:", out, flush=True)


if __name__ == "__main__":
    main()
