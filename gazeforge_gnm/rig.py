"""Drive the GNM rig from motion channels: eye rotations (vergent fixation), head/neck pose,
eyelid-gaze coupling, blinks, brows, pupil and mouth expression shape keys; keyframing."""
import math
import numpy as np
from mathutils import Quaternion, Vector

LID_COUPLING = dict(down_per_deg=1 / 30.0, down_max=0.75, up_per_deg=1 / 28.0, up_max=0.8,
                    lower_down_per_deg=1 / 40.0, lower_up_per_deg=1 / 60.0)


def action_fcurves(action):
    """F-curves of an action in Blender 4.2 (legacy) and 4.4+/5.x (layered actions)."""
    if hasattr(action, "fcurves") and not hasattr(action, "layers"):
        return list(action.fcurves)
    out = []
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    out.extend(cb.fcurves)
    except AttributeError:
        pass
    if not out and hasattr(action, "fcurves"):
        out = list(action.fcurves)
    return out


def _quat_between(a, b):
    a = Vector(a).normalized(); b = Vector(b).normalized()
    return a.rotation_difference(b)


def eye_rotation(eye_center, target):
    """Shortest-arc rotation taking the GNM eye forward axis (+Z) onto the line eye->target (Listing L1)."""
    return _quat_between((0, 0, 1), Vector(target) - Vector(eye_center))


class Rig:
    def __init__(self, head, eyes, arm):
        self.head, self.eyes, self.arm = head, eyes, arm
        self.kb = head.data.shape_keys.key_blocks
        self.kbe = eyes.data.shape_keys.key_blocks
        pb = arm.pose.bones
        self.eL, self.eR = pb["eye_L"].bone.head_local.copy(), pb["eye_R"].bone.head_local.copy()
        self.head_bone = pb["head"]
        self.neck_bone = pb["neck"]

    def refresh_joints(self):
        pb = self.arm.pose.bones
        self.eL, self.eR = pb["eye_L"].bone.head_local.copy(), pb["eye_R"].bone.head_local.copy()

    def apply(self, f, key_frame=None):
        """f: dict of channels (see motion.CHANNELS). If key_frame is given, keyframe everything."""
        pb = self.arm.pose.bones
        # head pose: split 40 % neck / 60 % head (yaw about +Y; pitch up = rotation about -X; roll about +Z)
        def hq(scale):
            return (Quaternion((0, 1, 0), math.radians(f["head_yaw"] * scale)) @
                    Quaternion((1, 0, 0), math.radians(-f["head_pitch"] * scale)) @
                    Quaternion((0, 0, 1), math.radians(-f["head_roll"] * scale)))
        pb["neck"].rotation_quaternion = hq(0.4)
        pb["head"].rotation_quaternion = hq(0.6)
        # eye-in-head = gaze - head (gaze is a world/camera-frame direction, the head carries the eyes)
        ey = f["gaze_yaw"] - f["head_yaw"]
        ep = f["gaze_pitch"] - f["head_pitch"]
        ey = float(np.clip(ey, -38, 38)); ep = float(np.clip(ep, -32, 30))
        mid = (self.eL + self.eR) / 2
        d = f.get("vergence_m", 0.8)
        y, p = math.radians(ey), math.radians(ep)
        target = mid + Vector((math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p))) * d
        # small down-and-in globe rotation during blinks (Collewijn et al. 1985)
        b = f["blink"]
        if b > 0:
            target = target + Vector((0, -0.03 * b, 0)) * d
        pb["eye_L"].rotation_quaternion = eye_rotation(self.eL, target)
        pb["eye_R"].rotation_quaternion = eye_rotation(self.eR, target)
        # eyelid-gaze coupling (upper lid follows vertical gaze, lower lid a little)
        c = LID_COUPLING
        down = min(max(-ep, 0) * c["down_per_deg"], c["down_max"])
        up = min(max(ep, 0) * c["up_per_deg"], c["up_max"])
        keys = {}
        for s in ("L", "R"):
            keys[f"blink_{s}"] = b
            keys[f"upper_lid_down_{s}"] = down * (1 - b)
            keys[f"upper_lid_raise_{s}"] = (up + f["lid_raise"]) * (1 - b)
            keys[f"lower_lid_raise_{s}"] = f["squint"] + 0.45 * f["smile"] - max(-ep, 0) * c["lower_down_per_deg"] \
                + max(ep, 0) * c["lower_up_per_deg"]
            keys[f"brow_raise_{s}"] = f["brow_raise"] + 0.25 * up
            keys[f"brow_lower_{s}"] = f["brow_lower"] + 0.12 * b
        keys["smile"] = f["smile"]
        for k, v in keys.items():
            if k in self.kb:
                self.kb[k].value = float(np.clip(v, self.kb[k].slider_min, self.kb[k].slider_max))
        if "pupil_dilate" in self.kbe:
            self.kbe["pupil_dilate"].value = float(np.clip(f["pupil"], -1, 1.5))
        if key_frame is not None:
            for n in ("neck", "head", "eye_L", "eye_R"):
                pb[n].keyframe_insert("rotation_quaternion", frame=key_frame)
            for k in keys:
                if k in self.kb:
                    self.kb[k].keyframe_insert("value", frame=key_frame)
            if "pupil_dilate" in self.kbe:
                self.kbe["pupil_dilate"].keyframe_insert("value", frame=key_frame)

    def clear_animation(self):
        for o in (self.arm, self.head.data.shape_keys, self.eyes.data.shape_keys):
            if o.animation_data:
                o.animation_data_clear()

    def keyframe_programme(self, frames, start=1):
        self.clear_animation()
        for i, f in enumerate(frames):
            self.apply(f, key_frame=start + i)
        for o in (self.arm, self.head.data.shape_keys, self.eyes.data.shape_keys):
            ad = o.animation_data
            if ad and ad.action:
                for fc in action_fcurves(ad.action):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"
        return start + len(frames) - 1
