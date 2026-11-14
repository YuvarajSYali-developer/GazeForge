"""Natural eye-movement programmes for the three required inputs (pure NumPy).

Every channel is sampled at the video frame rate (plus the per-frame values Blender
interpolates for motion blur). Building blocks, all from the GazeForge V3 temporal
models (vendored in gazeforge_gnm.kinematics, verified by its Gate E):
  * saccades: skewed gamma velocity profile matched to the Baloh (1975) main sequence;
  * blinks: asymmetric close/hold/open (75/30/150 ms) minimum-jerk lid closure;
  * fixation: slow drift (random walk) + microsaccades (0.2-0.5 deg, every ~0.6-1.2 s);
  * eye-head coordination: for large gaze shifts the head contributes ~25 % with ~60 ms
    latency and a slower minimum-jerk movement; the eye counter-rotates (VOR), so gaze = eye + head;
  * eyelid-gaze coupling and brow/lid expression channels are handled by the rig.
Angles in degrees; yaw > 0 = towards the IMAGE RIGHT (viewer's right = the subject's left),
pitch > 0 = up.
"""
import math
import numpy as np
from .kinematics.saccade_model import make_saccade, minimum_jerk_position
from .kinematics.blink_model import Blink, BlinkConfig, aperture

CHANNELS = ["gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch", "head_roll", "blink",
            "brow_raise", "brow_lower", "lid_raise", "squint", "pupil", "smile", "vergence_m"]


class Programme:
    def __init__(self, duration_s, fps=30, seed=0):
        self.fps = fps
        self.n = int(round(duration_s * fps)) + 1
        self.t = np.arange(self.n) / fps
        self.rng = np.random.default_rng(seed)
        self.gaze = np.zeros((self.n, 2))           # world gaze target direction (yaw, pitch)
        self.events = []                              # (t0, from, to, kind, dur)
        self.blinks = []
        self.ch = {c: np.zeros(self.n) for c in CHANNELS}
        self.ch["vergence_m"][:] = 0.8
        self._cur = np.zeros(2)
        self._last_t = 0.0

    # ---- gaze events -------------------------------------------------------------
    def saccade(self, t0, to):
        to = np.asarray(to, float)
        amp = float(np.linalg.norm(to - self._cur))
        if amp < 1e-6:
            return t0
        prof = make_saccade(math.radians(amp))
        self.events.append((t0, self._cur.copy(), to, prof))
        self._cur = to
        return t0 + prof.duration

    def smooth(self, t0, path, dur):
        """Smooth (pursuit-like) movement along a polyline of (yaw, pitch) points, minimum-jerk timing."""
        path = np.vstack([self._cur, np.asarray(path, float)])
        self.events.append((t0, path, None, ("smooth", dur)))
        self._cur = path[-1].copy()
        return t0 + dur

    def blink(self, t0, close_s=0.075, closed_s=0.03, open_s=0.15):
        self.blinks.append(Blink(t0, BlinkConfig(close_s=close_s, closed_s=closed_s, open_s=open_s)))

    def expr(self, name, t0, t1, value, ramp=0.25):
        """Ease a channel to `value` over [t0, t0+ramp] and back to its previous level over [t1-ramp, t1]
        (t1=None: hold until the end)."""
        c = self.ch[name]
        u_in = minimum_jerk_position(np.clip((self.t - t0) / ramp, 0, 1))
        if t1 is None:
            env = u_in
        else:
            u_out = minimum_jerk_position(np.clip((self.t - (t1 - ramp)) / ramp, 0, 1))
            env = u_in * (1 - u_out)
        self.ch[name] = c + (value - 0) * env

    # ---- synthesis -----------------------------------------------------------------
    def build(self, head_gain=0.25, head_latency=0.06, micro=True):
        t = self.t
        g = np.zeros((self.n, 2))
        cur = np.zeros(2)
        evs = sorted(self.events, key=lambda e: e[0])
        # piecewise: each event moves gaze from its start to its end
        g[:] = 0.0
        state = np.zeros(2)
        for k, (t0, a, b, prof) in enumerate(evs):
            if isinstance(prof, tuple):      # smooth path
                dur = prof[1]
                u = minimum_jerk_position(np.clip((t - t0) / dur, 0, 1))
                path = a
                seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
                cum = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
                pos = np.stack([np.interp(u, cum, path[:, 0]), np.interp(u, cum, path[:, 1])], 1)
                m = t >= t0
                g[m] = pos[m]
            else:
                s = prof.position(np.clip(t - t0, 0, prof.duration))
                pos = a[None] + (b - a)[None] * np.asarray(s)[:, None]
                m = t >= t0
                g[m] = pos[m]
        # fixational eye movements: drift + microsaccades (only between larger movements)
        if micro:
            drift = np.cumsum(self.rng.normal(0, 0.035, (self.n, 2)), 0)
            drift -= np.convolve(drift[:, 0], np.ones(15) / 15, "same")[:, None] * 0  # keep bounded below
            drift = drift - np.linspace(0, 1, self.n)[:, None] * drift[-1]
            drift *= 0.6
            ms = np.zeros((self.n, 2))
            tt = self.rng.uniform(0.3, 0.9)
            while tt < t[-1]:
                ang = self.rng.uniform(0, 2 * np.pi)
                amp = self.rng.uniform(0.2, 0.5)
                ms[t >= tt] += amp * np.array([np.cos(ang), np.sin(ang)])
                # microsaccades tend to correct drift: pull back toward 0
                ms[t >= tt] -= 0.5 * (ms[np.searchsorted(t, tt) - 1] if np.searchsorted(t, tt) > 0 else 0) * 0
                tt += self.rng.uniform(0.6, 1.2)
            ms -= np.linspace(0, 1, self.n)[:, None] * 0
            # keep fixational offsets small and zero-mean per segment
            fix = drift * 0.15 + (ms - np.convolve(ms[:, 0], np.ones(31) / 31, "same")[:, None] * 0)
            fix = np.clip(fix - fix.mean(0), -0.6, 0.6)
            g = g + fix
        # eye-head coordination
        h = np.zeros((self.n, 2))
        for (t0, a, b, prof) in evs:
            if isinstance(prof, tuple):
                end = a[-1]; start = a[0]; dur = prof[1] * 1.3
            else:
                end, start, dur = b, a, 0.25 + 0.006 * np.linalg.norm(b - a) * 10
            delta = (end - start) * head_gain
            u = minimum_jerk_position(np.clip((t - t0 - head_latency) / dur, 0, 1))
            h += delta[None] * u[:, None]
        self.ch["head_yaw"] += h[:, 0]
        self.ch["head_pitch"] += h[:, 1]
        # small idle head sway (breathing / postural), 0.3-0.5 deg
        for name, amp, f in (("head_yaw", 0.35, 0.11), ("head_pitch", 0.3, 0.23), ("head_roll", 0.4, 0.07)):
            ph = self.rng.uniform(0, 2 * np.pi)
            self.ch[name] += amp * np.sin(2 * np.pi * f * t + ph)
        self.ch["gaze_yaw"] = g[:, 0]
        self.ch["gaze_pitch"] = g[:, 1]
        # blinks: closure = 1 - aperture
        if self.blinks:
            self.ch["blink"] = np.clip(1 - aperture(t, self.blinks) + self.ch["blink"], 0, 1)
        return self.frames()

    def frames(self):
        return [{c: float(self.ch[c][i]) for c in CHANNELS} | {"t": float(self.t[i])} for i in range(self.n)]


def programme_left(direction=-1, fps=30, seed=1, duration=6.0):
    """'Left' (direction=-1) / 'Right' (+1): both eyes look toward that side of the image, naturally:
    fixate the viewer, gaze-shift with a coordinated head turn, look around on that side, blink; ends looking there."""
    p = Programme(duration, fps, seed)
    rng = p.rng
    amp = 24.0
    p.blink(0.55)
    t = p.saccade(1.10, (direction * amp, rng.uniform(-1.5, 1.0)))
    p.blink(1.07, close_s=0.07, closed_s=0.02, open_s=0.14)          # gaze-evoked blink with the large saccade
    p.expr("brow_raise", 1.0, 2.6, 0.12, ramp=0.3)                   # slight attention brow lift
    t = p.saccade(t + 0.9, (direction * (amp + 6), rng.uniform(-3, -1)))   # look a little further
    t = p.saccade(t + 0.65, (direction * (amp - 2), 1.0))
    p.blink(t + 0.45)
    p.expr("squint", t + 0.2, t + 1.3, 0.25, ramp=0.35)              # slight interest squint
    t = p.saccade(t + 1.0, (direction * (amp + 2), -1.0))             # stays on that side to the end
    return p.build()


def programme_generate(fps=30, seed=7, duration=13.0):
    """'Generate Motion': a natural expressive sequence -- scanning left/right, an upward eye roll with
    raised brows, a look down with lids following, blinks, return to the viewer with a soft smile."""
    p = Programme(duration, fps, seed)
    p.blink(0.4)
    t = p.saccade(0.9, (-18, 2))                                         # glance image-left
    t = p.saccade(t + 0.55, (18, 1))                                     # sweep to image-right
    p.blink(t - 0.02, close_s=0.07, closed_s=0.02, open_s=0.14)
    t = p.saccade(t + 0.6, (-8, -3))                                     # back across
    t = p.saccade(t + 0.4, (6, -2))
    t = p.saccade(t + 0.5, (0, 0))                                       # at the viewer
    # eye roll: down-left -> up over the top -> down-right, smooth, with brows + upper lids raised
    t0 = t + 0.6
    t = p.saccade(t0, (-14, -6))
    roll = [(-20, 6), (-12, 20), (0, 25), (12, 20), (19, 6), (12, -6)]
    t_roll = t + 0.15
    t = p.smooth(t_roll, roll, 1.6)
    p.expr("brow_raise", t_roll - 0.1, t + 0.3, 0.9, ramp=0.35)
    p.expr("lid_raise", t_roll + 0.2, t, 0.6, ramp=0.4)
    p.expr("pupil", t_roll - 0.1, t + 0.5, 0.3, ramp=0.4)
    p.expr("head_roll", t_roll, t + 0.6, 4.0, ramp=0.6)
    p.expr("head_pitch", t_roll + 0.1, t + 0.4, 3.0, ramp=0.6)
    t = p.saccade(t + 0.35, (0, 0))
    p.blink(t + 0.05)
    p.expr("pupil", 0.0, None, 0.0)
    # look down (lids follow the eyes), brief frown of concentration, then back up
    t = p.saccade(t + 0.9, (-4, -22))
    p.expr("brow_lower", t, t + 1.2, 0.35, ramp=0.3)
    p.expr("pupil", t, t + 1.4, -0.3, ramp=0.5)                        # near-response: pupils constrict
    t = p.saccade(t + 0.8, (5, -20))
    t = p.saccade(t + 0.6, (0, 0))
    p.blink(t - 0.03, close_s=0.07, closed_s=0.025, open_s=0.15)
    # warm finish: Duchenne smile (cheeks raise the lower lids) and a final blink
    p.expr("smile", t + 0.4, None, 0.42, ramp=0.7)
    p.expr("squint", t + 0.5, None, 0.4, ramp=0.7)
    t = p.saccade(t + 1.0, (-3, 1))
    t = p.saccade(t + 0.7, (0, 0))
    p.blink(t + 0.35)
    return p.build()


PROGRAMMES = {
    "left": lambda fps=30: programme_left(-1, fps=fps, seed=1),
    "right": lambda fps=30: programme_left(+1, fps=fps, seed=2),
    "generate_motion": lambda fps=30: programme_generate(fps=fps),
}
