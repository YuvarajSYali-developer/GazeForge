"""Vendored from GazeForge V3 (gazeforge/kinematics/blink_model.py, same author; verified by Gate E).

gazeforge.kinematics.blink_model -- eyelid aperture a(t) in [0, 1] with asymmetric closing/opening (spec 6.5).

Model (documented approximation; each phase uses the minimum-jerk S-curve as a smooth interpolant):
    t in [t0, t0 + Tc)            closing:  a = 1 - (1 - a_min) * S((t - t0) / Tc)
    t in [t0 + Tc, t0 + Tc + Th)  closed:   a = a_min
    t in [.., .. + To)            opening:  a = a_min + (1 - a_min) * S((t - t0 - Tc - Th) / To)
    otherwise                     open:     a = 1
S(u) = 10u^3 - 15u^4 + 6u^5, so a(t) is C^2 and the lid velocity is zero at every phase boundary.

Defaults (cited numbers; see research/reports/temporal.md for the quotes):
  * down (closing) phase "roughly twice as fast as the up phase, lasting roughly 75-100 milliseconds"
    (Willett, Maenner & Mayo 2023, Front. Syst. Neurosci. 17:1242654, citing Evinger, Manning & Sibony 1991,
    IOVS 32:387-400)  ->  Tc = 75 ms, To = 2 Tc = 150 ms;
  * full lid closure lasts ~10-50 ms (Ousler et al. 2014, via Willett 2023) -> Th = 30 ms;
  * whole blink 250-450 ms (Volkmann et al. 1982, via Willett 2023) -> Tc + Th + To = 255 ms;
  * for a 30 deg lid excursion, down phase peak ~840 deg/s, up phase ~400 deg/s (Evinger 1991 as reported in
    Wouters 2001 thesis, OCR text) -- used only as a comparison (E5 report), not fitted;
  * spontaneous blink rate "roughly 15 times per minute" (Willett 2023) -> optional Poisson schedule.

Pupil occlusion FLAG (approximation): `pupil_occluded = a < occlusion_aperture` (default 0.5). The occluded
FRACTION of the apparent pupil boundary is computed from the rendered lid geometry by Agent 6 (BVH), not here.
Not modelled: eye retraction / downward-nasal globe rotation during blinks, lid-saccade coupling, incomplete
blinks (other than via a_min), pupil-size change around blinks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .saccade_model import minimum_jerk_position, mj_velocity

__all__ = ["BlinkConfig", "Blink", "BLINK_STATES", "aperture", "aperture_velocity", "blink_state",
           "spontaneous_blinks", "pupil_occluded"]

BLINK_STATES = ("open", "closing", "closed", "opening")
_TOL = 1e-9   # s; phase boundaries are compared with this tolerance (float sums of the configured durations)


@dataclass(frozen=True)
class BlinkConfig:
    close_s: float = 0.075
    closed_s: float = 0.030
    open_s: float = 0.150
    min_aperture: float = 0.0
    occlusion_aperture: float = 0.5
    rate_per_min: float = 15.0              # used only by spontaneous_blinks
    lid_excursion_deg: float = 30.0         # only to express aperture velocity in deg/s for comparisons

    def __post_init__(self):
        for k in ("close_s", "open_s"):
            if not (getattr(self, k) > 0.0 and math.isfinite(getattr(self, k))):
                raise ValueError(f"{k} must be > 0")
        if not (self.closed_s >= 0.0):
            raise ValueError("closed_s must be >= 0")
        if not (0.0 <= self.min_aperture < 1.0):
            raise ValueError("min_aperture must be in [0, 1)")
        if not (0.0 < self.occlusion_aperture <= 1.0):
            raise ValueError("occlusion_aperture must be in (0, 1]")

    @property
    def total_s(self) -> float:
        return self.close_s + self.closed_s + self.open_s


@dataclass(frozen=True)
class Blink:
    t_start: float
    config: BlinkConfig = BlinkConfig()

    @property
    def t_closed(self):
        return self.t_start + self.config.close_s

    @property
    def t_opening(self):
        return self.t_closed + self.config.closed_s

    @property
    def t_end(self):
        return self.t_opening + self.config.open_s


def _masks(t, b):
    return ((t >= b.t_start - _TOL) & (t < b.t_closed - _TOL), (t >= b.t_closed - _TOL) & (t < b.t_opening - _TOL),
            (t >= b.t_opening - _TOL) & (t < b.t_end - _TOL))


def _check(blinks):
    bl = sorted(blinks, key=lambda b: b.t_start)
    for a, b in zip(bl, bl[1:]):
        if b.t_start < a.t_end:
            raise ValueError(f"blinks overlap: {a.t_start} + {a.config.total_s} > {b.t_start}")
    return bl


def aperture(t, blinks):
    """Lid aperture a(t) in [0, 1] (1 = open) for an array of times and a list of Blink."""
    t = np.asarray(t, dtype=np.float64)
    a = np.ones_like(t)
    for b in _check(blinks):
        c = b.config
        amp = 1.0 - c.min_aperture
        m1, m2, m3 = _masks(t, b)
        a = np.where(m1, 1.0 - amp * minimum_jerk_position((t - b.t_start) / c.close_s), a)
        a = np.where(m2, c.min_aperture, a)
        a = np.where(m3, c.min_aperture + amp * minimum_jerk_position((t - b.t_opening) / c.open_s), a)
    return a


def aperture_velocity(t, blinks):
    """da/dt (1/s)."""
    t = np.asarray(t, dtype=np.float64)
    v = np.zeros_like(t)
    for b in _check(blinks):
        c = b.config
        amp = 1.0 - c.min_aperture
        m1, m2, m3 = _masks(t, b)
        v = np.where(m1, -amp / c.close_s * mj_velocity((t - b.t_start) / c.close_s), v)
        v = np.where(m3, amp / c.open_s * mj_velocity((t - b.t_opening) / c.open_s), v)
    return v


def blink_state(t, blinks):
    """Integer codes into BLINK_STATES (0 open, 1 closing, 2 closed, 3 opening)."""
    t = np.asarray(t, dtype=np.float64)
    s = np.zeros(t.shape, dtype=np.int8)
    for b in _check(blinks):
        m1, m2, m3 = _masks(t, b)
        s = np.where(m1, 1, s)
        s = np.where(m2, 2, s)
        s = np.where(m3, 3, s)
    return s.astype(np.int8)


def pupil_occluded(a, config: BlinkConfig = BlinkConfig()):
    return np.asarray(a) < config.occlusion_aperture


def spontaneous_blinks(t0: float, t1: float, rng: np.random.Generator, config: BlinkConfig = BlinkConfig(),
                       min_gap_s: float = 0.0):
    """Poisson blink onsets at rate_per_min in [t0, t1 - total], non-overlapping (rejection of overlaps)."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    rate = config.rate_per_min / 60.0
    out, t = [], t0
    while rate > 0.0:
        t += rng.exponential(1.0 / rate)
        if t + config.total_s > t1:
            break
        if out and t < out[-1].t_end + min_gap_s:
            continue
        out.append(Blink(float(t), config))
    return out
