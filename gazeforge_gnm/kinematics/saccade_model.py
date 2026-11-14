"""Vendored from GazeForge V3 (gazeforge/kinematics/saccade_model.py, same author; verified by Gate E).

gazeforge.kinematics.saccade_model -- saccade main sequence and velocity profiles (spec 6.5, Upgrade D).

Pure NumPy (no scipy, no bpy). Units: radians, seconds. Degrees/ms appear only in the published constants
(converted on construction) and in reports.

Two profiles, both applied to the *target direction* (great-circle interpolation of the cyclopean gaze
direction, see `slerp_dir`); the binocular solver then enforces the kinematic law at every tick.

(i) MINIMUM-JERK baseline (Flash & Hogan 1985, J. Neurosci. 5(7):1688-1703; derived for arm movements):
        S(u) = 10u^3 - 15u^4 + 6u^5,  u = t / D,
        V(t) = (A / D) * 30 u^2 (1 - u)^2,  V_peak = 1.875 A / D at u = 0.5,
        zero velocity and acceleration at both ends.  The profile is SYMMETRIC (skewness t_peak / D = 0.5 for
        every amplitude), which is the known mismatch with saccades: natural saccade velocity profiles are
        symmetric for small saccades and increasingly skewed (early peak) for large ones
        (Van Opstal & Van Gisbergen 1987, Vision Res. 27(5):731-745, as reported by Duchowski et al. 2017
        and Alitappeh et al. 2022 -- see research/reports/temporal.md).  Minimum-jerk alone therefore does
        NOT model natural saccades; it is kept as a baseline only.

(ii) GAMMA-SHAPED skewed profile (published model): Van Opstal & Van Gisbergen (1987) describe saccade
     velocity profiles with a gamma function
        V(t) = alpha (t / beta)^(gamma - 1) exp(-t / beta),   2 < gamma < 15,
     "small values of gamma yield asymmetrical velocity profiles" (equation and range quoted from Duchowski
     et al. 2017, J. Eye Mov. Res. 10(3):4, Eq. 5; the 1987 full text could not be obtained here).
     SYNTHESIS APPROXIMATION (ours, documented): the gamma function has infinite support, so it is
     truncated where it falls to a fraction eps of its peak on both sides and the pedestal eps * peak is
     subtracted.  With x = t / beta and f(x) = (x/m)^(gamma-1) exp(-(x - m)) (m = gamma - 1, f(m) = 1):
        x_s < m < x_e  with f(x_s) = f(x_e) = eps,   L = x_e - x_s,   u = (x - x_s) / L in [0, 1],
        v(u) = (f(x) - eps) * L / I,   I = integral_{x_s}^{x_e} (f - eps) dx,   S(u) = integral_0^u v,
     so S(0) = 0, S(1) = 1, v(0) = v(1) = 0 (position C^1, velocity continuous; acceleration jumps at the two
     ends, as in the abrupt onset of real saccades).  Shape factor R = V_peak D / A = (1 - eps) L / I and
     skewness s = t_peak / D = (m - x_s) / L.  (gamma, eps) are solved per amplitude so that
        R  = V_peak(A) D(A) / A   (duration AND peak velocity of the configured main sequence), and
        s  = s_target(A)          (skewness model, below),
     with gamma clamped to the published range [2, 15]; any residual that the clamp leaves is reported
     (`SaccadeProfile.residuals`), never hidden.

Main sequence (default: Baloh, Sills, Kumley & Honrubia 1975, Neurology 25:1065, values as reported by
Duchowski et al. 2017):  D = 37 ms + 2.7 ms/deg * A,   V_peak = 551 deg/s * (1 - exp(-A / 14 deg)).
Other duration fits kept for comparison: Carpenter 1988  D = 21 + 2.2 A ms (as cited by Devillez et al. 2020);
Collewijn et al. 1988  D = 23 + 2.7 A ms (as reported by Duchowski et al. 2017).

Skewness target (ASSUMPTION built on a secondary-source statement): saccades reach peak velocity "after an
approximately fixed acceleration phase of about 20-25 ms" (Alitappeh et al. 2022, citing Van Opstal & Van
Gisbergen 1987), and small saccades are symmetric:  s_target(A) = min(0.5, t_acc / D(A)),  t_acc = 22.5 ms.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "MainSequence", "BALOH_1975", "CARPENTER_1988_DURATION", "COLLEWIJN_1988_DURATION", "MAIN_SEQUENCES",
    "SkewModel", "SaccadeProfile", "make_saccade", "minimum_jerk_position", "mj_velocity",
    "mj_acceleration", "gamma_shape_params", "PROFILES",
    "GAMMA_RANGE",
]

DEG = math.pi / 180.0
PROFILES = ("minimum_jerk", "gamma_skewed")
#: shape-parameter range of the gamma velocity profile (Van Opstal & Van Gisbergen 1987, via Duchowski 2017)
GAMMA_RANGE = (2.0, 15.0)
_EPS_RANGE = (1e-4, 0.999)


# ======================================================================================
# main sequence
# ======================================================================================
@dataclass(frozen=True)
class MainSequence:
    """Linear amplitude-duration and exponential amplitude-peak-velocity main sequence, SI units.

    D(A) = dur_intercept_s + dur_slope_s_per_rad * A;  V(A) = vp_max_rad_s * (1 - exp(-A / vp_a0_rad)).
    `vp_max_rad_s = None` means the source gives no peak-velocity fit (duration-only reference curve)."""
    name: str
    dur_intercept_s: float
    dur_slope_s_per_rad: float
    vp_max_rad_s: float | None
    vp_a0_rad: float | None
    source: str

    @classmethod
    def from_published(cls, name, a_ms, b_ms_per_deg, va_deg_s=None, a0_deg=None, source=""):
        return cls(name, a_ms * 1e-3, b_ms_per_deg * 1e-3 / DEG,
                   None if va_deg_s is None else va_deg_s * DEG, None if a0_deg is None else a0_deg * DEG, source)

    def duration(self, amplitude_rad):
        return self.dur_intercept_s + self.dur_slope_s_per_rad * np.asarray(amplitude_rad, dtype=np.float64)

    def peak_velocity(self, amplitude_rad):
        if self.vp_max_rad_s is None:
            raise ValueError(f"main sequence {self.name!r} has no peak-velocity fit")
        a = np.asarray(amplitude_rad, dtype=np.float64)
        return self.vp_max_rad_s * (1.0 - np.exp(-a / self.vp_a0_rad))


BALOH_1975 = MainSequence.from_published(
    "baloh_1975", 37.0, 2.7, 551.0, 14.0,
    "Baloh RW, Sills AW, Kumley WE, Honrubia V (1975) Neurology 25:1065-1070; D = 37 + 2.7A ms, "
    "Vp = 551(1 - exp(-A/14)) deg/s, values as reported by Duchowski et al. (2017) JEMR 10(3):4")
CARPENTER_1988_DURATION = MainSequence.from_published(
    "carpenter_1988", 21.0, 2.2, None, None,
    "Carpenter RHS (1988) Movements of the Eyes, 2nd ed.; D = 21 + 2.2A ms as cited by Devillez et al. (2020)")
COLLEWIJN_1988_DURATION = MainSequence.from_published(
    "collewijn_1988", 23.0, 2.7, None, None,
    "Collewijn H, Erkelens CJ, Steinman RM (1988) J Physiol 404:157; D = 23 + 2.7A ms as reported by "
    "Duchowski et al. (2017)")
MAIN_SEQUENCES = {m.name: m for m in (BALOH_1975, CARPENTER_1988_DURATION, COLLEWIJN_1988_DURATION)}


@dataclass(frozen=True)
class SkewModel:
    """s_target(A) = min(s_max, t_acc / D(A)) -- assumption (see module docstring)."""
    t_acc_s: float = 0.0225
    s_max: float = 0.5

    def target(self, duration_s):
        return np.minimum(self.s_max, self.t_acc_s / np.asarray(duration_s, dtype=np.float64))


# ======================================================================================
# minimum-jerk
# ======================================================================================
def minimum_jerk_position(u):
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def mj_velocity(u):
    """dS/du (multiply by A/D for rad/s)."""
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 30.0 * u * u * (1.0 - u) ** 2, 0.0)


def mj_acceleration(u):
    """d2S/du2 (multiply by A/D^2 for rad/s^2)."""
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u), 0.0)


# ======================================================================================
# truncated gamma profile
# ======================================================================================
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_GL_PANELS = 16


def _logf(x, g):
    m = g - 1.0
    return (g - 1.0) * np.log(x / m) - (x - m)


def _f(x, g):
    return np.exp(_logf(np.asarray(x, dtype=np.float64), g))


def _level_points(g, eps):
    """x_s < m < x_e with f = eps (bisection, monotone on each side)."""
    m = g - 1.0
    target = math.log(eps)
    lo, hi = 0.0, m                       # f increasing on (0, m)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0 or _logf(mid, g) < target:
            lo = mid
        else:
            hi = mid
    xs = 0.5 * (lo + hi)
    lo, hi = m, m + 1.0
    while _logf(hi, g) > target:
        hi = m + 2.0 * (hi - m)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _logf(mid, g) > target:
            lo = mid
        else:
            hi = mid
    xe = 0.5 * (lo + hi)
    return xs, xe


def _integral(g, eps, a, b):
    """integral_a^b (f(x) - eps) dx for arrays b (a scalar), composite Gauss-Legendre (16 x 32 nodes)."""
    b = np.asarray(b, dtype=np.float64)
    flat = b.reshape(-1)
    edges = a + (flat[:, None] - a) * np.linspace(0.0, 1.0, _GL_PANELS + 1)[None, :]   # (n, P+1)
    lo, hi = edges[:, :-1], edges[:, 1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    x = mid[..., None] + half[..., None] * _GL_X                                        # (n, P, 32)
    val = np.where(x > 0.0, _f(np.maximum(x, 1e-300), g), 0.0) - eps
    out = np.sum(np.sum(val * _GL_W, axis=-1) * half, axis=-1)
    return out.reshape(b.shape)


def _shape(g, eps):
    xs, xe = _level_points(g, eps)
    L = xe - xs
    I = float(_integral(g, eps, xs, np.array(xe)))
    R = (1.0 - eps) * L / I
    s = (g - 1.0 - xs) / L
    return R, s, xs, xe, I


def _solve_eps(g, R_target):
    """eps in _EPS_RANGE with R(g, eps) = R_target (R decreases with eps); clamped at the bounds."""
    lo, hi = _EPS_RANGE
    if _shape(g, hi)[0] >= R_target:
        return hi
    if _shape(g, lo)[0] <= R_target:
        return lo
    llo, lhi = math.log(lo), math.log(hi)
    for _ in range(48):
        mid = 0.5 * (llo + lhi)
        if _shape(g, math.exp(mid))[0] > R_target:
            llo = mid
        else:
            lhi = mid
    return math.exp(0.5 * (llo + lhi))


def gamma_shape_params(R_target: float, s_target: float):
    """(gamma, eps) of the truncated gamma profile with shape factor R and skewness s (gamma clamped to
    GAMMA_RANGE, eps to (1e-4, 0.999)). s increases with gamma at fixed R."""
    g_lo, g_hi = GAMMA_RANGE

    def s_of(g):
        return _shape(g, _solve_eps(g, R_target))[1]

    if s_of(g_hi) <= s_target:
        g = g_hi
    elif s_of(g_lo) >= s_target:
        g = g_lo
    else:
        llo, lhi = math.log(g_lo), math.log(g_hi)
        for _ in range(40):
            mid = 0.5 * (llo + lhi)
            if s_of(math.exp(mid)) < s_target:
                llo = mid
            else:
                lhi = mid
        g = math.exp(0.5 * (llo + lhi))
    return g, _solve_eps(g, R_target)


# ======================================================================================
# profile object
# ======================================================================================
@dataclass(frozen=True)
class SaccadeProfile:
    """Normalised saccade of amplitude A (rad) and duration D (s); time t in [0, D] from onset.

    position(t) in [0, 1] (fraction of the amplitude), velocity(t) rad/s, acceleration(t) rad/s^2."""
    kind: str
    amplitude: float
    duration: float
    gamma: float | None = None
    eps: float | None = None
    xs: float | None = None
    xe: float | None = None
    integral: float | None = None
    targets: dict = field(default_factory=dict)

    # -- normalised shape (u in [0, 1]) --
    def _S(self, u):
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        if self.kind == "minimum_jerk":
            return minimum_jerk_position(u)
        L = self.xe - self.xs
        x = self.xs + u * L
        return np.clip(_integral(self.gamma, self.eps, self.xs, x) / self.integral, 0.0, 1.0)

    def _V(self, u):
        u = np.asarray(u, dtype=np.float64)
        if self.kind == "minimum_jerk":
            return mj_velocity(u)
        L = self.xe - self.xs
        x = self.xs + np.clip(u, 0.0, 1.0) * L
        v = (_f(x, self.gamma) - self.eps) * L / self.integral
        return np.where((u >= 0.0) & (u <= 1.0), np.maximum(v, 0.0), 0.0)

    def _A(self, u):
        u = np.asarray(u, dtype=np.float64)
        if self.kind == "minimum_jerk":
            return mj_acceleration(u)
        L = self.xe - self.xs
        x = self.xs + np.clip(u, 0.0, 1.0) * L
        a = _f(x, self.gamma) * ((self.gamma - 1.0) / x - 1.0) * L * L / self.integral
        return np.where((u >= 0.0) & (u <= 1.0), a, 0.0)

    # -- physical units --
    def position(self, t):
        return self._S(np.asarray(t, dtype=np.float64) / self.duration)

    def velocity(self, t):
        return self.amplitude / self.duration * self._V(np.asarray(t, dtype=np.float64) / self.duration)

    def acceleration(self, t):
        return self.amplitude / self.duration ** 2 * self._A(np.asarray(t, dtype=np.float64) / self.duration)

    @property
    def u_peak(self) -> float:
        if self.kind == "minimum_jerk":
            return 0.5
        return (self.gamma - 1.0 - self.xs) / (self.xe - self.xs)

    @property
    def skewness(self) -> float:
        """t_peak / D (0.5 = symmetric)."""
        return self.u_peak

    @property
    def peak_velocity(self) -> float:
        """Analytic peak velocity (rad/s)."""
        return float(self.velocity(self.u_peak * self.duration))

    @property
    def shape_factor(self) -> float:
        """V_peak * D / A (1.875 for minimum-jerk)."""
        return self.peak_velocity * self.duration / self.amplitude

    @property
    def residuals(self) -> dict:
        """Model minus configured target (rad/s, s, -) -- what the gamma-range clamp leaves."""
        out = {}
        if "peak_velocity" in self.targets:
            out["peak_velocity"] = self.peak_velocity - self.targets["peak_velocity"]
        if "skewness" in self.targets:
            out["skewness"] = self.skewness - self.targets["skewness"]
        if "duration" in self.targets:
            out["duration"] = self.duration - self.targets["duration"]
        return out


@functools.lru_cache(maxsize=4096)
def make_saccade(amplitude_rad: float, kind: str = "gamma_skewed", main_sequence: MainSequence = BALOH_1975,
                 skew: SkewModel = SkewModel(), duration_s: float | None = None) -> SaccadeProfile:
    """Saccade profile for amplitude A (rad). Duration from the main sequence unless `duration_s` is given
    (used for microsaccades). For 'gamma_skewed' the peak velocity follows the main sequence too."""
    if kind not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {kind!r}")
    A = float(amplitude_rad)
    if not (math.isfinite(A) and A > 0.0):
        raise ValueError(f"saccade amplitude must be finite and > 0, got {A}")
    D = float(main_sequence.duration(A)) if duration_s is None else float(duration_s)
    if not (math.isfinite(D) and D > 0.0):
        raise ValueError(f"duration must be > 0, got {D}")
    targets = {"duration": D}
    if kind == "minimum_jerk":
        if main_sequence.vp_max_rad_s is not None and duration_s is None:
            targets["peak_velocity"] = float(main_sequence.peak_velocity(A))
        targets["skewness"] = float(skew.target(D))
        return SaccadeProfile("minimum_jerk", A, D, targets=targets)
    if main_sequence.vp_max_rad_s is None and duration_s is None:
        raise ValueError("gamma_skewed needs a main sequence with a peak-velocity fit")
    vp = float(main_sequence.peak_velocity(A))
    s_t = float(skew.target(D))
    targets.update(peak_velocity=vp, skewness=s_t)
    g, eps = gamma_shape_params(vp * D / A, s_t)
    _, _, xs, xe, I = _shape(g, eps)
    return SaccadeProfile("gamma_skewed", A, D, gamma=g, eps=eps, xs=xs, xe=xe, integral=I, targets=targets)
