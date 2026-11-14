# Five-direction validation report

Result: **5/5 PASS**

Pupil displacement relative to the neutral (reset) render, in pixels of the 1024x1024 image (x right, y down). Each eye is measured by projection of the 3-D pupil centre (`proj`) and by independent dark-blob detection in the PNG (`det`); both must move the expected way by >= 2.0 px with off-axis drift < 50 % of the on-axis shift.

| Test | Input (h, v) | Expected | Left eye proj dx, dy | Left eye det dx, dy | Right eye proj dx, dy | Right eye det dx, dy | Left | Right | Camera locked | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| LEFT | (-1, +0) | x < 0 | -6.73, +0.00 | -6.84, +0.00 | -6.90, +0.00 | -6.95, +0.01 | PASS | PASS | True | **PASS** |
| RIGHT | (+1, +0) | x > 0 | +6.90, +0.00 | +6.93, +0.01 | +6.73, +0.00 | +6.81, +0.05 | PASS | PASS | True | **PASS** |
| UP | (+0, +1) | y < 0 | +0.01, -6.82 | -0.03, -7.00 | -0.01, -6.82 | -0.03, -7.05 | PASS | PASS | True | **PASS** |
| DOWN | (+0, -1) | y > 0 | +0.01, +6.82 | -0.07, +6.88 | -0.01, +6.82 | -0.03, +6.92 | PASS | PASS | True | **PASS** |

## Reset

- `reset()` called 3x after `move(0.6, -0.8)`; max eye-bone quaternion difference vs neutral: 0.0e+00
- projected pupil shift vs neutral: L 0.0e+00 px, R 0.0e+00 px
- detected pupil shift vs neutral: L 0.0 px, R 0.0 px
- re-rendered image vs neutral render (8-bit): max abs diff 0.0, mean 0.0
- Status: **PASS**
