# GazeForge — internal CV & mathematics audit (v1 geometry)

Package audited: a frozen copy of the v1 package (read-only; `diff -rq` afterwards was empty).
v2 keeps this geometry unchanged — its ground truth is identical to v1 (max diff 0.0, acceptance test H) —
so every geometric verdict below applies to v2. The image-detection numbers are the v2 values.
Tooling: Blender 4.2.23 LTS as the `bpy` module (`python3.11 (bpy 4.2)`), numpy 2.4.
All gaze and projection maths was recomputed in numpy (`numeric_audit.py`, which uses no mathutils for maths).
bpy was used only to read the scene, plus one depsgraph evaluation of poses computed by the script.
No renders were launched. Image checks use the shipped PNGs, read by the script's own zlib PNG decoder.

## Summary

| # | Item | Verdict |
|---|------|---------|
| 1 | Fixation target geometry (T = F0 + h·0.05·r + v·0.05·u, 35 cm plane, bounds) | **OK** |
| 2 | Eye-centre coordinates, IPD, midpoint | **OK** |
| 3 | Binocular rotations (10 sequence frames + 4 extremes + neutral), rays through T | **OK** |
| 4 | Quaternion conjugation q_local = q_rest⁻¹·q_world·q_rest, roll, eyeball world rotation | **OK** |
| 5 | Camera K, R, t, P; 0.40 m distance; M → image centre | **OK** |
| 6 | Pixel-coordinate convention (corner origin, cx = 512) vs OpenCV (cx = 511.5) | **doc-fix** (internally consistent; users are not warned) |
| 7 | Pupil-centre 3-D/2-D labels and image re-detection | **OK** (independent detector: mean 0.11 px, max 0.26 px over 20 pupils) |
| 8 | "Shortest arc = Listing's law, zero torsion" | **doc-fix** (overclaim: L1 holds exactly; torsion is zero only in Listing coordinates; L2/cyclovergence not modelled) |
| 9 | Other conventions and README numbers | **doc-fix** (azimuth range sign, Euler semantics, eyelid 0.7→0.8, cornea apex 13→13.7 mm, 6.91→6.90 rounding) |

`numeric_audit.py`: **47/47 checks PASS** (full output at the end).
**No mathematical or geometric error was found.** All findings are documentation issues.

---

## 1. Fixation target geometry — OK

*Claim (README §8.4–8.6, §9).* F0 = M + 0.35·f and T = F0 + h·0.05·r + v·0.05·u, with |h|, |v| ≤ 1.

*Method.* M comes from the bone heads (item 2), and f, r, u from the calibration JSON. I checked f, r, u against the
head bone's rest frame: bone +Z = (0,−1,0) = f and bone +Y = (0,0,1) = u. They are orthonormal with r × u = f.
I recomputed T for each record and compared it with `extra.fixation_target`.

*Numbers.* The calibration constants are exactly 0.35, 0.40, 0.05 and 0.05 m.
Max |T_numpy − T_meta| = 2.2e-8 m (float32 storage).
(T − M)·f − 0.35 = 0 (max |dev| 0.0 m), so T lies in the 35 cm plane.
Max per-axis offset = 5.0000 cm. The corners of the input square are 7.07 cm from F0 (radial).
The shipped 10-frame sequence lies on the 5 cm circle.

## 2. Eye centres — OK

*Method.* I read the eye centres three ways:
- from the bone heads (`rig.matrix_world @ bone.matrix_local`; the rig matrix is identity and the head pose is rest);
- from the `Eyeball.L/R` world origins;
- from `calibration_json`.

*Numbers.* C_L = (+0.0293356, −0.1263118, 0.8034567) m and C_R is its mirror (asymmetry 0).
Bone vs calibration: max 7.0e-9 m. Eyeball origin vs bone head: 0.0 m.
M = (0, −0.1263118, 0.8034567) matches `eye_midpoint`.
IPD = 58.671 mm (README "58.7 mm" ✓).

## 3. Binocular rotations — OK

*Method.* For each eye I computed d = (T − C)/|T − C| and the primary axis a = bone +Y in world = (0,−1,0).
q = (cos θ/2, sin θ/2 · (a×d)/|a×d|) with θ = atan2(|a×d|, a·d), which is plain Rodrigues and uses no mathutils.
I compared this with:
- `extra.eye_rotation_quat_wxyz_*`;
- the Euler triple `eye_rotation_*`, rebuilt as R = Rz(rz)·Ry(ry)·Rx(rx).

I also checked each gaze ray R·a from C against T.

*Numbers.*
- Quaternion error: max 1.42e-7 rad. Euler error: max 1.69e-7 rad (float32).
- Reading the same Euler triple as Rx·Ry·Rz would be wrong by up to 1.08°, so the documented order matters and is correct.
- Gaze-ray miss distance: 5.6e-17 m (numpy) and ≤ 9.4e-8 m in metadata (float32 scene).
- The package `move()` matches numpy to 1.38e-7 rad for all 15 poses.
- Re-running the package `metadata()` reproduces `metadata.json` exactly (max diff 0).

Asymmetry between the eyes (deg; + yaw = towards image right; adduction = nasal rotation):

| pose | yaw L | yaw R | adduction L / R | elevation (Fick) L / R | vergence |
|---|---|---|---|---|---|
| neutral (0,0) | −4.791 | +4.791 | 4.791 / 4.791 | 0 / 0 | 9.582 |
| LEFT (−1,0) | −12.772 | −3.379 | **12.772 / −3.379** | 0 / 0 | 9.393 |
| RIGHT (+1,0) | +3.379 | +12.772 | **−3.379 / 12.772** | 0 / 0 | 9.393 |
| UP (0,+1) | −4.791 | +4.791 | 4.791 / 4.791 | +8.102 / +8.102 | 9.486 |
| DOWN (0,−1) | −4.791 | +4.791 | 4.791 / 4.791 | −8.102 / −8.102 | 9.486 |
| frame 2 (+0.951,+0.309) | +2.980 | +12.390 | −2.980 / 12.390 | +2.524 / +2.469 | 9.402 |

At full horizontal deflection, the abducting eye turns 3.38° from primary while the adducting eye turns 12.77°.
Relative to the converged neutral pose that is 8.17° and 7.98°, which is why horizontal pupil travel is 6.73 vs 6.90 px.
For asymmetric targets the Fick elevations of the two eyes differ by up to 0.090°, while the Helmholtz elevations
are identical (8.9e-16°): both eyes lie in a common plane of regard, as geometry requires.

## 4. Quaternion conjugation / world→local — OK

*Method.*
- **Algebra:** for 200 random rest frames, including arbitrary roll, I checked that a pose bone with
  R_rest·R(q_local), where q_local = q_rest⁻¹ q_world q_rest, equals R(q_world)·R_rest.
  This is exact (8.9e-16). The reversed order (q_rest q_world q_rest⁻¹) is wrong (error 1.98).
- **This rig:** the eye-bone rest matrix is a 180° rotation about world Z: bone +Y = −Y world (primary gaze),
  bone X = −X world, bone Z = +Z, roll 0, with 8.7e-8 float residue. Because a 180° rotation is an involution,
  both conjugation orders give the same result here, so the rig cannot distinguish them. The algebraic test confirms
  the order in `eyes.py` is the generally correct one, including for non-zero roll.
  `_bone_rest_world = rig_mw @ parent.matrix @ parent_rest⁻¹ @ bone_rest` is the standard rest-under-posed-parent formula.
- **Blender evaluation:** I wrote my own numpy q_local into the pose bones, evaluated the depsgraph and read back the
  `Eyeball.*` world matrices for 15 poses × 2 eyes:

| Quantity | Max error |
|---|---|
| Eyeball world rotation vs numpy R(q_world) | 1.16e-7 rad |
| Eyeball centre displacement | 0.0 m (pure rotation about the eyeball's own centre) |
| `Pupil` origin vs C + R·[0, −10.690 mm, 0] | 1.9e-8 m |

- At rest the Eyeball world rotation is identity (8.7e-8 rad), so "eyeball world rotation = rotation from primary
  position" holds.

## 5. Camera projection — OK

*Method.* I built K from the camera data:
- lens 62 mm, sensor 36 × 24 mm, fit HORIZONTAL, shift 0/0;
- 1024 × 1024 at 100 %, pixel aspect 1:1, perspective;
- fx = fy = 62/36 · 1024 = 1763.5556 px, cx = cy = 512.

R and t come from `RenderCamera.matrix_world`, converted from Blender camera axes (x right, y up, looking −z)
to OpenCV axes with diag(1, −1, −1). P = K[R|t].

*Numbers.*
- K vs `camera_intrinsics_K`: 1.4e-5 px.
- Reprojection with my P vs metadata `camera_matrix`: max 8.1e-4 px.
- Reprojection with my P vs `bpy_extras.world_to_camera_view`·(W, H) with y flipped: max 8.1e-4 px.
- `camera_matrix` is bit-identical in all 10 records.
- Camera centre − M = 0.400000036 m along f, lateral 0.0 m (float32: 36 nm).
- The optical axis is −f, image up is u and image right is r.
- M projects to (512.000, 512.000077); the 7.7e-5 px comes from the float32 value of cos(π/2) in the camera rotation.

## 6. Pixel-coordinate convention — doc-fix

*Relation.* The metadata, `project()` and P use **continuous corner-origin coordinates**.
The image spans [0, 1024] × [0, 1024], pixel (i, j) covers [i, i+1) × [j, j+1), and its centre is at (i + 0.5, j + 0.5).
In OpenCV's convention pixel centres are integers, so for any point:

  u_cv = u_meta − 0.5, v_cv = v_meta − 0.5, i.e. K_cv = [[1,0,−0.5],[0,1,−0.5],[0,0,1]]·K (cx = cy = 511.5), P_cv = that matrix · P.

If a user feeds `camera_matrix` to OpenCV (`projectPoints`, `solvePnP`, calibration comparison) and compares with
integer-indexed detections (moments, `argmax`, `findContours`), they get a systematic +0.5 px bias on both axes.
That is comparable to the label accuracy itself.

*Consistency inside the package.*
- `detect_pupil_in_image` adds +0.5 (the ROI test uses `xx + 0.5`, and the centroid adds `+ x0 + 0.5`).
  Tested on 25 anti-aliased synthetic discs with known corner-origin centres, its mean bias is (−0.003, −0.008) px.
  A missing +0.5 would give −0.5. Its max single error of 0.147 px is binary-threshold quantisation.
- The renders follow the same convention. With my own detector, the mean signed image − label offset over the 20
  sequence pupils is dx = +0.008 px (L −0.007, R +0.024) and dy = −0.053 px.
  The two eyes are mirror images about x = 512, so a half-pixel convention error would appear as a same-sign ≈0.5 px
  dx in both eyes. It does not.

*Verdict.* The maths is correct and consistent. The documentation must warn OpenCV users.
The phrase "OpenCV axes" in README §10 and in `extra.camera_matrix_convention` refers only to the camera axis
directions, but readers will easily take it to mean the OpenCV pixel convention.

## 7. Pupil-centre projection — OK

*Method.* P3 = C + R·[0, −d, 0] with d = `pupil_offset_local_m` = 10.690 mm, projected with my P.

*Numbers.* 3-D vs `extra.pupil_center_3D_*`: max 1.9e-8 m. 2-D vs `pupil_center_2D_*`: max 1.7e-4 px.

Projected pupil travel from neutral at the extremes:

| Direction | Left eye (px) | Right eye (px) |
|---|---|---|
| LEFT (−1, 0) | dx −6.731 | dx −6.905 |
| RIGHT (+1, 0) | dx +6.905 | dx +6.731 |
| UP (0, +1) | dy −6.822 | dy −6.822 |
| DOWN (0, −1) | dy +6.822 | dy +6.822 |

*Own image detector.* It is a different method from the package's, so it is not biased the same way:
- coarse: the minimum of a pupil-sized box filter inside an iris-radius ROI centred on the projected **eyeball** centre,
  which does not depend on gaze;
- fine: 120 rays from 0.45·r_p outward, with a per-ray half-way threshold between the pupil level and the level just
  outside the edge, giving bilinear sub-pixel edge points;
- an algebraic circle fit, trimmed to the best-fitting 70 % of rays (8 iterations) so that the dark lid margin in
  up-gaze is rejected;
- on synthetic discs: bias (−0.0004, +0.0033) px, max 0.06 px.

*Image vs label error, 20 sequence pupils:* mean **0.114 px**, median 0.098, RMS 0.137, max **0.262 px**
(frame 0 R; the up-gaze frames 0, 1 and 9 show −0.15…−0.26 px in y; all other frames are ≤ 0.14 px).
On the direction-test renders (±1 extremes and neutral) the max is 0.262 px (UP R).

My detector and the package's `extra.qa` detections differ by mean 0.23 px and max 0.58 px.
The package detector's own numbers (max 0.84, mean 0.30 px, confirmed from `extra.qa`) are larger mainly in up-gaze,
where its binary threshold picks up the adjacent lid shadow. In the direction test its UP detection is 7.48 px
against 6.82 px projected. The README's numbers are accurate for its detector, and the labels themselves are better
than that (≤ 0.26 px).

## 8. Listing's law — doc-fix (overclaim)

**(a) Axis in Listing's plane.** For all 20 metadata quaternions and the 5 extremes/neutral, the max |unit axis · primary axis|
is **4.4e-8**, i.e. zero to float32 precision. Every orientation is a rotation from the straight-ahead primary
position about an axis in the plane perpendicular to it. That is exactly Listing's law **L1** (single, head-fixed
Listing's plane, primary position = head forward), applied to each eye independently.

**(b) What "torsion" means.**
- "Zero torsion" holds only in Listing / rotation-vector coordinates: the rotation vector has no component along the primary axis.
- In Fick (horizontal about the head vertical, then vertical, then torsion) or Helmholtz (vertical about the head
  interaural axis, then horizontal, then torsion) gimbal coordinates, the same orientations carry the well-known
  geometric **"false torsion"** of oblique (tertiary) positions.
- Here: max |Fick torsion| **0.542°** (e.g. frame 1 R eye: yaw 9.53°, elevation 6.50°) and max |Helmholtz torsion|
  **0.546°** (opposite sign). At v = ±1 both eyes carry ∓0.34° Fick torsion because the converged eyes are 4.79° adducted.
- Pure horizontal or pure vertical (from primary) positions have zero torsion in all systems.
- It also matters what the metadata's Euler `ry` is: it is **not** torsion. It reaches ±0.54° although the Listing torsion is exactly 0 (item 9).

**(c) Binocular, near-fixation setting.**
- Mok, Ro, Cadera, Crawford & Vilis (1992, *Vision Res.* 32:2055–2064) found that during convergence each eye still
  obeys a planar Listing's law, but the planes are "rotated temporally". This produces "extorsion of the two eyes for
  downward gaze and intorsion for upward gaze".
- Their quantitative result: the displacement planes rotate ≈4.7° on a 30° isovergence surface (15° per eye), and
  Listing's plane and primary position rotate by twice that (≈9.4°).
- This binocular extension (L2) was formalised by van Rijn & van den Berg (1993, *Vision Res.* 33:691–708,
  "Listing's law extended to include eye vergence"). Tweed (1997, *Vision Res.*, pp. 1939–1951, doi:10.1016/S0042-6989(97)00002-3)
  analysed why L2 arises, attributing it to a combined stereo/motor optimisation.
- Bruno & Van den Berg (1997, *Exp. Brain Res.* 117:251–265) summarise that during convergence the eyes intort in
  up-gaze and extort in down-gaze, with cyclovergence roughly linear in convergence × elevation, though "different
  amounts of cyclovergence have been reported".
- The model uses one fixed primary position (−Y) per eye regardless of vergence, so it does **not** implement L2.
- I estimated the omitted effect with an L2 model in which each Listing's plane is rotated temporally by k·(vergence/2),
  with k = 2·4.7/15 = 0.627 from Mok et al. As a sanity check, this model reproduces Mok's ≈2.5° extorsion
  (2.59° predicted) at 30° down / 15° per eye.
- For this rig (vergence ≈ 9.4–9.6°, Listing-plane rotation ≈3.0°/eye), the missing torsion is ±0.21° per eye at
  v = ±1 and **±0.42° cyclovergence**: excyclo in down-gaze (frame 5), incyclo in up-gaze (frame 0).
  It is ±0.13° in frames 2/3/7/8 and zero for purely horizontal gaze. The effect scales linearly with k, which varies between studies.

**What the model does:** each eye rotates about its own centre by the minimum-angle rotation from a common,
head-fixed primary position (straight ahead). This satisfies monocular Listing's law (L1) exactly, with zero
Listing torsion. Both optical axes intersect exactly at T. The optical axis is treated as the line of sight.

**What it does not do:**
- vergence-dependent rotation of Listing's planes (L2) and the resulting cyclovergence (≈0.4° at the extremes here);
- an individual primary position that differs from straight ahead;
- angle kappa / visual axis;
- the torsional components of real fixation (ocular counter-roll, noise).

The README sentence presents shortest arc as being Listing's law for this binocular converged setting, with "zero
torsion about the line of sight". That overclaims: it is correct for far-viewing L1, not for near binocular fixation.
The magnitudes involved are small (< 0.5° here), so this is a documentation issue, not a label error.

## 9. Other checks

| Claim | Recomputed | Verdict |
|---|---|---|
| "converged eyes start 4.79° inwards" (§8.6, §11) | 4.7911° | OK |
| "±8.1° of elevation" (§8.6, §13.1) | 8.102° at (0,±1) (Fick and asin(g·u)); max over the whole input square 8.130° | OK |
| "up to 12.8° of eye-in-head azimuth" (§8.6) | 12.772° | OK |
| "−12.8°…+3.4° of per-eye azimuth" (§13.1) | L eye −12.772…+3.379; **R eye −3.379…+12.772** (+ = image right) | **doc-fix** (signed range true for L only; should say adduction/abduction) |
| "≈7 px" / "6.7–6.9 px, 6.8 px" (§13.1, results table) | 6.731 / 6.905 px horizontal, 6.822 px vertical | OK |
| §11 table "dx −6.91 / +6.91" (projected) | 6.9046 px → 6.90 (double rounding via 6.905) | cosmetic |
| fx = fy = 1763.56 px, cx = cy = 512 (§8.7) | 1763.5556 / 512 | OK, but see item 6 |
| IPD 58.7 mm (§8.1) | 58.671 mm | OK |
| Cornea "apex 13 mm from the rotation centre" (§6) | `corneal_apex_from_centre_m` = 13.718 mm | **doc-fix** |
| Eyelid targets "`eye-height1/2-incr`, 0.7" (§12) | `prepare_basemesh.py` lines 37–40 and `asset_source` in the calibration: **0.8** (§13.2 also says 0.8) | **doc-fix** |
| Euler XYZ "R = Rz·Ry·Rx, X applied first" (§7) | Confirmed (1.7e-7 rad); equals SciPy `from_euler('xyz', …)` (4e-8) | OK, but semantics undocumented |
| Euler sign semantics | rx < 0 ⇒ gaze up; rz > 0 ⇒ gaze towards image right (+X); ry ≠ torsion (up to ±0.539°) | **doc-fix** (add) |
| +h → image right, +v → image up (§7) | RIGHT: both pupils dx > 0; UP: both dy < 0 (image y down) | OK |
| Gaze-ray miss "~10⁻⁸ m", "< 1 µm" (§9, table) | 2.8e-8 … 9.4e-8 m | OK |
| Calibration self-check "> 10 µm" (§8.8) | `_check_calibration` tol = 1e-5 m | OK |
| Input validation [−1, 1] | Executed: 1.0000001, −1.5, NaN, inf → ValueError; bool / str → TypeError; ±1 and 1+5e-10 accepted (clamped) | OK |

---

## Discrepancies found

All are documentation / wording issues. No numerical label, rotation or projection error was found.

- **D1 (Listing overclaim)**: README §9 "*Why shortest arc*" paragraph, the `eyes.py` module docstring (lines 20–21)
  and the `eyes.py` line 308 comment equate shortest arc with Listing's law for this converged binocular setting and
  claim "zero torsion" without qualification. Fick/Helmholtz torsion reaches 0.54–0.55°, and L2 cyclovergence of
  ≈0.4° is not modelled.
- **D2 (pixel convention)**: README §7, §8.7, §10 and `extra.camera_matrix_convention` do not warn that cx = cy = 512 is
  the corner-origin convention (OpenCV: 511.5). Using P with integer-indexed detections gives a +0.5 px bias. "OpenCV axes" invites that misreading.
- **D3**: README §13.1 "−12.8°…+3.4° of per-eye azimuth" is the left eye's signed range only.
- **D4**: README §12 says eyelid targets 0.7; the asset script and calibration say 0.8 (as does §13.2).
- **D5**: README §6 cornea apex "13 mm"; actual 13.72 mm.
- **D6**: README §7 does not state the sign meaning of the Euler components, nor that the middle angle (ry) is not ocular torsion.
- **D7 (cosmetic)**: README §11 table "6.91" should be 6.90 (projected 6.9046 px).

## Corrected documentation text

**D1 — README §9.** Original:
> *Why shortest arc*: the rotation that takes the primary-position axis to a gaze direction with **zero torsion** about the line of sight is exactly the one whose rotation axis lies in the plane perpendicular to the primary axis — **Listing's law**, which human eyes obey for fixations from primary position.

Replacement:
> *Why shortest arc*: the shortest-arc rotation from the primary (straight-ahead) position has its rotation axis in the plane perpendicular to the primary axis, so every eye orientation lies in one head-fixed Listing's plane — the model implements the monocular form of **Listing's law (L1)** for each eye, with primary position = head forward. "Zero torsion" holds in Listing's (rotation-vector) sense; expressed in Fick or Helmholtz angles the same oblique orientations carry the usual geometric "false torsion" (up to 0.54° in the shipped sequence). The model does **not** implement the binocular extension of Listing's law (L2): during convergence real Listing's planes rotate temporally (Mok et al., 1992; van Rijn & van den Berg, 1993), giving relative extorsion in down-gaze and intorsion in up-gaze at near. For this rig's ≈9.5° vergence the omitted cyclovergence is estimated at ≈0.4° at v = ±1 (scaled from Mok et al.'s data); angle kappa and individual primary positions are also not modelled.

**D1 — `eyes.py` docstring, lines 20–21.** Original:
> `Each eye independently: q = shortest-arc rotation taking its primary-position optical axis onto (T - C_eye). Shortest-arc from primary position == Listing's law (zero ocular torsion).`

Replacement:
> `Each eye independently: q = shortest-arc rotation taking its primary-position optical axis onto (T - C_eye). Shortest arc from a fixed straight-ahead primary position == monocular Listing's law (L1; zero torsion in Listing coordinates). Vergence-dependent L2 cyclovergence is not modelled.`

**D1 — `eyes.py` line 308 comment.** Original: `# shortest arc: Listing's law, no torsion` → Replacement: `# shortest arc from primary: Listing's law L1 (zero Listing torsion)`

**D2 — README §7, "Image coordinates".** Original:
> * **Image coordinates**: pixels, origin at the **top-left corner** of the image, x right, y **down**; pixel *i* spans [i, i+1). Image centre = (512, 512).

Replacement:
> * **Image coordinates**: pixels, origin at the **top-left corner** of the image, x right, y **down**; pixel *i* spans [i, i+1), so the centre of pixel (i, j) is at (i + 0.5, j + 0.5) and the image centre is (512, 512). **This differs from the OpenCV / integer-index convention** (pixel centres at integers): to convert any 2-D value in this package to OpenCV pixel coordinates subtract 0.5 from x and y — equivalently use cx = cy = 511.5, i.e. P_cv = [[1, 0, −0.5], [0, 1, −0.5], [0, 0, 1]] · P. `detect_pupil_in_image` already returns corner-origin coordinates.

**D2 — README §10, `camera_matrix` bullet.** Original:
> * `camera_matrix` — the **3×4 projection matrix P = K [R | t]** in OpenCV axes (camera x right, y down, z forward). `[x, y, 1]ᵀ ∝ P [X, Y, Z, 1]ᵀ` maps a world point to the pixel coordinates used for `pupil_center_2D_*` (top-left-corner origin).

Replacement:
> * `camera_matrix` — the **3×4 projection matrix P = K [R | t]** with OpenCV camera *axes* (x right, y down, z forward). `[x, y, 1]ᵀ ∝ P [X, Y, Z, 1]ᵀ` maps a world point to the pixel coordinates used for `pupil_center_2D_*`: origin at the top-left image **corner**, pixel centres at half-integers (cx = cy = 512). For OpenCV's pixel-centre convention (cx = cy = 511.5) subtract 0.5 from both image coordinates, or use [[1, 0, −0.5], [0, 1, −0.5], [0, 0, 1]] · P.

**D2 — README §8.7.** Original: "→ fx = fy = 1763.56 px, cx = cy = 512 px, no skew, no distortion." → Replacement: "→ fx = fy = 1763.56 px, cx = cy = 512 px in corner-origin pixel coordinates (= 511.5 in OpenCV's integer-centre convention), no skew, no distortion."

**D2 — `extra.camera_matrix_convention` string (eyes.py `metadata()`).** Append: `"; pixel centres are at half-integers (corner origin): subtract 0.5 from x and y for OpenCV integer-centre pixel coordinates (cx = cy = 511.5)"`.

**D3 — README §13.1.** Original:
> ±5 cm at 35 cm is ±8.1° of elevation and −12.8°…+3.4° of per-eye azimuth, i.e. ≈7 px of pupil travel at 1024².

Replacement:
> ±5 cm at 35 cm is ±8.1° of elevation and, per eye, up to 12.8° of adduction and 3.4° of abduction (signed azimuth, + = image right: left eye −12.8°…+3.4°, right eye −3.4°…+12.8°), i.e. 6.7–6.9 px of pupil travel at 1024².

**D4 — README §12.** Original: "eyelid aperture targets (`eye-height1/2-incr`, 0.7)." → Replacement: "eyelid aperture targets (`eye-height1/2-incr`, 0.8)."

**D5 — README §6, Cornea row.** Original: "spherical cap (radius of curvature 7.8 mm, apex 13 mm from the rotation centre)" → Replacement: "spherical cap (radius of curvature 7.8 mm, apex 13.7 mm from the rotation centre)".

**D6 — README §7, "Eye rotations".** Original:
> * **Eye rotations** (`eye_rotation_L/R`): rotation of each eyeball **in the world frame relative to its primary (straight-ahead) position**, exported as Euler angles in radians, Blender order `XYZ` (R = Rz·Ry·Rx, X applied first). Quaternions (w, x, y, z) are also exported in `extra`.

Replacement (append to the original):
> … also exported in `extra`. R = Rz·Ry·Rx is SciPy's `Rotation.from_euler('xyz', [rx, ry, rz])` (lower-case = extrinsic); reading the triple as intrinsic X-Y-Z (Rx·Ry·Rz) is wrong by up to 1.1°. In world axes rx < 0 means gaze up and rz > 0 means gaze towards image right (world +X, the subject's left). The middle angle ry is a property of the Euler decomposition, **not ocular torsion** (it reaches ±0.54° in the sequence although the Listing torsion is exactly zero); use the quaternion, or `extra.gaze_dir_*` / `extra.yaw_deg_*` / `extra.pitch_deg_*`, for gaze direction.

**D7 — README §11 table.** "dx −6.91 / −7.05" (LEFT, right eye) → "dx −6.90 / −7.05"; "dx +6.91 / +6.97" (RIGHT, left eye) → "dx +6.90 / +6.97".

## Confirmation that the original specification remains satisfied

| Requirement | Evidence (this audit) | Status |
|---|---|---|
| Nominal fixation 35 cm in front of the eye midpoint | (T − M)·f = 0.35 m exactly for all records; F0 = M + 0.35·f | ✓ |
| Camera 40 cm in front of the eye midpoint | 0.400000036 m along f, lateral 0, optical axis through M (M → (512, 512)) | ✓ |
| ±5 cm motion bound | per-axis offset ≤ 5.0000 cm; sequence on the 5 cm circle | ✓ |
| Inputs h, v ∈ [−1, 1] | out-of-range / NaN / inf / non-numbers rejected; ±1 accepted (executed) | ✓ |
| θk = 2πk/10, h = sin θk, v = cos θk, clockwise from top | all 10 records exact to 1e-12; frame 0 = (0, 1) | ✓ |
| Each eye rotates about its own centre | Blender-evaluated eyeball centre displacement 0.0 m over 15 poses | ✓ |
| Both eyes converge on the target | numpy miss 5.6e-17 m; scene/metadata ≤ 9.4e-8 m | ✓ |
| 1024 × 1024 render | resolution 1024 × 1024 @ 100 %, square pixels; PNGs decoded at 1024 wide | ✓ |
| Camera fixed | `camera_matrix` bit-identical in all 10 records; transform locks; P reproduced | ✓ |
| Metadata labels correct | rotations ≤ 1.7e-7 rad, pupils ≤ 1.7e-4 px vs independent numpy; image re-detection mean 0.11 px, max 0.26 px | ✓ |

## What could not be verified

- Cycles' internal pixel-sample placement directly, because no renders were allowed. The corner-origin convention of the
  PNGs is supported only empirically (mean signed image − label offset +0.008 / −0.053 px over 20 pupils, mirror-symmetric eyes).
- The sphere-fit residual "< 0.6 mm" and the MakeHuman provenance (§8.1, §12); the raw helper geometry was not re-fitted.
- Cross-version claims (4.5 LTS, 5.0); only Blender 4.2.23 was used.
- Render-dependent claims (re-render diff 0/255, timings) and the acceptance tests A–I, which were not re-run.
- The magnitude of L2 cyclovergence. My estimate uses the Mok et al. (1992) coefficient only. For van Rijn & van den
  Berg (1993) and Tweed (1997) I could access only bibliographic data, an abstract paraphrase, and a thesis summary,
  not their coefficients. Reported cyclovergence magnitudes vary between studies (Bruno & Van den Berg 1997).

## Literature consulted (only what was actually retrieved)

- Mok D, Ro A, Cadera W, Crawford JD, Vilis T (1992). Rotation of Listing's plane during vergence. *Vision Res* 32(11):2055–2064. (full text PDF read; quotes above)
- Van Rijn LJ, Van den Berg AV (1993). Binocular eye orientation during fixations: Listing's law extended to include eye vergence. *Vision Res* 33(5–6):691–708, doi:10.1016/0042-6989(93)90189-4. (bibliographic record + van Rijn's 1994 thesis)
- Tweed D (1997). Visual-motor optimization in binocular control. *Vision Res* pp. 1939–1951, doi:10.1016/S0042-6989(97)00002-3. (abstract paraphrase)
- Bruno P, Van den Berg AV (1997). Torsion during saccades between tertiary positions. *Exp Brain Res* 117:251–265. (full text read; summary of cyclovergence literature)

---

## Actual output of `numeric_audit.py`

Command: `python3.11 (bpy 4.2) numeric_audit.py <package> --json numeric_audit_summary.json` (exit code 0, ≈4 s)

```text
Read blend: "<package>/scene.blend"

== 2. Eye centres ==
PASS  eye centres: bone heads == calibration JSON  max 6.99e-09 m
PASS  eye centres: Eyeball.L/R world origin == bone heads  max 0.00e+00 m
PASS  midpoint == calibration eye_midpoint  M = [0.0, -0.1263118, 0.8034567]
PASS  IPD (|C_L - C_R|) == 58.7 mm (README)  IPD = 58.671 mm
PASS  eye centres mirror-symmetric about x = 0, same y/z  asym 0.0e+00 m
PASS  rig world matrix identity-like & head pose = rest  
PASS  head frame: cal axes orthonormal (r x u = f) and match the head bone (+Z = f, +Y = u)  f=[0.0, -1.0, 0.0] r=[1.0, 0.0, 0.0] u=[0.0, 0.0, 1.0] ; head-bone +Z=[0.0, -1.0, 0.0] +Y=[0.0, 0.0, 1.0]
PASS  bone +Y (primary axis) == head forward for both eyes  A_L=[0.0, -1.0, 0.0] roll: bone X=[-1.0, 0.0, 0.0] Z=[0.0, 0.0, 1.0]

== 1. Fixation target ==
PASS  calibration constants are 0.35 / 0.40 / 0.05 / 0.05 m  
PASS  T(h,v) recomputed == metadata extra.fixation_target  max 2.21e-08 m
PASS  T lies in the 35 cm plane: (T-M).f = 0.35  max |dev| 0.00e+00 m
PASS  |offsets| <= 5 cm per axis  max |offset| 5.0000 cm

== 3. Binocular rotations ==
PASS  sequence: 10 frames, theta_k = 2*pi*k/10, h = sin, v = cos, frame 0 = top (h=0, v=1)  
PASS  numpy shortest-arc rotation == metadata quaternion (all 20 eyes)  max 1.42e-07 rad
PASS  numpy rotation == metadata Euler XYZ with R = Rz.Ry.Rx  max 1.69e-07 rad
      info: reading the same Euler triple as Rx.Ry.Rz would be wrong by up to 1.0776 deg
PASS  gaze rays (numpy) pass through T  max miss 5.59e-17 m
PASS  metadata gaze_ray_miss_m < 1 um  max 9.40e-08 m
      frame   h      v     yawL    yawR   pitchL  pitchR  addL   addR   |rotL|  |rotR|  verg  (deg)
        0 +0.000 +1.000  -4.791  +4.791  +8.102  +8.102  4.791  4.791   9.405   9.405  9.486
        1 +0.588 +0.809  +0.009  +9.525  +6.593  +6.503 -0.009  9.525   6.593  11.516  9.454
        2 +0.951 +0.309  +2.980 +12.390  +2.524  +2.469 -2.980 12.390   3.904  12.630  9.402
        3 +0.951 -0.309  +2.980 +12.390  -2.524  -2.469 -2.980 12.390   3.904  12.630  9.402
        4 +0.588 -0.809  +0.009  +9.525  -6.593  -6.503 -0.009  9.525   6.593  11.516  9.454
        5 +0.000 -1.000  -4.791  +4.791  -8.102  -8.102  4.791  4.791   9.405   9.405  9.486
        6 -0.588 -0.809  -9.525  -0.009  -6.503  -6.593  9.525 -0.009  11.516   6.593  9.454
        7 -0.951 -0.309 -12.390  -2.980  -2.469  -2.524 12.390 -2.980  12.630   3.904  9.402
        8 -0.951 +0.309 -12.390  -2.980  +2.469  +2.524 12.390 -2.980  12.630   3.904  9.402
        9 -0.588 +0.809  -9.525  -0.009  +6.503  +6.593  9.525 -0.009  11.516   6.593  9.454
      LEFT    (-1,+0) yaw L -12.772 R -3.379 | pitch +0.000/+0.000 | adduction L +12.772 R -3.379 | vergence 9.393
      RIGHT   (+1,+0) yaw L +3.379 R +12.772 | pitch +0.000/+0.000 | adduction L -3.379 R +12.772 | vergence 9.393
      UP      (+0,+1) yaw L -4.791 R +4.791 | pitch +8.102/+8.102 | adduction L +4.791 R +4.791 | vergence 9.486
      DOWN    (+0,-1) yaw L -4.791 R +4.791 | pitch -8.102/-8.102 | adduction L +4.791 R +4.791 | vergence 9.486
      NEUTRAL (+0,+0) yaw L -4.791 R +4.791 | pitch +0.000/+0.000 | adduction L +4.791 R +4.791 | vergence 9.582
PASS  extremes: gaze rays through T  
PASS  README '4.79 deg inward' at neutral  4.7911 / 4.7911 deg
PASS  README '+-8.1 deg elevation' at (0,+-1)  UP 8.1021, DOWN -8.1021
      eye-in-head azimuth range: L eye [-12.772, +3.379]  R eye [-3.379, +12.772] deg (+ = image right)
PASS  README '-12.8...+3.4 deg per-eye azimuth' == 12.8 adduction / 3.4 abduction  max adduction 12.772, max abduction 3.379 deg; NOTE the signed range is mirrored for the R eye
      max |elevation| over the full input square: 8.130 deg (at h where the eye is straight below/above T)

== 4. Quaternion conjugation / bone space ==
PASS  Eyeball world rotation at rest pose == identity (so metadata = rotation from primary)  8.74e-08 rad
PASS  q_local = q_rest^-1 q_world q_rest is the correct conjugation (random rest frames incl. roll)  err 8.9e-16; reversed order err 1.98
      rig rest quaternions: L [0.0, 4.4e-08, 0.0, 1.0]  (180 deg about Z: both conjugation orders coincide for THIS rig, so the rig cannot distinguish them)
PASS  Blender-evaluated Eyeball world rotation == numpy q_world (24 eyes)  max 1.16e-07 rad
PASS  eyeball centres never translate (rotation about own centre)  max 0.00e+00 m
PASS  Pupil object origin == C + R.[0,-pupil_offset,0]  max 1.88e-08 m; offset 10.690 mm

== 5. Camera projection ==
      lens 62.0 mm, sensor 36.0x24.0 fit HORIZONTAL, shift (0.0,0.0), res 1024x1024@100%, pixel aspect 1.0:1.0, type PERSP
      K = [[1763.5556, 0.0, 512.0], [0.0, 1763.5556, 512.0], [0.0, 0.0, 1.0]]
PASS  resolution 1024 x 1024 @100%, square pixels, perspective, no shift  
PASS  fx = 62/36*1024 = 1763.556 px (README)  fx 1763.5556
PASS  rebuilt K == metadata camera_intrinsics_K  max 1.4e-05 px
PASS  rebuilt P == metadata camera_matrix (reprojection of test points)  max 8.1e-04 px
PASS  rebuilt P == bpy_extras world_to_camera_view * (W, H) with y flipped  max 8.1e-04 px
PASS  camera_matrix identical in all 10 records (camera fixed)  max diff 0.0
PASS  camera centre 0.40 m from M along head forward axis  along 0.400000036 m, lateral 0.0e+00 m, |d| 0.400000036
PASS  camera optical axis = -forward (looks back at the face), image up = head up  optical axis [-0.0, 1.0, 4e-08]
PASS  M projects to (512, 512) (corner-origin convention)  [512.0, 512.0000770874616]

== 6. Pixel-coordinate convention ==
      metadata/P use continuous coords with the image spanning [0,1024]; pixel (i,j) centre = (i+0.5, j+0.5).
      OpenCV/integer-index convention (pixel centre = integer): u_cv = u_meta - 0.5, v_cv = v_meta - 0.5,
      equivalently K_cv = K with cx, cy = 511.5. Using P as-is with integer pixel indices biases by +0.5 px.
PASS  own detector on 25 synthetic discs: |bias| < 0.02 px, max error < 0.1 px  bias (-0.0004,+0.0033) max 0.0595 px
PASS  package detect_pupil_in_image follows the same corner-origin (+0.5) convention (|bias| < 0.05 px)  bias (-0.0028,-0.0081) px, max single error 0.147 px (binary-threshold quantisation); a missing +0.5 would give a -0.5 bias

== 7. Pupil-centre projection ==
PASS  pupil 3-D (numpy) == metadata extra.pupil_center_3D  max 1.88e-08 m
PASS  pupil 2-D (numpy P) == metadata pupil_center_2D  max 1.73e-04 px
      projected pupil travel from neutral (dx, dy px): LEFT_L (-6.73,+0.00); LEFT_R (-6.90,-0.00); RIGHT_L (+6.90,-0.00); RIGHT_R (+6.73,+0.00); UP_L (+0.01,-6.82); UP_R (-0.01,-6.82); DOWN_L (+0.01,+6.82); DOWN_R (-0.01,+6.82)
      frame 0 L: proj ( 640.837, 505.178)  img ( 640.853, 504.928)  d=(+0.016,-0.250) |d|=0.251  r_p=9.03px
      frame 0 R: proj ( 383.163, 505.178)  img ( 383.159, 504.915)  d=(-0.004,-0.262) |d|=0.262  r_p=9.03px
      frame 1 L: proj ( 644.872, 506.441)  img ( 644.881, 506.283)  d=(+0.009,-0.158) |d|=0.158  r_p=9.03px
      frame 1 R: proj ( 387.141, 506.519)  img ( 387.206, 506.365)  d=(+0.065,-0.154) |d|=0.167  r_p=9.03px
      frame 2 L: proj ( 647.395, 509.867)  img ( 647.482, 509.832)  d=(+0.087,-0.035) |d|=0.094  r_p=9.03px
      frame 2 R: proj ( 389.573, 509.915)  img ( 389.679, 509.853)  d=(+0.106,-0.063) |d|=0.124  r_p=9.03px
      frame 3 L: proj ( 647.395, 514.133)  img ( 647.416, 514.149)  d=(+0.021,+0.016) |d|=0.026  r_p=9.03px
      frame 3 R: proj ( 389.573, 514.085)  img ( 389.635, 514.102)  d=(+0.062,+0.018) |d|=0.064  r_p=9.03px
      frame 4 L: proj ( 644.872, 517.559)  img ( 644.868, 517.552)  d=(-0.004,-0.007) |d|=0.008  r_p=9.03px
      frame 4 R: proj ( 387.141, 517.481)  img ( 387.153, 517.468)  d=(+0.012,-0.013) |d|=0.017  r_p=9.03px
      frame 5 L: proj ( 640.837, 518.822)  img ( 640.828, 518.926)  d=(-0.009,+0.103) |d|=0.104  r_p=9.03px
      frame 5 R: proj ( 383.163, 518.822)  img ( 383.146, 518.891)  d=(-0.017,+0.069) |d|=0.071  r_p=9.03px
      frame 6 L: proj ( 636.859, 517.481)  img ( 636.870, 517.581)  d=(+0.011,+0.100) |d|=0.101  r_p=9.03px
      frame 6 R: proj ( 379.128, 517.559)  img ( 379.184, 517.636)  d=(+0.056,+0.077) |d|=0.096  r_p=9.03px
      frame 7 L: proj ( 634.427, 514.085)  img ( 634.412, 514.163)  d=(-0.015,+0.079) |d|=0.080  r_p=9.03px
      frame 7 R: proj ( 376.605, 514.133)  img ( 376.606, 514.112)  d=(+0.001,-0.021) |d|=0.021  r_p=9.03px
      frame 8 L: proj ( 634.427, 509.915)  img ( 634.330, 509.819)  d=(-0.097,-0.097) |d|=0.137  r_p=9.03px
      frame 8 R: proj ( 376.605, 509.867)  img ( 376.578, 509.840)  d=(-0.027,-0.027) |d|=0.038  r_p=9.03px
      frame 9 L: proj ( 636.859, 506.519)  img ( 636.769, 506.318)  d=(-0.090,-0.201) |d|=0.220  r_p=9.03px
      frame 9 R: proj ( 379.128, 506.441)  img ( 379.109, 506.213)  d=(-0.019,-0.229) |d|=0.229  r_p=9.03px
      |error| px: mean 0.114  median 0.098  RMS 0.137  max 0.262  (n=20)
      signed: mean dx +0.008 (L -0.007, R +0.024), mean dy -0.053
PASS  own image detection vs projected labels: max < 1.0 px  max 0.262, mean 0.114
PASS  no half-pixel convention bias in x (|mean dx| < 0.25 px; eyes not mirrored)  mean dx +0.008 px
      own detector vs package extra.qa detections: mean 0.227, max 0.577 px
      direction-test renders (+-1 extremes, neutral): image - projection (px): LEFT_L (-0.023,-0.000); LEFT_R (-0.034,-0.013); RIGHT_L (+0.074,-0.037); RIGHT_R (+0.123,-0.018); UP_L (+0.016,-0.250); UP_R (-0.004,-0.262); DOWN_L (-0.009,+0.103); DOWN_R (-0.017,+0.069); NEUTRAL_L (+0.050,-0.015); NEUTRAL_R (+0.033,-0.037)
PASS  own image detection vs projection on direction-test renders: max < 1.0 px  max 0.262 px

== 8. Listing's law ==
PASS  all rotation axes lie in Listing's plane (axis . primary axis)  max |component| 4.37e-08
      torsion of each eye in Fick and Helmholtz gimbals (deg; + = EXTORSION, i.e. 12 o'clock meridian temporal)
      f0       (+0.000,+1.000)  Fick tors L -0.340 R -0.340 | Helmholtz tors L +0.337 R +0.337 | elev Fick +8.102/+8.102 Helm +8.130/+8.130
      f1       (+0.588,+0.809)  Fick tors L +0.001 R -0.542 | Helmholtz tors L -0.001 R +0.546 | elev Fick +6.593/+6.503 Helm +6.593/+6.593
      f2       (+0.951,+0.309)  Fick tors L +0.066 R -0.268 | Helmholtz tors L -0.066 R +0.274 | elev Fick +2.524/+2.469 Helm +2.528/+2.528
      f3       (+0.951,-0.309)  Fick tors L -0.066 R +0.268 | Helmholtz tors L +0.066 R -0.274 | elev Fick -2.524/-2.469 Helm -2.528/-2.528
      f4       (+0.588,-0.809)  Fick tors L -0.001 R +0.542 | Helmholtz tors L +0.001 R -0.546 | elev Fick -6.593/-6.503 Helm -6.593/-6.593
      f5       (+0.000,-1.000)  Fick tors L +0.340 R +0.340 | Helmholtz tors L -0.337 R -0.337 | elev Fick -8.102/-8.102 Helm -8.130/-8.130
      f6       (-0.588,-0.809)  Fick tors L +0.542 R -0.001 | Helmholtz tors L -0.546 R +0.001 | elev Fick -6.503/-6.593 Helm -6.593/-6.593
      f7       (-0.951,-0.309)  Fick tors L +0.268 R -0.066 | Helmholtz tors L -0.274 R +0.066 | elev Fick -2.469/-2.524 Helm -2.528/-2.528
      f8       (-0.951,+0.309)  Fick tors L -0.268 R +0.066 | Helmholtz tors L +0.274 R -0.066 | elev Fick +2.469/+2.524 Helm +2.528/+2.528
      f9       (-0.588,+0.809)  Fick tors L -0.542 R +0.001 | Helmholtz tors L +0.546 R -0.001 | elev Fick +6.503/+6.593 Helm +6.593/+6.593
      LEFT     (-1.000,+0.000)  Fick tors L -0.000 R +0.000 | Helmholtz tors L -0.000 R +0.000 | elev Fick -0.000/-0.000 Helm -0.000/-0.000
      RIGHT    (+1.000,+0.000)  Fick tors L -0.000 R +0.000 | Helmholtz tors L +0.000 R -0.000 | elev Fick -0.000/-0.000 Helm -0.000/-0.000
      UP       (+0.000,+1.000)  Fick tors L -0.340 R -0.340 | Helmholtz tors L +0.337 R +0.337 | elev Fick +8.102/+8.102 Helm +8.130/+8.130
      DOWN     (+0.000,-1.000)  Fick tors L +0.340 R +0.340 | Helmholtz tors L -0.337 R -0.337 | elev Fick -8.102/-8.102 Helm -8.130/-8.130
      NEUTRAL  (+0.000,+0.000)  Fick tors L -0.000 R +0.000 | Helmholtz tors L -0.000 R -0.000 | elev Fick -0.000/-0.000 Helm -0.000/-0.000
      max |Fick torsion| 0.542 deg, max |Helmholtz torsion| 0.546 deg (non-zero 'false torsion' of a Listing eye)
PASS  Helmholtz elevations of the two eyes equal (common plane of regard)  max |dElev_H| 8.9e-16 deg; Fick elevations differ by up to 0.0901 deg
      L2 model sanity: 15 deg/eye vergence, 30 deg down, Listing plane +9.4 deg temporal -> +2.59 deg (Mok et al. 1992 report ~2.5 deg extorsion)
      f0       (+0.000,+1.000) L2 plane rot 2.97 deg -> missing torsion L -0.211 R -0.211 deg (cyclovergence -0.422, + = excyclo)
      f1       (+0.588,+0.809) L2 plane rot 2.96 deg -> missing torsion L -0.171 R -0.170 deg (cyclovergence -0.340, + = excyclo)
      f2       (+0.951,+0.309) L2 plane rot 2.95 deg -> missing torsion L -0.065 R -0.064 deg (cyclovergence -0.129, + = excyclo)
      f3       (+0.951,-0.309) L2 plane rot 2.95 deg -> missing torsion L +0.065 R +0.064 deg (cyclovergence +0.129, + = excyclo)
      f4       (+0.588,-0.809) L2 plane rot 2.96 deg -> missing torsion L +0.171 R +0.170 deg (cyclovergence +0.340, + = excyclo)
      f5       (+0.000,-1.000) L2 plane rot 2.97 deg -> missing torsion L +0.211 R +0.211 deg (cyclovergence +0.422, + = excyclo)
      f6       (-0.588,-0.809) L2 plane rot 2.96 deg -> missing torsion L +0.170 R +0.171 deg (cyclovergence +0.340, + = excyclo)
      f7       (-0.951,-0.309) L2 plane rot 2.95 deg -> missing torsion L +0.064 R +0.065 deg (cyclovergence +0.129, + = excyclo)
      f8       (-0.951,+0.309) L2 plane rot 2.95 deg -> missing torsion L -0.064 R -0.065 deg (cyclovergence -0.129, + = excyclo)
      f9       (-0.588,+0.809) L2 plane rot 2.96 deg -> missing torsion L -0.170 R -0.171 deg (cyclovergence -0.340, + = excyclo)
      LEFT     (-1.000,+0.000) L2 plane rot 2.94 deg -> missing torsion L -0.000 R -0.000 deg (cyclovergence -0.000, + = excyclo)
      RIGHT    (+1.000,+0.000) L2 plane rot 2.94 deg -> missing torsion L -0.000 R -0.000 deg (cyclovergence -0.000, + = excyclo)
      UP       (+0.000,+1.000) L2 plane rot 2.97 deg -> missing torsion L -0.211 R -0.211 deg (cyclovergence -0.422, + = excyclo)
      DOWN     (+0.000,-1.000) L2 plane rot 2.97 deg -> missing torsion L +0.211 R +0.211 deg (cyclovergence +0.422, + = excyclo)
      NEUTRAL  (+0.000,+0.000) L2 plane rot 3.00 deg -> missing torsion L -0.000 R -0.000 deg (cyclovergence -0.000, + = excyclo)
      max |L2 cyclovergence| not modelled: 0.422 deg (k = 0.627 x half-vergence; scales linearly with k)

== wrapper cross-check (package eyes.move vs this script) ==
PASS  package move() rotations == numpy (14 poses)  max 1.38e-07 rad
PASS  package project() == numpy P (pupils, 14 poses)  max 1.73e-04 px
PASS  package metadata() reproduces metadata.json (10 frames)  max 0.00e+00

47/47 checks PASS
JSON_SUMMARY {"eye_centres": {"C_L": [0.029335640370845795, -0.12631183862686157, 0.8034566640853882], "C_R": [-0.029335640370845795, -0.12631183862686157, 0.8034566640853882], "M": [0.0, -0.12631183862686157, 0.8034566640853882], "IPD_mm": 58.67128074169159}, "target": {"F0": [0.0, -0.47631183862686155, 0.8034566640853882], "max_err_vs_metadata_m": 2.209519806276686e-08, "plane_dev_m": 0.0, "corner_radial_offset_cm": 7.071067811865479}, "rotations": {"max_quat_err_rad": 1.4219028776391624e-07, "max_euler_err_rad": 1.6885123833562333e-07, "max_miss_m": 5.594315114139762e-17, "extremes": {"LEFT": {"h": -1.0, "v": 0.0, "yaw_L": -12.77160177607203, "pitch_L": 0.0, "adduction_L": 12.77160177607203, "ang_L": 12.771601776071998, "yaw_R": -3.37887925662848, "pitch_R": 0.0, "adduction_R": -3.37887925662848, "ang_R": 3.3788792566284624, "vergence": 9.392722519443591, "miss_max": 0.0}, "RIGHT": {"h": 1.0, "v": 0.0, "yaw_L": 3.37887925662848, "pitch_L": 0.0, "adduction_L": -3.37887925662848, "ang_L": 3.3788792566284624, "yaw_R": 12.77160177607203, "pitch_R": 0.0, "adduction_R": 12.77160177607203, "ang_R": 12.771601776071998, "vergence": 9.392722519443591, "miss_max": 0.0}, "UP": {"h": 0.0, "v": 1.0, "yaw_L": -4.791111194722861, "pitch_L": 8.102072221963756, "adduction_L": 4.791111194722861, "ang_L": 9.40452247239532, "yaw_R": 4.791111194722861, "pitch_R": 8.102072221963756, "adduction_R": 4.791111194722861, "ang_R": 9.40452247239532, "vergence": 9.48635781803155, "miss_max": 6.938893903907228e-18}, "DOWN": {"h": 0.0, "v": -1.0, "yaw_L": -4.791111194722861, "pitch_L": -8.102072221963756, "adduction_L": 4.791111194722861, "ang_L": 9.40452247239532, "yaw_R": 4.791111194722861, "pitch_R": -8.102072221963756, "adduction_R": 4.791111194722861, "ang_R": 9.40452247239532, "vergence": 9.48635781803155, "miss_max": 6.938893903907228e-18}, "NEUTRAL": {"h": 0.0, "v": 0.0, "yaw_L": -4.791111194722862, "pitch_L": 0.0, "adduction_L": 4.791111194722862, "ang_L": 4.791111194722739, "yaw_R": 4.791111194722862, "pitch_R": 0.0, "adduction_R": 4.791111194722862, "ang_R": 4.791111194722739, "vergence": 9.582222389445684, "miss_max": 5.594315114139762e-17}}}, "readme_numbers": {"inward_deg": 4.791111194722862, "elev_deg": 8.102072221963756, "max_adduction_deg": 12.77160177607203, "max_abduction_deg": 3.37887925662848, "max_elev_square_deg": 8.130101469702868}, "camera": {"K": [[1763.5555555555557, 0.0, 512.0], [0.0, 1763.5555555555557, 512.0], [0.0, 0.0, 1.0]], "P": [[1763.5555555555557, 511.99999999999955, 2.23802308028098e-05, 269.4716617059542], [0.0, 512.0000770874612, -1763.5555331753233, 1686.4121658739195], [0.0, 0.9999999999999991, 4.371138828673789e-08, 0.5263118392694418]], "dist_along_m": 0.40000003576278687, "lateral_m": 0.0, "M_px": [512.0, 512.0000770874616], "max_P_vs_meta_px": 0.0008114406965091803, "max_P_vs_w2cv_px": 0.0008095093343172266}, "convention": {"own_detector_synth_bias_px": [-0.00038397497206489107, 0.0032916383357974154], "own_detector_synth_max_px": 0.05948191358619638, "pkg_detector_synth_bias_px": [-0.0028049323142261074, -0.00813105864493366], "pkg_detector_synth_max_px": 0.1470112509383759}, "pupil_travel_px": {"LEFT_L": [-6.731068866796477, 2.2737367544323206e-13], "LEFT_R": [-6.904600278898613, -1.1368683772161603e-13], "RIGHT_L": [6.904600278898556, -1.1368683772161603e-13], "RIGHT_R": [6.731068866796477, 2.2737367544323206e-13], "UP_L": [0.00518253171469496, -6.822371746625322], "UP_R": [-0.0051825317147518035, -6.822371746625322], "DOWN_L": [0.005182575287108193, 6.822371748933392], "DOWN_R": [-0.005182575287278723, 6.822371748933392]}, "pupil_image_check": {"n": 20, "mean_px": 0.11350208908266073, "median_px": 0.09823334935443828, "rms_px": 0.13739776960342484, "max_px": 0.2623886994092137, "mean_dx": 0.008279603021915705, "mean_dy": -0.05278133147728283, "mean_dx_L": -0.007018304362827621, "mean_dx_R": 0.02357751040665903, "per_pupil_px": [0.25068994735917766, 0.2623886994092137, 0.15827810101036602, 0.16738351214809064, 0.09426992044809476, 0.12355051165694694, 0.02645787487230659, 0.06438774120370463, 0.007823068616835497, 0.01739845501312263, 0.10381042217893649, 0.07064037567331032, 0.10093061108128605, 0.0955360876275905, 0.08021581967478887, 0.021180038460261615, 0.1369878764794592, 0.038202643332575906, 0.22045045042903852, 0.2294596249781082], "vs_pkg_qa_mean_px": 0.22689087207149244, "direction_test_px": {"LEFT_L": [-0.022562523977740057, -0.00015449864508809696], "LEFT_R": [-0.034351653827968676, -0.012582897144682192], "RIGHT_L": [0.07402943064357714, -0.036830325386404184], "RIGHT_R": [0.12264055852807587, -0.01825209771112668], "UP_L": [0.01566901112028063, -0.2501997837678118], "UP_R": [-0.0038496982159017534, -0.2623604570077305], "DOWN_L": [-0.00920412633831802, 0.10340158514895847], "DOWN_R": [-0.016594830965516394, 0.06866348564187774], "NEUTRAL_L": [0.0502162531254271, -0.015142015268907016], "NEUTRAL_R": [0.032821836149196315, -0.0373709139647076]}}, "listing": {"max_axis_out_of_plane": 4.3712229948190815e-08, "max_fick_torsion_deg": 0.5423081153200753, "max_helmholtz_torsion_deg": 0.5462526373802075, "L2_k": 0.6266666666666667, "L2_sanity_mok_deg": 2.5905056517420855, "max_L2_cyclovergence_deg": 0.4223046880164546}, "wrapper": {"rot_rad": 1.3816457041339077e-07, "proj_px": 0.00017283096024846287, "meta_repro": 0.0}, "n_checks": 47, "n_fail": 0}
```
