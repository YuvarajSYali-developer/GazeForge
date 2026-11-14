# GazeForge v2 — engineering report

Samsung PRISM XR Hackathon 2026 · Track 01 (Synthetic Eye Movement & Virtual Camera Rendering).
Every number below comes from a file in this package; the file is named beside it.

## 0. Mandatory vs optional

| Item | Status | Where |
|---|---|---|
| `scene.blend`: textured mesh, eye bones, materials | **Mandatory — done** | root |
| `wrapper/eyes.py`: `move(h, v)`, `reset()`, `render()` | **Mandatory — done** | `wrapper/` |
| `renders/`: 10 clockwise frames, θk = 2πk/10, 1024², starts at top | **Mandatory — done** | `renders/frame_00..09.png` |
| `metadata.json`: frame, theta, input, eye_rotation_L/R, pupil_center_2D_L/R, camera_matrix | **Mandatory — done** | root |
| `README.md`: setup and run instructions | **Mandatory — done** | root |
| 5-direction test (left, right, up, down, reset) | **Mandatory — 5/5** | `reports/direction_validation.md` |
| Contact sheet or 1–2 min demo video | **Mandatory (either) — both** | `renders/contact_sheet.png`, `GazeForge_demo.mp4` (separate) |
| Presentation (PPT or PDF) | **Mandatory (general rules) — both** | `GazeForge_Submission.pptx` / `.pdf` (separate) |
| Public/shared GitHub repository | **Mandatory (general rules) — repo ready, push pending (team account)** | git history in this repo |
| Acceptance suite A–I, boards, cross-version, reproducibility, audits | Optional (our own evidence) | `reports/`, `docs/audit/` |
| Demo profile, Windows smoke-test script | Optional | `wrapper/generate_sequence.py --demo`, `wrapper/tools/windows_smoke_test.ps1` |

## 1. Architecture

`scene.blend` holds the MakeHuman (CC0) head, the `EyeRig` armature (`head`, `eye.L`, `eye.R`; each bone
head is an eyeball centre fitted to MakeHuman's eye helpers), two rigid eye assemblies (eyeball → iris →
pupil, cornea) parented to their bones, lashes, the locked `RenderCamera` and fixed lights, plus a
`calibration_json` that is the single source of truth. `wrapper/eyes.py` turns (h, v) into the target
T = F₀ + 0.05·h·r + 0.05·v·u on the 35 cm plane, computes a shortest-arc quaternion per eye from its own
centre, poses only the eye bones, checks that camera, lights, head and resolution are unchanged, renders, and
reports rotations, 2-D pupils from P = K[R|t] and the camera matrix. `generate_sequence.py` drives the 10
frames, metadata, run log and validation; `validate_directions.py` and `run_acceptance_tests.py` verify.

## 2. Before (v1) → after (v2)

| | v1 (`edd9e7c`) | v2 |
|---|---|---|
| Ground truth (rotations, 2-D pupils, P, all frames) | reference | **identical, max diff 0.0** (acceptance H) |
| Labels vs pupils detected in the PNGs | max 0.84 px, mean 0.30 px | **max 0.26 px, mean 0.11 px** (`run_log.json`) |
| Directions | 5/5 | 5/5 |
| Acceptance | 9/9 | 9/9, 63/63 checks, now incl. v1-preservation check |
| Eye region | dark socket corners, flat iris, white sclera, no lashes, painted brows | conjunctiva/waterline, 2K iris with relief, off-white veined sclera, ray-cast-rooted lashes, hair-stroke brows (`reports/boards/eye_detail_before_after.png`) |
| Profiles | one | quality (official) + demo (`demo_run/`, cannot overwrite official outputs) |
| Reports | absolute machine paths | portable (`<pkg>`, `<tmp>`) |
| Exit at shutdown | — | exit 0 (segfault from a stats handler fixed) |

## 3. Scene · source · renders · metadata

* Scene: `scene.blend` (saved with Blender 4.2.23 LTS, all 7 textures packed).
* Source: `wrapper/` — `build_scene.py` rebuilds the scene deterministically; `tools/` holds the gate, audits and checks.
* Renders: `renders/frame_00..09.png` (32 spp, seed 0, Filmic), `renders/contact_sheet.png` (sequence board, true-scale ×3 insets).
* Metadata: `metadata.json`, 10 records, required fields plus `extra` (convention strings, pupils in 3-D, profile).

## 4. Validation (all run, all on the final ZIP where marked)

| Check | Result | Evidence |
|---|---|---|
| 5-direction test, rendered pixels | 5/5 (≈6.7–7.1 px per eye per axis, reset 0 px / 0 image diff) | `reports/direction_validation.md` |
| Acceptance A–I on the packaged ZIP | 9/9, 63/63 | `reports/acceptance_report.md` |
| Projection vs detection, 20 pupils | max 0.26 px, mean 0.11 px | `reports/run_log.json` |
| Independent maths audit (NumPy re-derivation) | consistent; 7 doc fixes applied | `docs/audit/math_audit.md` |
| Cross-version replay | 4.5.15 LTS and 5.0.1: 0.0 px / 0.0 rad, 5/5 each | `reports/cross_version.md` |
| Reproducibility | frames 0 and 5 re-rendered pixel-identical | `reports/cross_version.md` |
| Separate red-team QA pass on the packaged ZIP (own numpy + own detector) | PASS, 0 critical; findings fixed (§9) | §9 |

## 5. Benchmark (2-vCPU cloud VM, Cycles CPU, Blender 4.2.23 LTS)

| Profile | Samples | Scene ready | Per frame | 10 frames + metadata | Validation | Total | Peak RSS |
|---|---|---|---|---|---|---|---|
| quality (official) | 32 | 0.17 s | 46.2 s | 463 s | 276 s | 743 s | 1.6 GB |
| demo | 8 | 0.18 s | 18.8 s | 189 s | 113 s | 306 s | 1.6 GB |

Sources: `reports/run_log.json`, `demo_run/reports/run_log.json` (demo run not shipped in the ZIP).

## 6. Separate deliverables

`GazeForge.zip` (the bundle) · `GazeForge_Submission.pptx` and `.pdf` (15 slides, all figures from package
data) · `GazeForge_demo.mp4` (1080p30, under 2 min, built from the recorded demo run and real renders) ·
`renders/contact_sheet.png` · `reports/boards/direction_board.png` · `reports/acceptance_report.md` · this report.

## 7. Not tested

* **Windows**: no Windows machine with enough free disk was available. `wrapper/tools/windows_smoke_test.ps1`
  (≈5 min) runs the API and the 5-direction test with `blender.exe` and writes `demo_run\windows\`.
* **Blender executable** (`blender -b … --python …`): only the official `bpy` module builds (4.2.23, 4.5.15,
  5.0.1) were run. They are the same code as the executable; the command form is documented but was not run here.
* GPU rendering, macOS.

## 8. Limitations

±5 cm at 35 cm gives only ≈7 px pupil travel at 40 cm (geometry, not a bug — boards use labelled magnified
insets) · static eyelids (≈3.4 mm clearance above the pupil at full up-gaze) · non-refracting cornea, optical
axis = gaze axis, fixed 4 mm pupil · binocular Listing L2 / cyclovergence (≈0.4°) not modelled · one identity, one lighting set-up.

## 9. Red-team QA

A separate QA pass audited the packaged ZIP with its own NumPy geometry and its own pupil detector, rendering
only at ≤ 8 samples into its own folders. Verdict: **PASS — nothing that would fail the organisers' evaluation.**

Measured independently: eye centres drift 0.0 m over 10 poses; gaze rays miss the common target by ≤ 6.0·10⁻⁸ m;
fixation 0.35 m and camera 0.400 m from the eye midpoint, which projects to (512, 512); P reproduces every 2-D
pupil to 8.8·10⁻⁵ px; Euler XYZ matches the quaternions to 4.8·10⁻⁸; own renders 5/5 (≈6.8–7.0 px per eye);
sequence inputs exact, clockwise from the top; labels vs own detection on the official PNGs max 0.18 px
(mean 0.09); all 10 records reproduce exactly on 4.2, 4.5 and 5.0; refused low-sample runs left all files unchanged.

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | major | Internal development notes (`docs/V2_PLAN.md`) shipped; audit called "independent" | plan removed; audit retitled "internal"; wording fixed |
| 2 | minor | `reset()` could not recover from a camera/light/world-strength edit (1.2·10⁻⁷ round-off) | exact channel restore + world strength; tolerance 1 µm; verified |
| 3 | minor | "never overwrite" too absolute: `eyes.render()` defaulted to `renders/`; symlinked root bypassed check | default `demo_run/renders/`; `renders/frame_NN.png` refused outside a quality sequence run; `realpath`; wording narrowed |
| 4 | minor | Numeric strings and `np.bool_` accepted by `move()` | rejected (TypeError); verified |
| 5 | minor | README timings ≠ shipped run log; no demo log | table regenerated from `run_log.json`; `run_log_demo.json` shipped |
| 6 | minor | Bad `--samples` crashed (exit 1) | clean message, exit 2 |
| 7 | minor | `metadata()` "same schema" wording | corrected |
| 8 | minor | BOARDS.md vs EXECUTION.md on where boards are written | corrected |
| 9 | minor | Possible `relpath` crash across Windows drives | `relpath_safe` fallback |
| 10 | minor | README tree listed 5 of 9 tools | updated |
| info | — | contact sheet replaced by `make_boards.py`; temp folders left by the acceptance suite; ZIP name vs registered `CollegeName_TeamName` | documented; temp folders now removed at exit; name to confirm |

The maths audit had flagged "6.91 vs 6.90" as a README rounding issue (D7); the cause was double rounding in the report generator (6.9046 → 6.905 → "6.91");
values are now stored to 4 decimals and the official 5-direction report was regenerated at quality settings.

## 10. Jury defence

See `docs/audit/math_audit.md` and `reports/`.
