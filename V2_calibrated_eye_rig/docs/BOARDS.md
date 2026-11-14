# GazeForge presentation boards

`make_boards.py` turns a GazeForge output package into five scientific figures ("boards"). Each board is written
as a PNG (at least 2400 px on the long side) and as a PDF. The boards add to the official renders and never
replace them. The script reads the package and writes only to `--out`, which must lie outside the package or
under its `reports/`; `--contact-sheet` may only name `renders/contact_sheet.png`. Running
`generate_sequence.py` writes a quick clock-layout `renders/contact_sheet.png`; running this script afterwards
replaces it with the annotated sequence board shipped in the package.

## Run

```bash
python3 make_boards.py --pkg <package_dir> --out <output_dir> [--before <older_package_dir>]
# e.g.
python3 wrapper/make_boards.py --pkg . --out reports/boards --before <older_package> --contact-sheet renders/contact_sheet.png
```

Requirements: CPython 3 with numpy, Pillow and matplotlib (Agg backend). bpy is not needed (the Python bundled with Blender usually lacks matplotlib, so use a system Python).
A run takes about 8–9 s on one core. The output is deterministic: two runs produce byte-identical PNGs and PDFs.

## Boards

| File | What it shows | Data used |
|---|---|---|
| `sequence_board` | The 10 frames on a clock. Frame k sits at θk clockwise from 12 o'clock. The centre has a clockwise arrow, tick marks for k and the equations θk = 2πk/10, h = sin θk, v = cos θk. Each tile has the full-frame thumbnail with a rectangle on the eye region. Below it is a framed inset of that region at ×3 relative to the thumbnail, joined to the rectangle by leader lines. Green crosses mark `pupil_center_2D_L/R`. Each tile is labelled with k, θ, h and v. | `renders/frame_*.png`, `metadata.json` |
| `direction_board` | Rows NEUTRAL, LEFT, RIGHT, UP, DOWN, RESET. Each row has 48×48 px crops of both eyes at Lanczos ×10, labelled by subject side and image side. A red ring marks the neutral reference and is the same in every row. A green + marks the measured projected position and a green × the measured detected position. A green arrow runs from reference to projected position at true scale. A table gives dx/dy (projected and detected), along/cross values and PASS/FAIL. RESET shows the shifts, the quaternion difference and the image difference. | `reports/direction_validation.json`, `reports/direction_test/*.png` |
| `eye_detail` | Native eye crops (144×96 px) at Lanczos ×4 for neutral and the four extreme directions, both eyes side by side. The scale bar is drawn outside the pixels (10 px ≈ 2.27 mm at the eye-centre depth, from mm = px·Z/fx). | `reports/direction_test/*.png`, metadata eye centres and K |
| `eye_detail_before_after` | Only made with `--before`. For each condition the BEFORE row (older package) sits above the AFTER row (this package). Both use the same crop windows. Each pair shows max/mean \|after−before\| computed on the native 8-bit crops. | both packages' `reports/direction_test/*.png` |
| `camera_calibration` | To-scale top view (X–Y) and side view (Y–Z) in cm relative to M. It shows the eyeballs (sclera radius), M, F0, the ±5 cm target bound, the camera, the field-of-view (FOV) wedge with half-angle atan(c/f), the optical axis and dimension lines (35 / 40 / 5 cm and the FOV width at M). Detail A enlarges the eyes and shows the IPD. A table lists IPD, positions, distances, fx/fy, cx/cy, focal length in mm, FOV, where M projects, how far M is off the optical axis, and the eye depth. | `metadata.json` frame 0 |
| `gaze_rays` | To-scale top view of rays from C_L and C_R to the targets for h = −1, 0, +1, and side view for v = −1, 0, +1. Each view has a computed yaw or pitch table and the vergence. Panels near the eyes show the primary direction and the yaw/pitch arcs. A front view of the fixation plane shows the ±5 cm square, the 10 metadata targets labelled k and the ±1 calibration targets. A table lists the per-frame yaw, pitch and vergence from metadata. | `metadata.json` |

## Where the landmarks come from

- **Pixel landmarks.** Pupil centres come from `pupil_center_2D_L/R` (metadata) or from `direction_validation.json`
  (projected and detected neutral positions, plus dx and dy). No landmark is moved, snapped or exaggerated.
  Every crop is drawn with `imshow(extent=(x0, x1, y1, y0))` in the original image's pixel coordinates (pixel i spans
  [i, i+1)). Markers are plotted in those same coordinates, so the crop/scale transform for markers is exactly the one
  used for the pixels. The table values in the direction board are the JSON values. Its positions are recomputed from
  them: neutral + (dx, dy), for both projected and detected values.
- **Geometry.** Eye centres, M, the camera pose, K and the targets come from metadata. The head axes f, r and u are
  the camera axes in `camera_to_world_blender`. The fixation distance (0.35 m) and the target bounds (±0.05 m) are
  derived from the 10 metadata fixation targets rather than hard-coded. The ±1 targets in `gaze_rays` are
  *computed from calibration*: T = M + d·f + h·b_h·r + v·b_v·u. They are labelled that way and kept separate from the
  green metadata targets.
- **Not in metadata.** The sensor width (36 mm), the lens (62 mm) and the sclera radius (12.5 mm) are parsed from the
  package `README.md` with regexes. If parsing fails the documented defaults are used, and the board says which
  source it used.
- **Self-checks printed on the boards.**
  - Metadata θ/h/v against the sampling equations: deviation 0.
  - Yaw and pitch recomputed from the metadata targets and eye centres against `extra.yaw_deg` and `pitch_deg`: ≤ 6.3e-6°.
  - P·M = (512.000, 512.000), and M lies 0.0000 mm off the optical axis.

## Style

The background is white with dark-grey text in DejaVu Sans, plus DejaVu Sans Mono for number tables. There is one
blue accent family. Red marks reference or neutral items (M, F0, the neutral pupil). Green marks measured or
ground-truth pupil positions and the metadata targets. There are no gradients; the only fill is the flat, light FOV
wedge.
