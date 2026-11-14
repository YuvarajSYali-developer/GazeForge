# Cross-version portability and reproducibility

`scene.blend` is saved by **Blender 4.2.23 LTS** (the reference run). The same file was opened with the official `bpy` builds of newer Blender versions; `wrapper/tools/cross_version_check.py` replays all 10 poses of `metadata.json` and runs the 5-direction test (8 samples, renders in a temp folder).

| Blender | Python | Max 2-D pupil diff vs metadata.json | Max rotation diff | Max camera_matrix diff | 5-direction test |
|---|---|---|---|---|---|
| 4.2.23 LTS (reference) | 3.11.18 | — (produced metadata.json) | — | — | 5/5 (`direction_validation.md`) |
| 5.0.1 | 3.11.18 | 0.0 px | 0.0 rad | 0.0 | 5/5 (LEFT PASS, RIGHT PASS, UP PASS, DOWN PASS, RESET PASS) |
| 4.5.15 LTS | 3.11.18 | 0.0 px | 0.0 rad | 0.0 | 5/5 (LEFT PASS, RIGHT PASS, UP PASS, DOWN PASS, RESET PASS) |

Ground truth is bit-identical across versions: labels depend only on the rig and camera, not on the renderer.

## Reproducibility (Blender 4.2.23 LTS)

`wrapper/tools/reproducibility_check.py` re-rendered official frames with the quality settings (32 samples, seed 0, no adaptive sampling) into a temp folder and compared them with `renders/`:

| Frame | Max abs diff (8-bit) | Mean abs diff | Identical pixels | Metadata pupil diff |
|---|---|---|---|---|
| 0 | 0.0 | 0.0 | True | 0.0 px |
| 5 | 0.0 | 0.0 | True | 0.0 px |
