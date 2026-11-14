# Execution guide

Run everything from the package root (the folder with `scene.blend`). `BLENDER` is the Blender
executable; with the `bpy` module replace `"$BLENDER" -b scene.blend --python X -- ARGS` by
`python X ARGS`.

## Profiles — official outputs are protected

| Profile | Command | Samples | Writes to |
|---|---|---|---|
| **quality** (official) | `generate_sequence.py` | 32 (stored in scene.blend) | `renders/`, `metadata.json`, `reports/` |
| **demo** (live, fast) | `generate_sequence.py -- --demo` | 8 | `demo_run/renders`, `demo_run/metadata.json`, `demo_run/reports` |
| custom | `generate_sequence.py -- --samples N --out DIR` | N | `DIR/...` |

Any run below the quality sample count that would write into the package root is refused (exit 2,
symlinks resolved). `eyes.py` / `eyes.render()` without a path write to `demo_run/renders/`, and
`eyes.render()` refuses to write `renders/frame_NN.png`; only a quality run of `generate_sequence.py`
rewrites the official frames and `metadata.json`. Regenerating the official
outputs always re-runs the 5-direction validation.

## Linux / macOS

```bash
BLENDER=blender
"$BLENDER" -b scene.blend --python wrapper/generate_sequence.py                 # official: 12.4 min on 2 vCPUs
"$BLENDER" -b scene.blend --python wrapper/generate_sequence.py -- --demo       # live demo: 5.1 min on 2 vCPUs
"$BLENDER" -b scene.blend --python wrapper/validate_directions.py               # 5-direction test only
"$BLENDER" -b scene.blend --python wrapper/eyes.py -- --move -1 0 --out /tmp/left.png --samples 8
python3 wrapper/make_boards.py --pkg . --out reports/boards --contact-sheet renders/contact_sheet.png
"$BLENDER" -b --python wrapper/run_acceptance_tests.py -- --zip ../GazeForge.zip \
           --reference-metadata wrapper/tests/v1_reference_metadata.json
```

## Windows (PowerShell)

```powershell
$BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
& $BLENDER -b scene.blend --python wrapper\generate_sequence.py
& $BLENDER -b scene.blend --python wrapper\generate_sequence.py -- --demo
& $BLENDER -b scene.blend --python wrapper\validate_directions.py
& $BLENDER -b scene.blend --python wrapper\eyes.py -- --move -1 0 --out $env:TEMP\left.png --samples 8
powershell -ExecutionPolicy Bypass -File wrapper\tools\windows_smoke_test.ps1   # ~5 min, writes demo_run\windows\
```

## Interactive (Blender Python console or a script)

```python
import sys; sys.path.insert(0, r"<package>/wrapper")
import eyes
eyes.reset(); eyes.move(0.0, 1.0); print(eyes.render("/tmp/up.png")); print(eyes.metadata()["pupil_center_2D_L"])
```
