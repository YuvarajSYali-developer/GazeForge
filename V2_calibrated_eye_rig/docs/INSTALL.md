# Installation

GazeForge needs **one** of the following. Nothing else (no add-ons, no GPU, no internet).

## Option A — Blender executable (recommended for evaluators)

1. Install **Blender 4.2 LTS** (or 4.5 LTS / 5.0) from blender.org.
2. Unzip the submission; open a terminal in the folder that contains `scene.blend`.
3. Check: `blender --version` (Linux/macOS) or
   `& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --version` (PowerShell).

## Option B — `bpy` from PyPI (headless servers / CI)

Requires **Python 3.11** (the bpy 4.2–5.0 wheels are built for 3.11).

```bash
python3.11 -m venv .venv && . .venv/bin/activate          # Windows: py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install "bpy==4.2.*"                                    # ~350 MB wheel, includes numpy
python -c "import bpy; print(bpy.app.version_string)"
```

## Optional — presentation boards

`wrapper/make_boards.py` runs on any Python 3 with `pip install numpy pillow matplotlib`.
It is not needed for the evaluated pipeline.

## What was actually tested

| Environment | Status |
|---|---|
| Linux x86-64, `bpy` 4.2.23 LTS / Python 3.11.18 | full pipeline, acceptance A–I (reference run) |
| Linux x86-64, `bpy` 4.5.15 LTS and 5.0.1 | metadata replay identical, 5-direction 5/5 |
| Blender **executable** (any OS) | not run by us — use `wrapper/tools/windows_smoke_test.ps1` (Windows) or the commands in docs/EXECUTION.md |
| Windows | not run by us (see README §11) — smoke-test script provided |
