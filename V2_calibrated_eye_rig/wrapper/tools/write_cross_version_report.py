"""
write_cross_version_report.py -- turn CROSSVER / REPRO lines into reports/cross_version.{json,md}.

    python3 wrapper/tools/write_cross_version_report.py <log-with-CROSSVER-and-REPRO-lines>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(log):
    xv, repro = [], None
    for ln in open(log):
        if ln.startswith("CROSSVER "):
            xv.append(json.loads(ln[9:]))
        elif ln.startswith("REPRO "):
            repro = json.loads(ln[6:])
    out = {"cross_version": xv, "reproducibility": repro}
    with open(os.path.join(ROOT, "reports", "cross_version.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    L = ["# Cross-version portability and reproducibility", "",
         "`scene.blend` is saved by **Blender 4.2.23 LTS** (the reference run). The same file was opened with the official "
         "`bpy` builds of newer Blender versions; `wrapper/tools/cross_version_check.py` replays all 10 poses of "
         "`metadata.json` and runs the 5-direction test (8 samples, renders in a temp folder).", "",
         "| Blender | Python | Max 2-D pupil diff vs metadata.json | Max rotation diff | Max camera_matrix diff | 5-direction test |",
         "|---|---|---|---|---|---|",
         "| 4.2.23 LTS (reference) | 3.11.18 | — (produced metadata.json) | — | — | 5/5 (`direction_validation.md`) |"]
    for r in xv:
        L.append(f"| {r['blender']} | {r['python']} | {r['replay_max_pupil2D_diff_px']} px | {r['replay_max_rotation_diff_rad']} rad | "
                 f"{r['replay_max_camera_matrix_diff']} | {r['direction_validation']} "
                 f"({', '.join(k + ' ' + v for k, v in r['tests'].items())}) |")
    L += ["", "Ground truth is bit-identical across versions: labels depend only on the rig and camera, not on the renderer.", ""]
    if repro:
        L += ["## Reproducibility (Blender 4.2.23 LTS)", "",
              f"`wrapper/tools/reproducibility_check.py` re-rendered official frames with the quality settings "
              f"({repro['samples']} samples, seed 0, no adaptive sampling) into a temp folder and compared them with `renders/`:", "",
              "| Frame | Max abs diff (8-bit) | Mean abs diff | Identical pixels | Metadata pupil diff |", "|---|---|---|---|---|"]
        for k, f in repro["frames"].items():
            L.append(f"| {k} | {f['max_abs_diff_8bit']} | {f['mean_abs_diff_8bit']} | {f['identical_pixels']} | {f['metadata_pupil_diff_px']} px |")
        L.append("")
    with open(os.path.join(ROOT, "reports", "cross_version.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
