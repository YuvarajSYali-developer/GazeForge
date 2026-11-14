"""
package_submission.py -- build <CollegeName_TeamName>.zip with the challenge's root-level layout.

    python wrapper/package_submission.py [--name GazeForge] [--out DIR]

The organisers name the bundle team_name.zip; registered team names follow CollegeName_TeamName,
so pass the exact registered name with --name.

Plain CPython (no bpy needed). The archive root holds scene.blend, wrapper/, renders/,
metadata.json, README.md (+ reports/ with the generated validation evidence).
No enclosing project folder is added.
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REQUIRED = ["scene.blend", "metadata.json", "README.md", "wrapper/eyes.py", "wrapper/generate_sequence.py",
            "renders/contact_sheet.png"] + [f"renders/frame_{k:02d}.png" for k in range(10)]
INCLUDE_DIRS = ["wrapper", "renders", "reports", "docs"]
ROOT_FILES = ["scene.blend", "metadata.json", "README.md", "THIRD_PARTY_NOTICES.md"]
SKIP = {"__pycache__", ".DS_Store", "demo_run", ".git"}


def main(argv):
    name = argv[argv.index("--name") + 1] if "--name" in argv else "GazeForge"
    out_dir = os.path.abspath(argv[argv.index("--out") + 1]) if "--out" in argv else os.path.dirname(ROOT)
    missing = [p for p in REQUIRED if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        sys.exit(f"missing required files: {missing}")
    files = [f for f in ROOT_FILES if os.path.isfile(os.path.join(ROOT, f))]
    for d in INCLUDE_DIRS:
        for base, dirs, fs in os.walk(os.path.join(ROOT, d)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            for f in sorted(fs):
                if f in SKIP or f.endswith(".pyc"):
                    continue
                files.append(os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/"))
    zpath = os.path.join(out_dir, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            # .blend is already compressed; PNGs too -> store to save time, deflate the rest
            ctype = zipfile.ZIP_STORED if rel.endswith((".blend", ".png", ".npz")) else zipfile.ZIP_DEFLATED
            z.write(os.path.join(ROOT, rel), rel, compress_type=ctype)
    print(f"wrote {zpath}  ({len(files)} files, {os.path.getsize(zpath)/1e6:.1f} MB)")
    return zpath


if __name__ == "__main__":
    main(sys.argv[1:])
