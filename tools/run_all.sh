#!/bin/bash
# Rebuild every output from the GNM model file.
#   GNM=/path/to/gnm_head.npz LANDMARKS=/path/to/GNM/gnm/shape/data/landmarks/head_sparse_68.txt tools/run_all.sh
# Needs: python3 with numpy scipy opencv-python h5py pillow (textures) and Blender 4.2+ on PATH (or BLENDER=...).
set -e
GNM=${GNM:?set GNM=/path/to/gnm_head.npz}
LANDMARKS=${LANDMARKS:?set LANDMARKS=.../head_sparse_68.txt from the GNM repository}
BLENDER=${BLENDER:-blender}
cd "$(dirname "$0")/.."
python3 tools/make_textures.py "$GNM" "$LANDMARKS" textures
python3 tools/make_previews.py textures "$GNM" textures/previews
python3 tools/make_scalp_density.py "$GNM" "$LANDMARKS" assets/gnm_scalp_density.npy
"$BLENDER" -b --python build_scene.py -- --gnm "$GNM" --set 02 --out GazeForge_GNM.blend
"$BLENDER" -b --python build_timeline_blend.py -- --gnm "$GNM" --set 03 --out GazeForge_GNM_Timeline.blend
"$BLENDER" -b GazeForge_GNM.blend --python gnm_eyes.py -- --input Left --texture 02 --out videos/left.mp4
"$BLENDER" -b GazeForge_GNM.blend --python gnm_eyes.py -- --input Right --texture 06 --out videos/right.mp4
"$BLENDER" -b GazeForge_GNM.blend --python gnm_eyes.py -- --input "Generate Motion" --texture 03 --out videos/generate_motion.mp4
"$BLENDER" -b --python render_stills.py --
echo ALL_DONE
