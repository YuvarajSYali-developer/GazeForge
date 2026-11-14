# AI disclosure

GazeForge was conceived, directed and submitted by **Yuvaraj S Yali** (solo entry). In line with the hackathon's
requirement to disclose AI use, this file states how AI tools were used.

## Tools used

* **Claude (Anthropic)**, an AI assistant. It was used throughout development as a coding and writing assistant.

## What the AI assisted with

* Writing and debugging the Python code:
  * the GNM evaluator;
  * the semantic-decoder port;
  * the texture generator;
  * the Blender scene, rig and motion scripts;
  * the Colab notebook and the Windows test script.
* Researching published reference values (saccade main sequence, blink timing, skin scattering constants, iris
  anatomy) and implementing them.
* Drafting documentation: the README, slides text and reports.
* Running renders, checks and packaging in a cloud workspace.

## What the author did

* Chose the track, the approach and the scope.
* Provided the logic and specifications that the code implements: detailed written specifications of the modules,
  behaviours, acceptance checks and rules (for example, the GazeForge V3 master specification and the V2-protection
  rules). The AI wrote code to these specifications, and the author reviewed it.
* Supplied the inputs: the GNM model file, the organisers' evaluation criteria and the submission template.
* Made the design decisions: GNM over FLAME, procedural textures, which features to build, and what to submit.
* Reviewed the outputs and approved the final submission.

## What was not AI-generated

* **No generative image or video model was used.** All textures, renders and videos come from the deterministic code in
  this repository (NumPy and Blender Cycles). Anyone can regenerate them with `tools/run_all.sh`.
* **No real person's photographs or scans were used.**
* **Third-party assets** are credited in `THIRD_PARTY_NOTICES.md`: the GNM head model (Google, Apache-2.0), and
  MakeHuman CC0 assets in V2.

All results reported in this repository are reproducible from the code. Their verification steps are described in the
README.
