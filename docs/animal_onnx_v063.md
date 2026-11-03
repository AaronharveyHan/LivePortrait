# LivePortrait V0.6.3 — Geometric dual-source mouth morph (EXPERIMENTAL)

**Scope:** A separately testable, CPU-only 2-D geometric mouth morphing experiment on top of V0.6.2 reference images. **Not** learned 3-D jaw articulation, accurate phoneme lip sync, temporal-video system, or guaranteed natural in-between teeth/tongue.

## Why

V0.6.2 blended RGB mouth regions from two photographs of the SAME dog. Its 0%/100% endpoints looked more realistic than generated mouths, but teeth/tongue double-exposed at 25–75%. V0.6.3 introduces manually annotated corresponding mouth features and warps both reference images to an intermediate 2-D geometry **before** blending. Only a mouth ROI is modified; the closed reference provides face, head, ears and background everywhere else.

This iteration only **adds** files; does not edit previous LivePortrait ONNX models or original V0.6–V0.6.2 entrypoints. No retraining or GPU.

## Files

- `src/utils/animal_mouth_geometry.py`: landmark contracts; fixed-boundary triangle mesh; 2-D affine inverse remapping; soft/dominant texture selection; ROI mask; diagnostics and guide overlays
- `src/utils/animal_mouth_annotator.py`: fully offline self-contained HTML two-image point annotation helper
- `scripts/run_animal_mouth_geometry.py`: preflight, template/annotator creation, output files and per-alpha metrics
- `tests/test_animal_mouth_geometry.py`, `tests/test_animal_mouth_geometry_cli.py`: CPU unit tests and no-weights CLI integration
- `docs/animal_onnx_v063.md`: this file

### Sources and alignment

Use the exact registered references from a **completed V0.6.2 run**:

```text
build/onnx_animal_v062/yawn_preview/
  closed_reference.png
  open_reference_aligned.png
  report.json
```

The first image is the original/ONNX closed state; the second is the OPEN state **already translated into closed-image coordinates** by V0.6.2. Do NOT mark raw unaligned photos and then apply those coordinates to aligned images. The CLI verifies reference hashes embedded in landmark JSON and fails safely when sources change. Do not use two different dogs; `--same-dog` is an explicit user assertion, not an identity classifier.

**Important for the yawning dog:** the V0.6.2 example ROI `0.29,0.49,0.74,0.83` may clip the protruding tongue or chin. Inspect the 512px previews first; try `--roi '0.26,0.45,0.79,0.94'` only if the relevant mouth remains inside and outside changes are acceptable. No fixed anatomical detector is supplied.

## Windows PowerShell: apply

```powershell
cd C:\Users\ES\Documents\GitHub\LivePortrait
conda activate liveportrait-onnx
$git = 'C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe'
& $git branch --show-current
& $git status --short
& $git apply --check .\liveportrait_v0.6.3_mouth_geometry.patch
& $git apply .\liveportrait_v0.6.3_mouth_geometry.patch
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

Required CPU packages already installed in your `liveportrait-onnx`: `numpy`, `opencv-python-headless`, `scipy`, `pillow`. No new model weights. Existing V0.6.2 129 tests + **27 new tests** = expected **156** tests in the same environment.

## 1. Verify input references

```powershell
python scripts/run_animal_mouth_geometry.py --plan `
  --reference-dir build/onnx_animal_v062/yawn_preview
```

The tool consumes V0.6.2 outputs only, so run `run_animal_dual_source.py` in **original** or **onnx** mode beforehand, in a separate folder for each mode. `original` is a fast baseline; `onnx` has the existing two F/M/W/G reconstructions. V0.6.3 does **not** run ONNX itself or modify W/G.

## 2. Create offline click-to-annotate editor

```powershell
$ref = 'build/onnx_animal_v062/yawn_preview'
$ann = 'build/onnx_animal_v063/yawn_annotation'
python scripts/run_animal_mouth_geometry.py `
  --reference-dir $ref `
  --output-dir $ann `
  --same-dog `
  --roi '0.26,0.45,0.79,0.94' `
  --init-landmarks

Start-Process 'chrome.exe' -ArgumentList ([System.Uri](Resolve-Path "$ann/annotate_mouth.html").Path).AbsoluteUri
```

Outputs:

- `mouth_landmarks_template.json` (unverified placeholder coordinates; **do not use directly**)
- `closed_guide.png`, `open_guide.png` (coordinate grid and estimated positions)
- `annotate_mouth.html` (offline local click-to-annotate tool with embedded images, no external site)

Inside HTML: select each of 5 names and click **the corresponding location on both images**; each label needs both points. Use the same anatomical position where possible, not an imagined location on the tongue or a different tooth. Labels: `left_corner`, `upper_lip`, `right_corner`, `lower_lip`, `chin`. Some positions can be visually occluded; if a correspondence cannot be identified, **do not fake it**: choose a better video pair or manually revise the landmark scheme. Save the downloaded `mouth_landmarks_verified.json` into your repository (e.g. `$ann`) without overwriting the template.

The editor automatically sets `verified=true` once every landmark is clicked twice. The CLI still checks duplicate points, finite coordinates, geometric folds and image hashes; it cannot prove the human markup is semantically correct.

## 3. Generate morphing comparison

```powershell
python scripts/run_animal_mouth_geometry.py `
  --reference-dir $ref `
  --landmarks "$ann/mouth_landmarks_verified.json" `
  --same-dog `
  --policies both `
  --alphas '0,0.25,0.5,0.75,1' `
  --feather-px 24 `
  --output-dir build/onnx_animal_v063/yawn_geometry

mspaint.exe 'build/onnx_animal_v063/yawn_geometry/comparison_sheet.png'
```

`--policies both` generates three rows side-by-side:

1. `v062_rgb_blend`: original V0.6.2 pixel-only crossfade
2. `soft`: warp closed/open into the same interpolated geometry and **crossfade** textures; can still double teeth/tongue
3. `dominant`: warp geometry, choose closed texture below `alpha=0.5` and open texture at/above `alpha=0.5`; avoids two overlapping textures but can visibly **pop** at the switching point

Do not confuse `dominant` with perfect lip sync; it is a diagnostic texture-policy tradeoff. The border of the ROI is feathered, but neither policy invents real hidden tissue. At alpha 0 the image exactly equals closed reference; at alpha 1 the **mouth ROI** uses the aligned open reference, while everything else remains from closed. The tool protects other output files and requires `--overwrite` only for its own prior render outputs; landmark templates are never overwritten.

Other files: `mouth_soft_000.png...`, `mouth_dominant_...`, `report.json` (source hashes, alpha list, ROI, per-alpha descriptive pixel metrics and known limitations). Any mesh fold/invalid landmark should abort before a partial sequence is written.

## Acceptance criteria

1. **Engineering:** prior 129 tests pass, plus 27 geometry/CLI tests; output reference hashes match; no output outside mouth ROI; alpha 0 pixel-exact closed; no unintended editing of V0.6.2 files.
2. **Geometry:** hand-marked 2-D correspondences are checked on both images; no mesh fold; moving mouth corner/chin behaves coherently in 0.25/0.5/0.75 states.
3. **Visual:** compare against V0.6.2 at equal alphas. Intermediate teeth/tongue ghosting is reduced **without** a conspicuous facial seam, double mouth or background change. Record any failures.
4. **Temporal:** the `dominant` method switches its texture at alpha 0.5; if it visibly pops, **do not use in video**. Video, audio, phoneme and advanced occlusion handling are out of scope.

If there are no sufficiently reliable corresponding features because one jaw pose occludes the other, or the photographs differ in yaw/pose, **reject this pair** rather than claim geometric blending will solve it. The stable-photo constraint remains critical.

## Revert

```powershell
& $git apply --reverse --check .\liveportrait_v0.6.3_mouth_geometry.patch
& $git apply --reverse .\liveportrait_v0.6.3_mouth_geometry.patch
```

Only reverse if no later edits depend on these files; generated files under `build/` are separate and are not removed by reverse patch.
