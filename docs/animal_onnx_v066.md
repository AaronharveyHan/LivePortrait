# LivePortrait Animal CPU — V0.6.6 multi-frame cavity + anchored tongue

V0.6.6 is an **experimental** off-line RGB processing step built on the committed
V0.6.5 geometry_peak pipeline. It **does not** change ONNX F/M/W/G, checkpoints,
legacy inference entrypoints, automatic segmentation, or the verified annotation
formats. It is not audio lip-sync, 3D jaw articulation, or real occluded-mouth
reconstruction.

## Requirements

- Active branch: `perf/low-vram-animal-inference` with **V0.6.5 geometry_peak** committed.
- Environment: `conda activate liveportrait-onnx` (numpy, scipy, cv2, Pillow).
- Existing V0.6.2 `closed_reference.png` and `open_reference_aligned.png` from
  **original mode** (matching raw intermediates), V0.6.3 verified endpoints,
  V0.6.4 verified middle landmark JSON, V0.6.5 verified tongue polygon JSON.
- Keep **the exact same original image files** used when exporting annotations.
  Both V0.6.4 and V0.6.5 validators check their SHA-256 fingerprints.

## Install and verify

```powershell
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git apply --check .\liveportrait_v0.6.6_cavity_root.patch
& $git apply .\liveportrait_v0.6.6_cavity_root.patch
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

## First run — compare with V0.6.5 on your dog

```powershell
$pair = 'build/onnx_animal_v062/yawn_pair'
$ref  = 'build/onnx_animal_v062/yawn_preview'
$lm   = 'build/onnx_animal_v063/yawn_annotation/mouth_landmarks_verified.json'
$mp   = 'build/onnx_animal_v064/yawn_annotation/mouth_mid_landmarks_verified.json'
$ta   = 'build/onnx_animal_v065/yawn_annotation'

python scripts/run_animal_mouth_v066.py `
  --reference-dir $ref `
  --landmarks $lm `
  --mid "0.333333=$pair/frame_01.png" `
  --mid "0.666667=$pair/frame_02.png" `
  --mid-points $mp `
  --tongue-manifest "$ta/mouth_tongue_polygons_verified.json" `
  --same-dog `
  --opening-mode geometry_peak `
  --openings '0,0.25,0.5,0.75,1' `
  --tongue-values '0,0,0.2,0.6,1' `
  --donor-strength 0.9 `
  --root-depth 0.58 `
  --root-feather-px 9 `
  --shadow-strength 0.12 `
  --output-dir build/onnx_animal_v066/yawn_cavity_root
```

Inspect:

```powershell
mspaint.exe 'build/onnx_animal_v066/yawn_cavity_root/comparison_sheet.png'
```

Three rows:

1. `v065_baseline`: unchanged V0.6.5 jaw+independent tongue.
2. `v066_cavity_only`: new donor-enhanced tongue-free mouth base.
3. `v066_rooted_tongue`: new jaw base with tongue fixed at a jaw-relative root.

Additional files: `cavity_ref_XX.png`, `donor_confidence_XX.png` (grayscale;
bright means other reference pixels contributed), `tongue_root_alpha_XXX.png`,
per-opening PNGs, `report.json`. `cavity.references[].donor_coverage_fraction`
and `telea_fallback_fraction_estimate` are proxy diagnostics, **not proven
physical correctness**. If no donor covers a region, Telea fills it.

## Controlled ablation tests

Run **separate output directories** (never overwrite original images):

- `--donor-strength 0`: old Telea cavity only, but still with the new anchored tongue.
- `--donor-strength 1`: maximal visible donor contribution.
- `--root-depth 0.45` vs `--root-depth 0.7`: tongue-root location relative
  to upper/lower lip. Manual anatomical correctness is not assumed.
- `--root-feather-px 4` vs `--root-feather-px 16`: root seam fading.
- `--shadow-strength 0`: disable synthetic contact shadow.
- `--cavity-method layered_nearest`: sharpest reference policy versus
  `layered_flow` (default). Both may suffer temporal artifacts.

`--strict-continuity` can enforce `--max-opening-step`, `--max-tongue-step`
and `--max-roi-mae`; metrics alone **do not** prove perceptual realism.
`--holdout OPENING=FILE` additionally requires an unseen independent image,
not used as a reference or mask source. Its jaw opening and tongue amount must
be assigned independently. Holdout ROI MAE is reported against V0.6.5, but
brightness, alignment, occlusions and target-control uncertainty affect it.

## Safety and scientific limitations

- Source images and verified JSONs are never overwritten. Existing outputs
  require a distinct destination or explicit `--overwrite`.
- No changes to face, ears, fur or background outside the mouth ROI.
- Donor-aware reconstruction samples verified **tongue-free** pixels inside
  estimated landmark mouth cavities; it can accidentally copy teeth, fur or
  other wrong depth layers. `donor_confidence_XX.png` is evidence provenance,
  not semantic certainty. Telea remains a fallback.
- Tongue root is estimated from the top of the user-drawn tongue polygon,
  then anchored to a relative upper/lower lip point; it is **not** a 3D root
  joint or a verified anatomical landmark.
- `geometry_peak` only models the opening branch through the measured peak
  (Frame 2 in the Yawn.gif experiment). Post-peak Frame 3 can still be an
  appearance donor, but is NOT a modeled closing animation.
- Most important acceptance gate: visually compare 25/50/75% lip continuity,
  root attachment, cavity authenticity, donor map and temporal jumps on your
  actual unmodified photos, plus one independent holdout if available.
