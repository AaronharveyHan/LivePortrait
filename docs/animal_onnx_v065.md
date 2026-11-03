# LivePortrait animal CPU V0.6.5 — independently controlled jaw and tongue

**Base:** `AaronharveyHan/LivePortrait`, branch `perf/low-vram-animal-inference`, V0.6.4 commit `9c9206a`.

V0.6.5 is a **standalone CPU-only 2D appearance experiment**. It does **not** modify ONNX F/M/W/G, model weights, the V0.6.2 dual-photo CLI, V0.6.3 geometric CLI, or the V0.6.4 multi-reference CLI.

## Changes vs V0.6.4

1. **Measured opening** (default `--opening-mode geometry`): compute a dimensionless 2D jaw aperture = perpendicular separation between verified `upper_lip` and `lower_lip` divided by width between mouth corners. Normalize the four reference measurements to `[0,1]` and map requested opening values into the corresponding intervals of original frame time. If measurements are non-monotonic, **reject** rather than claim evenly spaced jaw states. `--opening-mode time` is only a labelled diagnostic fallback.
2. **Independently marked tongue**: an offline browser annotator requires a hand-drawn tongue polygon or a deliberate “no visible tongue” choice on **each aligned reference frame**. It exports verified normalized coordinates with SHA256 references for the exact input image files; the renderer rejects stale/mismatched annotations.
3. **Two separate appearance layers**: per-reference tongue regions are removed from the mouth background by OpenCV classical inpainting (approximate, may be wrong); the V0.6.4 geometric cavity pipeline runs on those photos. The tongue is then independently scaled and positioned relative to the mouth landmarks with a feathered mask. `--tongue-values` is independent of `--openings`.
4. **Continuity audit**: record each step's opening delta, tongue delta, and mouth ROI RGB MAE. `--strict-continuity` stops before writing output if any threshold fails; it is a **rate/diagnostic constraint**, not an auto-smoothing model.
5. **Independent holdout evaluation**: optionally compare V0.6.4 baseline and V0.6.5 predictions to a frame whose content is absent from **all** references and tongue masks. Pixel MAE is descriptive; input registration and control values influence it.

## Requirements

Use the existing `liveportrait-onnx` Conda environment. Python 3.10+, NumPy, SciPy, Pillow, and `opencv-python-headless` (or `opencv-python`). ONNX Runtime is **not used** by this command. All images must show **the same dog**, with comparable pre-crop and angle. V0.6.3 endpoint landmarks and V0.6.4 intermediate landmarks must be human-verified.

## 1. Apply the Patch (once)

```powershell
cd C:\Users\ES\Documents\GitHub\LivePortrait
conda activate liveportrait-onnx
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git status --short
& $git apply --check .\liveportrait_v0.6.5_independent_jaw_tongue.patch
& $git apply .\liveportrait_v0.6.5_independent_jaw_tongue.patch
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

## 2. Generate an offline tongue polygon annotator

The reference photos are **the same four images used in V0.6.4**, and the mid-point annotations you already exported can be reused. **Important:** `--mid` values are original **frame-time labels**. They are **not** assumed to equal jaw opening percentages.

```powershell
$pair = "build/onnx_animal_v062/yawn_pair"
$ref  = "build/onnx_animal_v062/yawn_preview"
$lm   = "build/onnx_animal_v063/yawn_annotation/mouth_landmarks_verified.json"
$mp   = "build/onnx_animal_v064/yawn_annotation/mouth_mid_landmarks_verified.json"
$ta   = "build/onnx_animal_v065/yawn_annotation"

python scripts/run_animal_mouth_v065.py `
    --reference-dir $ref `
    --landmarks $lm `
    --mid "0.333333=$pair/frame_01.png" `
    --mid "0.666667=$pair/frame_02.png" `
    --mid-points $mp `
    --same-dog `
    --init-tongue `
    --output-dir $ta
```

Open it in Chrome:

```powershell
$html = (Resolve-Path "$ta/annotate_tongue.html").Path
Start-Process chrome.exe -ArgumentList ([System.Uri]$html).AbsoluteUri
```

For **all four frames**, draw tongue-only polygons (8–20 clicks are usually easier than 3). Avoid teeth, lip, nose and whole mouth cavity. On Frame 0, which seems to have no tongue, choose **本帧无舌头**. For every frame, either click **确认本帧多边形** or **本帧无舌头**. Export `mouth_tongue_polygons_verified.json`, copy it into `$ta`.

**Do not manually set `verified=true` without marking each frame.** The editor tracks individual confirmation in the browser, and the Python renderer checks geometry, SHA256, and frame counts. The exported HTML contains photos inline, loads no external scripts or images, and works offline.

## 3. Render independent jaw/tongue controls

```powershell
python scripts/run_animal_mouth_v065.py `
    --reference-dir $ref `
    --landmarks $lm `
    --mid "0.333333=$pair/frame_01.png" `
    --mid "0.666667=$pair/frame_02.png" `
    --mid-points $mp `
    --tongue-manifest "$ta/mouth_tongue_polygons_verified.json" `
    --same-dog `
    --opening-mode geometry `
    --openings "0,0.25,0.5,0.75,1" `
    --tongue-values "0,0,0.2,0.6,1" `
    --output-dir build/onnx_animal_v065/yawn_independent
```

`--openings` and `--tongue-values` must have **the same number** of entries. For example, if `--openings` is fixed at `0.5`, changing tongue amount `0.2` → `0.8` changes tongue appearance **without changing the target jaw opening**. The default tongue curve is a **hand-chosen demonstration** that delays tongue emergence until opening ≥ 0.25; you can override it.

View `build/onnx_animal_v065/yawn_independent/comparison_sheet.png`. The three rows show: **V0.6.4 Layered Flow**, **V0.6.5 cavity without tongue**, and **V0.6.5 independent tongue**. Also inspect per-frame `tongue_alpha_*.png`, `tongue_reference_mask_*.png`, and `cavity_ref_*.png` to spot incorrect segmentation and inpainting.

`report.json` records `reference_aperture_width_ratios`, `reference_openings_normalized`, mapping back to original time, selected tongue source, controls, changed pixels outside ROI, and continuity audit. If `opening-mode geometry` rejects measurements, inspect `upper_lip`, `lower_lip` and mouth corners. The fallback `--opening-mode time` is a **diagnostic**, not a calibrated opening test.

## 4. Continuity guard

```powershell
python scripts/run_animal_mouth_v065.py ... `
    --strict-continuity `
    --max-opening-step 0.30 `
    --max-tongue-step 0.50 `
    --max-roi-mae 32 `
    --output-dir build/onnx_animal_v065/strict_trial
```

Use the complete command from §3 in place of `...`. The guard compares *consecutive rendered frames*, not a learned realism metric. A high step can be reduced by adding intermediate requested openings and tongue values, or by improving alignment, polygon outlines, or geometry. Lower limits may reject valid fast physical motion; passing does **not** prove natural motion.

## 5. Held-out frame testing

To evaluate, **remove the held-out photo from every `--mid` reference, its geometry manifest, and tongue annotation**. Recreate the intermediate landmark JSON and tongue manifest for only the remaining frames. Then pass `--holdout "OPENING=path_to_unseen.png"` with the externally specified opening control and optionally `--holdout-tongue 0.5` (externally specified tongue control). The renderer refuses identical binary content even if the holdout was copied under a new filename. The ROI RGB MAE is **not** an anatomical or lip-sync score. Do not claim independent validation if the held-out image was included in references, masks, or geometry choices during the trial.

## 6. Output and known limitations

Outputs: `comparison_sheet.png`, `report.json`, `mouth_v064_flow_*.png`, `mouth_v065_cavity_*.png`, `mouth_v065_tongue_*.png`, `tongue_alpha_*.png`, `cavity_ref_*.png`, `tongue_reference_mask_*.png`.

- Jaw aperture is only a **2D landmark ratio**. Not a physical 3D jaw articulation or automatic phoneme mapping.
- Tongue polygons are drawn manually. The cavity under an occluding tongue **cannot be recovered truthfully** from a photo; classical inpainting may leave soft/incorrect texture.
- The tongue texture is scaled/translated approximately. No tongue-root tracking, topology-changing surface rendering or true occlusion. Users may still see popping, lip/tongue overlap or visibly artificial stretching.
- Independent tongue control allows physically impossible combinations (fully closed jaw with a large visible tongue). Select appropriate controls or implement an explicit geometric constraint later.
- The parser and asset files do not change any older source code, model graph, or checkpoint.
