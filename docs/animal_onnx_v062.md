# LivePortrait V0.6.2 — Dual-source mouth-state experiment

**Status: experimental image-space blend, NOT learned lip sync or a 3-D articulated jaw.**

## Why two source photos

V0.6/V0.6.1 showed that one closed-mouth source can produce unnatural generated teeth/tongue with large global or latent-KP motion. V0.6.2 instead requires two **photos of the same dog**, one naturally closed-mouth and one visibly open-mouth, with as closely matched head pose, scale, lighting, background and crop as possible. Ideally extract both from the same stable video.

This version deliberately does **not** switch complete frames. It holds the closed-source output everywhere outside a soft rectangular mouth ROI, and fades from closed to open appearance inside that ROI. The upper face may be aligned using a conservative 2-D translation derived from the top half of the cropped inputs. This is a proof-of-concept appearance composite. **It does not move the jaw, generate correct phonemes, learn multiple source states, or guarantee plausible in-between mouth positions.** In particular, fractional alphas may show ghosted tongues or doubled teeth.

The `--same-dog` flag is an **explicit user assertion**; no automatic identity validation takes place. Official s25 and s32 are **different dogs**, so they are not valid evidence for same-dog blending. Use `--different-dog-demo` only to test software behavior, never to evaluate mouth realism.

## New files

- `src/utils/animal_dual_source.py`: source ONNX reference reconstruction; same-run F/M cache; strict image and ROI checks; conservative upper-face-only translation; soft mouth blending; descriptive pixel-difference metrics.
- `scripts/run_animal_dual_source.py`: CLI, ONNX session reuse, `--plan`, two input modes, JSON reporting, PNGs and contact sheet.
- `tests/test_animal_dual_source.py`, `tests/test_animal_dual_source_cli.py`: CPU / mocked ONNX tests.
- `docs/animal_onnx_v062.md`: this file.

**All existing V0.6/V0.6.1 files and original LivePortrait inference entrypoints remain unchanged.** No ONNX weights or checkpoints are modified. No audio/video encoding.

## Patch installation on Windows 10/11 PowerShell

From `C:\Users\ES\Documents\GitHub\LivePortrait` with your prior **V0.6.1 patch applied**:

```powershell
conda activate liveportrait-onnx
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git branch --show-current
& $git status --short
& $git apply --check .\liveportrait_v0.6.2_dual_source.patch
& $git apply .\liveportrait_v0.6.2_dual_source.patch
python -m py_compile src/utils/animal_dual_source.py scripts/run_animal_dual_source.py
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

V0.6.1 reportedly passed **106 tests** in your environment. V0.6.2 adds **23 new tests**, therefore **129 expected** if the existing tests and environment are unchanged. New tests were exercised without actual model weights; true inference will need your existing F/M/W/G ONNX files.

## A) Plan (no ONNX execution)

```powershell
python scripts/run_animal_dual_source.py --plan
```

## B) Fast visual diagnostic using original photos (NO ONNX, ~seconds)

Prepare **same-dog** cropped, front-facing photos `assets/dog_closed.png` and `assets/dog_open.png` (paths below are examples). This mode directly blends original images, not model reconstructions.

```powershell
python scripts/run_animal_dual_source.py `
  --source-closed assets/dog_closed.png `
  --source-open assets/dog_open.png `
  --same-dog `
  --input-mode original `
  --alignment translation `
  --output-dir build/onnx_animal_v062/original
```

Open `build/onnx_animal_v062/original/contact_sheet.png` and `mouth_mask.png`. If alignment fails, improve the source poses/crops; for explicit manual visual inspection use `--alignment none` (risk: doubled mouth features).

## C) Two genuine ONNX reference reconstructions (model sessions loaded once)

```powershell
python scripts/run_animal_dual_source.py `
  --source-closed assets/dog_closed.png `
  --source-open assets/dog_open.png `
  --same-dog `
  --input-mode onnx `
  --threads 4 `
  --roi "0.29,0.49,0.74,0.83" `
  --feather-px 24 `
  --alphas "0,0.25,0.5,0.75,1" `
  --output-dir build/onnx_animal_v062/onnx
```

ONNX mode loads F/M/W/G **once**, for each unique photo runs `F + M + W + G` **once** to reconstruct a no-motion reference (source keypoints used as both W keypoint inputs); caches the F/M intermediate values and final RGB in the same process. All five alpha frames are then inexpensive **image-space** composites. It does not re-run W/G for every alpha. This is intentionally different from V0.6 single-source human driving.

### Experimental comparison with existing official sample (NOT same dog)

```powershell
python scripts/run_animal_dual_source.py `
  --source-closed assets/examples/source/s25.jpg `
  --source-open assets/examples/source/s32.jpg `
  --different-dog-demo `
  --input-mode original `
  --alignment none `
  --output-dir build/onnx_animal_v062/different_dog_diagnostic
```

**Never interpret this as successful same-dog lip sync**; it only demonstrates the image compositing API.

## Output semantics

| File | Meaning |
|---|---|
| `closed_reference.png` | Closed-source reconstruction (`onnx`) or resized original (`original`) |
| `open_reference_aligned.png` | Open-source reconstruction or original after upper-face translation |
| `mouth_mask.png` | Soft, spatial mouth ROI; white shows strongest open-source contribution |
| `mouth_000.png` | Exactly closed reference (alpha=0) |
| `mouth_250.png`, `mouth_500.png`, `mouth_750.png` | Appearance crossfades (can ghost tongue/teeth) |
| `mouth_1000.png` | Closed head/background plus aligned open *mouth* patch, not complete open face |
| `contact_sheet.png` | All alphas side-by-side |
| `report.json` | SHA256 of both source files, transform, ROI, alpha values, model stages, descriptive changes and limitations |

`--roi x0,y0,x1,y1` is **normalized to the 512×512 image**. Default is a rough location based on earlier Shiba Inu examples; **not a mouth detector**. Check the mask overlay yourself. In both modes the inputs are merely resized to 256×256 using the pre-existing V0.6 reader; animal detection and facial crop are **not** supplied. Adjust the ROI for your own dog, and use `--alignment none` only if you have externally aligned images. The registration mask covers the upper face and never intentionally fits the mouth; pose and scale differences are not automatically corrected.

The report includes `outside_region_changed_pixels`, which must be zero for each alpha; **this is a compositing invariant, not proof of high quality**. Neither alpha=0.5 nor alpha=1 guarantees half/open jaw angle. Keep identity and same-camera constraints in mind. The next phase would need local geometric warping and/or additional reference states for stable video/phoneme synthesis.

## Safety and reversibility

- Existing output files are never overwritten unless `--overwrite` is used. The flag only affects the script's predefined output names.
- Inputs are never overwritten and the output must stay inside the repository under a **different directory**.
- No external API, network call, CUDA or video generation occurs.
- To reverse **if no later work depends on this patch**:

```powershell
& $git apply --reverse --check .\liveportrait_v0.6.2_dual_source.patch
& $git apply --reverse .\liveportrait_v0.6.2_dual_source.patch
```
