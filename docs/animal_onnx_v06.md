# LivePortrait Animal ONNX V0.6: single-frame CPU inference (experimental)

Target repository: `AaronharveyHan/LivePortrait`, branch `perf/low-vram-animal-inference`.
Base commit **`04c97c123c6fd1c95176f3439b33b585df7b0abf`** (V0.5). This patch **adds files only**.
No changes to `inference_animals.py`, `LivePortraitPipelineAnimal`, model weights, default branches or runtime switches.

## Scope and mathematical behavior

The standalone chain executes real **animal v1.1, batch=1 FP32, CPU ONNX Runtime** graphs:

1. `F(source_image)` -> `feature_3d` `[1,32,16,64,64]`;
2. `M(source)` -> canonical source kp, pose, scale, expression, translation;
3. `M(driving_first)` and `M(driving_target)` -> first-frame anchor and target driving motion;
4. Convert M's 66-bin headpose logits using softmax with degrees `sum(p_i * i)*3 - 97.5`, compute row-vector rotation `R = (Rz @ Ry @ Rx).T`;
5. Compute `x_s = scale_s * (kp_s @ R_s + exp_s) + t_s.xy`;
6. Compute `x_d0` and `x_di` with source canonical keypoints and source scale, but driving pose/expression/translation;
7. Use relative motion `x_d = x_s + (x_di - x_d0) * sqrt(ConvexHull(x_s).volume) / sqrt(ConvexHull(x_d0).volume)`, then apply `driving_multiplier` around `x_s`. **SciPy ConvexHull is 3D volume, NOT 2D surface area**;
8. `W(feature_3d, kp_driving=x_d, kp_source=x_s)` -> `[1,256,64,64]`;
9. `G(warped_feature)` -> `[1,3,512,512]`, convert to RGB uint8 PNG.

Matches the **relative-motion/no-stitching** branch in `src/live_portrait_pipeline_animal.py` and `transform_keypoint` from `src/live_portrait_wrapper.py`.
It is NOT equivalent to arbitrary original GUI/CLI settings.

### Hard limitations (important)

- No face/animal detector, no crop or facial tracking. Supply **three pre-cropped/aligned animal face images**, or the `--synthetic` smoke mode. Non-cropped photos may generate unusable results.
- **No Stitching S, paste-back, audio or MP4/GIF output.** The original animal inference defaults may enable stitching; turn it off when comparing to V0.6. This patch never loads a human retargeting checkpoint.
- `--driving-first` is the animation anchor, `--driving-target` the frame to animate toward. Using the **same driving frame for both produces zero relative movement by design**.
- `--synthetic` uses deterministic random RGB inputs for a graph smoke test. It **does not assess visual quality** or motion realism.
- CPU-only. GT 1030 (2 GB) remains untested. Four ONNX sessions are loaded at once and peak RAM may be high.
- No direct PyTorch end-to-end pixel parity or real-dog visual comparison has yet been performed. The V0.4/V0.5 per-model parity checks remain valid but do not guarantee full-chain visual parity.
- The original pipeline uses OpenCV resize. This script tries OpenCV first, then falls back to Pillow bilinear resize; pixel-level differences from preprocessing may arise.

## Install and basic tests (PowerShell)

Save `apply_liveportrait_low_vram_v06.py` to the **repo root**.

```powershell
cd C:\Users\ES\Documents\GitHub\LivePortrait
conda activate liveportrait-onnx
python .\apply_liveportrait_low_vram_v06.py --dry-run
python .\apply_liveportrait_low_vram_v06.py
python -m py_compile src/utils/animal_onnx_chain.py scripts/run_animal_onnx_frame.py
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

V0.6 adds **24** `test_animal_*py` tests (one requires SciPy, one additionally checks against upstream Torch `camera.py` when available). With 58 previously passing tests the expected total is **82**. The latter test is skipped where CPU PyTorch is absent; SciPy test is skipped if SciPy is missing. The 24 include 6 CLI tests that run Python with `-S` to prove `--plan` works without any site-packages.

## Install two lightweight dependencies into `liveportrait-onnx`

You have already installed CPU PyTorch 2.10.0, ONNX 1.23.2 and ORT 1.23.2. This standalone script additionally uses:

```powershell
conda activate liveportrait-onnx
python -m pip install scipy pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If `opencv-python-headless` is already installed it is preferred for image decoding/resizing; this is optional. Pillow is used to save PNGs.

## Check all 4 exported ONNX graphs

```powershell
python scripts/run_animal_onnx_frame.py --plan
```

Expected on the previously validated machine: four `onnx_ready: true` entries, IF all four `.onnx` files and `.json` manifests remain in `build/onnx_animal_v04/`.
This check is metadata-only: it does not load graphs.

## First REAL four-graph ONNX execution using deterministic synthetic RGB

```powershell
python scripts/run_animal_onnx_frame.py `
    --synthetic `
    --threads 4 `
    --output build/onnx_animal_v06/smoke.png `
    --report-json build/onnx_animal_v06/smoke.json
```

This performs F once, M three times, W once, G once and generates a PNG and JSON report. It is a real ONNX inference run using meaningless RGB noise, **not a real dog animation test**. Repeating the same output filenames requires `--overwrite`.

The JSON report includes `F_source`, `M_source`, `M_first`, `M_target`, `motion_postprocess`, `W`, `G`, `RGB_output` stage times; total inference time; session initialization time; ConvexHull ratio; and optional RSS memory (install `psutil` to enable). Peak RSS is NOT measured in this version; the three RSS samples are snapshots only.

## Actual pre-cropped dog/driver frame example

The inputs MUST all be crop-aligned beforehand, and are normalized to 256x256 RGB.

```powershell
python scripts/run_animal_onnx_frame.py `
    --source assets/my_dog_crop.png `
    --driving-first assets/driver_frame_000.png `
    --driving-target assets/driver_frame_015.png `
    --threads 4 `
    --output build/onnx_animal_v06/dog_015.png `
    --report-json build/onnx_animal_v06/dog_015.json
```

Replace placeholder input paths with your real photos; the script will **reject missing input files**. A still photo alone cannot show a temporal driving difference.

## Troubleshooting

- `--plan` reports missing ONNX/manifest: rerun `python scripts/validate_animal_onnx.py --mode preflight --model all`; do not reinstall the models until you know what's missing.
- `No module named scipy`: `python -m pip install scipy`; relative motion uses 3D ConvexHull.
- `No module named PIL` or image decode failure: `python -m pip install pillow`, or install `opencv-python-headless`.
- `Degenerate 3D keypoints`: source and first-driving hull 3D volume must be nonzero. This is detected and reported rather than guessing a multiplier.
- ONNX Session contract mismatch: re-export the named model using the V0.4 tool; do not ignore the mismatch.
- Slow first run: CPU four-graph session initialization and W/G computation can be significant. The JSON separates session setup from inference.
- High RAM usage or thermal/performance instability: lower `--threads`, close other heavy applications. 4 threads are an initial conservative baseline, **not an established optimal CPU setting**.

## Safe commit (after user-side verification)

```powershell
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git status --short
& $git add src/utils/animal_onnx_chain.py scripts/run_animal_onnx_frame.py tests/test_animal_onnx_chain.py tests/test_animal_onnx_chain_cli.py docs/animal_onnx_v06.md
& $git diff --cached --check
& $git commit -m "feat(animal): add experimental ONNX CPU single-frame chain"
& $git push origin perf/low-vram-animal-inference
```

**Do not commit** `pretrained_weights/`, `build/onnx_animal_v04/`, `build/onnx_animal_v06/` or huge external-data weights accidentally. The V0.6 installer itself is not automatically committed.
