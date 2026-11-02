# LivePortrait Animal ONNX V0.4 (experimental)

Base: `perf/low-vram-animal-inference` V0.3 commit `52fd65623a07cebbe595f2f2613f035d1b4ea611`.

## Scope

- Adds **new files only**; does not change the animal inference pipeline, `main`, checkpoints or runtime backend.
- Defines fixed **animal v1.1 / batch 1 / FP32 / input 256 × 256** contracts for F, M, W, G.
- `--plan` requires **only standard-library Python** (no CUDA, torch, ONNX or weights).
- Actual export requires compatible PyTorch (prefer a recent version with the dynamo exporter), PyYAML, ONNX and onnxscript. `--validate-runtime` additionally needs onnxruntime and NumPy.
- V0.4 is **an export experiment**, not a claim of successful complete ONNX inference or low-VRAM acceleration.

## Contracts

| Model | Inputs | Outputs | Export readiness |
|---|---|---|---|
| F | `source_image` `[1,3,256,256]` | `feature_3d` `[1,32,16,64,64]` | Try first |
| M | `image` `[1,3,256,256]` | `pitch/yaw/roll` `[1,66]`, `t` `[1,3]`, `exp` `[1,63]`, `scale` `[1,1]`, `kp` `[1,63]` | Separate export |
| W | `feature_3d`, `kp_driving` `[1,21,3]`, `kp_source` `[1,21,3]` | `warped_feature` `[1,256,64,64]` | **Experimental**; 5D GridSample |
| G | `warped_feature` `[1,256,64,64]` | `output_image` `[1,3,512,512]` | Separate export |

M exports **raw** network outputs, not the postprocessed degrees and keypoint reshaping in `LivePortraitWrapper.get_kp_info`. W exports only `ret_dct['out']` (the feature actually consumed by G), not the diagnostic deformation/occlusion tensors. W positional input order is `feature_3d, kp_driving, kp_source`.

## Installation and offline tests (Windows PowerShell)

```powershell
conda activate liveportrait-test
python .\apply_liveportrait_low_vram_v04.py --dry-run
python .\apply_liveportrait_low_vram_v04.py
python -m py_compile src/utils/animal_onnx_contract.py scripts/export_animal_onnx.py
python -m unittest discover -s tests -p 'test_animal_*py' -v
python scripts/export_animal_onnx.py --plan --model all
```

Expected: 12 existing tests + 15 new tests = **27 tests** if V0.2/V0.3 tests are intact. The `--plan` command is deliberately possible in `liveportrait-test`, which does not have PyTorch installed. It reports checkpoint availability without opening the files.

## Optional real export (separate CPU PyTorch environment)

```powershell
# Use a separate, compatible CPU PyTorch environment; do not install
# the large framework into liveportrait-test just to run --plan.
python -m pip install pyyaml onnx onnxscript onnxruntime numpy

# Export one model at a time; animal v1.1 weights must already exist.
python scripts/export_animal_onnx.py --model f --validate-runtime
python scripts/export_animal_onnx.py --model m --validate-runtime
python scripts/export_animal_onnx.py --model g --validate-runtime

# W is higher-risk and requires a recent exporter/runtime supporting
# volumetric GridSample (5D). Do not assume successful parity.
python scripts/export_animal_onnx.py --model w --experimental-w --opset 20 --validate-runtime
```

These commands also require **CPU PyTorch**, installed separately using the official CPU wheel appropriate to your Python and OS; the pip line above intentionally does not install torch. Export uses model checkpoint paths under `pretrained_weights/liveportrait_animals/base_models_v1.1` and writes to `build/onnx_animal_v04`. Do not stage binary ONNX outputs or weight files unless explicitly intended.

- Default exporter is `dynamo` with `opset 20`; it may require newer compatible PyTorch / ONNX packages. Different torch versions may produce different export success rates.
- `--model all` includes W and is **rejected** unless `--experimental-w` is also present.
- `--validate-runtime` performs an optional ONNX Runtime CPU parity check with default `atol=rtol=1e-3` against PyTorch for sample tensors; this is **not** an end-to-end animation quality test.
- For `W`, ONNX `GridSample` v20 and newer specify volumetric input; an older exporter may generate an invalid 3D-grid graph. This is why W is opt-in and runtime validation is encouraged.
- Some large models require substantial CPU RAM/time during export, even if inference will later run on a smaller GPU. Export models individually.

## Safe Git commit

```powershell
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git status --short
& $git add src/utils/animal_onnx_contract.py scripts/export_animal_onnx.py tests/test_animal_onnx_contract.py tests/test_animal_onnx_export_cli.py docs/animal_onnx_v04.md
& $git diff --cached --check
& $git commit -m "feat(animal): add ONNX export contract and CPU-safe planning"
& $git push origin perf/low-vram-animal-inference
```

Only run the commit commands after reviewing the staged diff. The installer itself is not committed by default.
