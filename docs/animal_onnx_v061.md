# LivePortrait Animal ONNX V0.6.1 — experimental latent-keypoint motion controls

Applies **on top of V0.6** on branch `perf/low-vram-animal-inference` (V0.6 standalone 4-graph chain). This is an **ordinary unified Git patch**; it does not edit the original animal animation pipeline, weights, ONNX files or the default Git branch.

## Key constraint: latent points are not mouth landmarks

Animal v1.1's **21 implicit learned keypoints have no verified index-to-mouth or jaw map**. `mouth_kp_indices` and `lower_mouth_kp_indices` in this experimental feature mean **user-selected latent points**, not proven anatomical features. A point may affect eyes, nose, cheeks or unrelated regions. **There is deliberately no hardcoded default “mouth” preset.**

The earlier experiment showed that globally increasing `--driving-multiplier` from 1.0 to 1.15–1.50 distorted the Shiba Inu's nose/mouth. The new feature keeps global motion at 1.0 and optionally modifies **only selected points** after relative-motion calculation and before the W ONNX graph. It cannot create physical jaw structure that the trained model cannot synthesize.

### Math

For selected implicit point `i` and axis `c`:

`kp_new[i,c] = kp_source[i,c] + scale[c] * (kp_driving[i,c] - kp_source[i,c])`.

For `lower_mouth_kp_indices`, additionally:

`kp_new[i,1] += signed_lower_y_bias`.

The Y axis is a learned 3-D coordinate, **not a guaranteed anatomical down direction**. Start with signed tiny offsets and inspect results. Nonselected points are bitwise unchanged. Both incoming keypoint arrays are left unmodified. Defaults yield identical inference behavior to V0.6.

## Patch application (PowerShell)

From `C:\Users\ES\Documents\GitHub\LivePortrait`:

```powershell
conda activate liveportrait-onnx
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git branch --show-current
& $git status --short
& $git apply --check .\liveportrait_v0.6.1_local_mouth.patch
& $git apply .\liveportrait_v0.6.1_local_mouth.patch
python -m py_compile src/utils/animal_onnx_chain.py src/utils/animal_mouth_enhancement.py scripts/run_animal_onnx_frame.py
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

Expected **106 total tests** if the user's previously passing **82 V0.6 tests** remain present and unchanged; this patch adds 24 tests. If the GitHub Desktop internal git is installed elsewhere, use the correct path or `git` on PATH. `git apply --check` does not mutate files; a failed check means your V0.6 files differ from the originals used by this patch. **Do not use `git apply --reject` or force replacement without resolving differences.**

To reverse *only if there are no subsequent edits depending on it*:

```powershell
& $git apply --reverse --check .\liveportrait_v0.6.1_local_mouth.patch
& $git apply --reverse .\liveportrait_v0.6.1_local_mouth.patch
```

The V0.6 source itself may still be locally uncommitted even if V0.5 is pushed. Verify the branch and `git status --short` before applying; this patch expects **the V0.6 files to exist**. The `.patch` file need not be committed.

## A) Confirm V0.6 compatibility (NO enhancement)

Use a fresh output filename and keep the known-good 1.0x image for comparison:

```powershell
$dir = "build/onnx_animal_v06/official_dog_demo"
python .\scripts\run_animal_onnx_frame.py `
  --source "$dir/dog_source.png" `
  --driving-first "$dir/driver_first.png" `
  --driving-target "$dir/candidate_0239.png" `
  --threads 4 `
  --output "$dir/v061_default.png" `
  --report-json "$dir/v061_default.json" `
  --dump-kp-json "$dir/v061_default_kp.json"
```

This must match the previous V0.6 output (`out_candidate_0239.png`) on the same inputs, subject to deterministic FP32 runtime behavior. This run also records `source`, `driving_before_mouth`, `driving_after_mouth`, and `delta_after_minus_before` (all [1,21,3]). All enhanced deltas should be zero with defaults.

## B) Find candidate point indices by visual ablation — no anatomy assumptions

**Never assume index 0 (or any other index) is the mouth.** For an exploratory influence scan, perturb **one index at a time**, holding the images fixed, and inspect which face region changes. The following is deliberately a **short** sample (0, 1, 2) and does NOT imply they are mouth points:

```powershell
foreach ($i in @(0, 1, 2)) {
  python .\scripts\run_animal_onnx_frame.py `
    --source "$dir/dog_source.png" `
    --driving-first "$dir/driver_first.png" `
    --driving-target "$dir/candidate_0239.png" `
    --threads 4 `
    --mouth-kp-indices "$i" `
    --mouth-lower-kp-indices "$i" `
    --mouth-lower-y-bias 0.005 `
    --output "$dir/probe_kp_$i.png" `
    --dump-kp-json "$dir/probe_kp_$i.json"
  if ($LASTEXITCODE -ne 0) { throw "Probe $i failed" }
}
```

Then try **negative** `--mouth-lower-y-bias -0.005` if necessary: the sign is not necessarily anatomically down. Only after finding reliable candidates, select them together for a targeted scale sweep. Running all 21 indices would be slow (each launches the full W/G pipeline); do not run many sessions concurrently on limited hardware.

## C) Test a *verified experimental* candidate set

**Example indices `5,7` below are placeholders only, not a confirmed mouth map; substitute your discovered point IDs.**

```powershell
python .\scripts\run_animal_onnx_frame.py `
  --source "$dir/dog_source.png" `
  --driving-first "$dir/driver_first.png" `
  --driving-target "$dir/candidate_0239.png" `
  --driving-multiplier 1.0 `
  --mouth-kp-indices "5,7" `
  --mouth-y-scale 1.15 `
  --mouth-lower-kp-indices "7" `
  --mouth-lower-y-bias 0.005 `
  --threads 4 `
  --output "$dir/v061_experiment.png" `
  --report-json "$dir/v061_experiment.json" `
  --dump-kp-json "$dir/v061_experiment_kp.json"
```

The JSON report records configuration and latency of an optional `mouth_enhancement` stage. Dumps do **not** infer facial anatomy or reveal pose by themselves. The tool refuses duplicate, out-of-range, nonfinite or inconsistent index/scale inputs; nonzero Y bias requires the lower indices to be a subset of the selected indices. Scale values are constrained to 0..3, signed bias to -0.05..0.05 as a conservative experiment bound. Never interpret these latent units as physical pixels.

## Baseline and safety

- Keep the **V0.6 global-multiplier 1.0** result as the reference. The 1.15–1.50 *global* multiplier sweep already looked worse; do **not** treat the new local multiplier as a validated improvement yet.
- Compare source / baseline / default / altered PNGs at actual size, and look especially for misplaced mouth, face swelling, lip artifacts, nose distortion or temporal instability.
- This is a **single-frame ONNX experiment**: no lip-sync from audio, no true mouth-retargeting model, no crop, no S stitching, no video output. No claim of natural dog speech.
- Do not commit large weights or ONNX artifacts into the repository.
