# LivePortrait Animal V0.5 — ONNX Runtime CPU 驗證與效能基準（實驗功能）

基準分支：`perf/low-vram-animal-inference`。基準提交：`44af66b28b2021d21cd1e076f5dc7b695c5f8ccd`（V0.4）。

## 適用範圍

V0.5 **只新增工具**，不替換原有 LivePortrait 推理、不修改 F/M/W/G 模型結構、不更改 checkpoint。支持固定的 animal v1.1、batch=1、FP32、256×256 合約。

- `preflight`：只有 Python 標準庫也可運行。檢查 F/M/W/G 權重文件是否存在、非空，ONNX 模型與 V0.4 manifest 是否存在、符合接口聲明。**不校驗實際權重內容、ONNX 算子支持或數值正確性**。
- `benchmark`：使用 ONNX Runtime **CPUExecutionProvider** 執行指定 ONNX 模型，輸出延遲的 `mean/p50/p95/min/max`；依賴 `numpy` 和 `onnxruntime`，無需 PyTorch、模型權重或 CUDA。
- `compare`：載入真實 animal v1.1 `.pth`，在 CPU 上執行 PyTorch 模型，並對同樣的輸入運行 ONNX Runtime，比對形狀、float32 型別、最大/平均絕對誤差、相對誤差、超出容差元素數量。依賴權重、CPU PyTorch、PyYAML、NumPy 和 ONNX Runtime；不需要 CUDA。預設 `atol=1e-3, rtol=1e-3`，**誤差合格不代表動畫主觀品質合格**。

W 模型存在 5D `GridSample` 兼容性風險，執行 `compare/benchmark` 需要 `--experimental-w`；只列出 `preflight` 不需要。

## 依賴和準備

只使用你的現有輕量測試環境驗證新工具，**不要為了 `preflight` 安裝 PyTorch**。

```powershell
conda activate liveportrait-test
python -m unittest discover -s tests -p 'test_animal_*py' -v
python scripts/validate_animal_onnx.py --mode preflight --model all
```

如四個 `.pth` 不在默認的 `pretrained_weights/`，可指定實際權重根目錄（應包含 `liveportrait_animals/base_models_v1.1/`）：

```powershell
python scripts/validate_animal_onnx.py --mode preflight --model all `
  --weights-root 'D:\Models\LivePortrait\pretrained_weights' `
  --onnx-dir build/onnx_animal_v04 `
  --output-json build/onnx_animal_v05/preflight.json
```

此時報告中 `ready_compare=false` 是正常的：它表示必要文件尚未全部齊備或 manifest 校驗未通過。`--strict` 可以讓文件缺失時返回退出碼 2，適用於 CI。

## 實際模型導出與驗證

**前提**：先備妥 animal v1.1 權重，並在一個獨立的 CPU 模型環境中安裝與現有模型兼容的 PyTorch、PyYAML、ONNX、onnxscript、NumPy、ONNX Runtime。Windows 上可使用獨立的 Conda 環境；`onnxruntime` 等包可以使用清華 PyPI 鏡像，CPU PyTorch 的版本與安裝來源需遵循對應官方 wheel 支持矩陣。

使用 V0.4 導出工具優先單獨導出 F：

```powershell
python scripts/export_animal_onnx.py --model f --validate-runtime
```

成功生成 `build/onnx_animal_v04/animal_f_v1.1.onnx` 和 `.json` 後：

```powershell
# PyTorch / ONNX Runtime CPU 數值一致性
python scripts/validate_animal_onnx.py --mode compare --model f `
  --output-json build/onnx_animal_v05/f_parity.json

# 不需 PyTorch 或權重，只需已導出的 ONNX / manifest
python scripts/validate_animal_onnx.py --mode benchmark --model f `
  --threads 2 --warmup 1 --repeat 5 `
  --output-json build/onnx_animal_v05/f_latency.json
```

單模型測試通過後可依序嘗試 M、G，再嘗試 W：

```powershell
python scripts/export_animal_onnx.py --model m --validate-runtime
python scripts/export_animal_onnx.py --model g --validate-runtime
python scripts/export_animal_onnx.py --model w --experimental-w --opset 20 --validate-runtime
python scripts/validate_animal_onnx.py --mode compare --model w --experimental-w
```

一次處理四個模型將增加 RAM 和執行時間，建議 **一次執行一個模型**。測試小樣本的平均延遲不能替代真實視頻端到端吞吐、峰值 RAM 或 2GB 顯存設備上的性能。

## 報告語義

- `ready_benchmark=true`：ONNX 文件非空且 manifest 與接口契約一致；**仍未證明 ONNX 模型內容能真正執行**。
- `ready_compare=true`：另外要求存在非空 `.pth` 權重；**沒有檢查權重 hash、來源可信性或 checkpoint 結構**。
- `passed=true`（`benchmark`）：當次 ONNX Runtime CPU 輸出形狀/型別驗證成功；不代表性能已達標。
- `passed=true`（`compare`）：每個輸出張量均通過 `abs(a-b) <= atol + rtol*abs(a)`，且無 NaN/Inf。任何失敗使退出碼為 3。
- 非權重／缺少導出文件等準備錯誤返回退出碼 2；異常不會被誤報為成功。

報告為 JSON，除非顯式指定 `--output-json`，工具不會另存報告文件。

## 源碼構成

| 路徑 | 用途 |
| --- | --- |
| `src/utils/animal_onnx_validation.py` | 純 Python 檔案/manifest 檢查、ORT I/O 契約、數值指標、延遲統計 |
| `scripts/validate_animal_onnx.py` | preflight / benchmark / compare 命令行 |
| `tests/test_animal_onnx_validation.py` | 純 CPU 測試；含 NumPy，無 torch / ORT |
| `tests/test_animal_onnx_validate_cli.py` | `python -S` 標準庫模式和失敗安全測試 |
| `docs/animal_onnx_v05.md` | 本文檔 |

## 提交與清理

```powershell
$git = 'C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe'
& $git branch --show-current
& $git status --short
& $git add src/utils/animal_onnx_validation.py scripts/validate_animal_onnx.py tests/test_animal_onnx_validation.py tests/test_animal_onnx_validate_cli.py docs/animal_onnx_v05.md
& $git diff --cached --check
& $git commit -m 'test(animal): add ONNX CPU parity and performance validation'
& $git push origin perf/low-vram-animal-inference
```

不要把模型權重、生成的 `.onnx` 文件及大型驗證輸出提交到 Git，除非明確需要。
