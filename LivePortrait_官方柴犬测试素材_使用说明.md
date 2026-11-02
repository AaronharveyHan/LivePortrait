# LivePortrait V0.6 官方柴犬實拍測試素材準備包

此包不含重新發布的圖片或影片：直接使用 LivePortrait **官方倉庫已附帶**的素材，因此離線即可準備。

- 柴犬來源照片 `assets/examples/source/s25.jpg`
- 官方示例搭配的人臉驅動影片 `assets/examples/driving/d19.mp4`（藉由人臉表情驅動柴犬嘴部與頭部）
- 此二者是原始 `app_animals.py` 的官方動物動畫示例配對。

## 一、操作（Windows PowerShell）

把下載的 `prepare_liveportrait_official_dog_demo.py` 放在 Git 倉庫根目錄（與 `inference_animals.py` 同一層）。

```powershell
cd C:\Users\ES\Documents\GitHub\LivePortrait
conda activate liveportrait-onnx

Test-Path .\assets\examples\source\s25.jpg
Test-Path .\assets\examples\driving\d19.mp4

python .\prepare_liveportrait_official_dog_demo.py
```

成功後在 `build/onnx_animal_v06/official_dog_demo/` 自動得到：

- `dog_source.png`：256x256 來源柴犬
- `driver_first.png`：驅動影片第一影格
- `driver_target.png`：約 0.5 秒處的驅動影格（可調）
- `candidate_XXXX.png`：多個驅動影格候選
- `contact_sheet.jpg`：一頁預覽所有候選，方便選動作變化大但不變形的時刻
- `selection.json`：FPS、影格編號等中繼資料

## 二、真正執行 V0.6 四模型 ONNX 推理

```powershell
$dir = 'build/onnx_animal_v06/official_dog_demo'

python .\scripts\run_animal_onnx_frame.py `
    --source "$dir/dog_source.png" `
    --driving-first "$dir/driver_first.png" `
    --driving-target "$dir/driver_target.png" `
    --threads 4 `
    --output "$dir/result.png" `
    --report-json "$dir/result.json"

Invoke-Item "$dir/contact_sheet.jpg"
Invoke-Item "$dir/result.png"
```

這會用實拍柴犬照片和影片驅動幀進行 ONNX 推理，而非 Synthetic Random；但結果是否真實合理，必須用肉眼檢查。

如果想直接**對照有沒有真正動起來**，可再跑一次 `--driving-target "$dir/driver_first.png"`，把輸出設為 `baseline.png`。這次的首幀和目標幀相同，相對運動應為零；與前述 `result.png` 並排比較，才能更清楚看出頭部與嘴部的變化。包內 `run_liveportrait_official_dog_demo.ps1` 會自動跑這兩張輸出（如果你的 PowerShell 執行政策允許）。

## 三、換一個目標影格／嘴部動作

如預設影格動作不明顯：查看 `contact_sheet.jpg` 上的候選影格編號（或 `selection.json`），指定不同編號，例如：

```powershell
python .\prepare_liveportrait_official_dog_demo.py --target-frame 8 --overwrite
```

之後用相同的推理命令重跑；若 `result.png` 已存在，加上 `--overwrite`。注意 `--target-frame` 必須小於影片實際可解碼影格數。

## 四、重要限制

1. 此準備腳本使用**中心正方形裁剪**，不是 X-Pose 動物臉部檢測或精確關鍵點對齊。這比拉伸整張照片更適合測試，但不等於官方的自動動物裁剪。若生成臉部位置偏移，應手動調整裁剪。
2. 官方 `d19.mp4` 為人臉驅動影片，不是柴犬影片。這是官方提供的跨物種表情遷移測試組合；能檢驗嘴部／頭部傳遞，但不保證形變自然。
3. `driver_first` 和 `driver_target` 必須是同一個影片中不同時間的影格，否則相對運動可能為零。
4. V0.6 不包括 Stitching、Pasteback、音訊、錄影或自動裁剪；僅生成單張 512x512 PNG。
5. 產生的測試圖與報告位於 `build/`，不應被直接提交到 GitHub。
