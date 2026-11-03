# LivePortrait V0.7.0 — CPU 动物表情跟随视频（ONNX F/M/W/G）

**基线**：`AaronharveyHan/LivePortrait`，工作分支 `perf/low-vram-animal-inference`，已包含 V0.6.6，核对 HEAD `fde08300503b8fdbee78e2afb1990b96efbca1ed`。

## 目标与边界

重新回到 LivePortrait 原本的模型驱动路线：**一张狗狗源照片 + 真人驱动视频 → ONNX 动作迁移 → 流式 MP4**。不再用 RGB 透明度混合、舌头多边形、嘴部手工几何插值作为主链路。

- 复用 V0.6 已数值对齐的固定形状 Animal v1.1 模型 F / M / W / G；CPUExecutionProvider，FP32、batch=1。
- 首次仅运行 `F(source)`、`M(source)`、`M(driving_first)`、3D ConvexHull 比例。后续每帧执行 `M(frame)`、`W`、`G`、编码，不缓存视频帧列表。
- 输出 512×512、H.264 yuv420p **无音频** MP4；可选 `--comparison` 三栏视频（驱动 / 源图 / 输出）。
- 允许 `--motion-mode full`（默认）、`expression-only`（冻结驱动 R/t，仅使用表情 exp 的相对变化）、`pose-only`（冻结 exp，仅使用 R/t 的相对变化）。这些是模型隐式空间的诊断模式，**不是精确语义的嘴部/眨眼通道识别**。
- 没有自动人脸检测、视频逐帧跟踪、S stitching、贴回原背景或音频同步；建议事先提供面部在画面中心且姿态变化不大的驱动视频。
- 无法保证 GT 1030 2GB 实时执行；本阶段仅验证 CPU 离线生成和缓存、吞吐、资源变化。**真实的 W/G 性能和动物表情效果必须在用户 Windows 环境验证**。

## 应用 Git Patch

放入项目根目录，运行：

```powershell
cd C:\Users\ES\Documents\GitHub\LivePortrait
conda activate liveportrait-onnx
$git = "C:\Users\ES\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
& $git branch --show-current
& $git status --short
& $git apply --check .\liveportrait_v0.7.0_expression_video.patch
if ($LASTEXITCODE -ne 0) { throw "Patch check failed" }
& $git apply .\liveportrait_v0.7.0_expression_video.patch
if ($LASTEXITCODE -ne 0) { throw "Patch apply failed" }
```

本补丁只**新增** `src/utils/animal_onnx_video.py`、`scripts/run_animal_onnx_video.py`、两个测试、本文档。不覆盖现有脚本/模型权重/默认分支。

## 依赖验证

```powershell
python -c "import numpy, cv2, scipy, imageio, PIL, psutil, onnxruntime; print('dependencies OK')"
python scripts/run_animal_onnx_video.py --plan
python -m unittest discover -s tests -p 'test_animal_*py' -v
```

`--plan` 是不加载模型也不解码视频的元数据检查。如果四个 ONNX 的 `onnx_ready` 没有全部为 `true`，先恢复 `build/onnx_animal_v04` 文件。首次生成视频需要 `imageio-ffmpeg`（常随 imageio[ffmpeg] 安装）。

## 首轮运行建议：仅 10 帧

下面使用你已有的狗狗头部裁剪 `build/dog01_pawpose_round1/input_crops/dog_closed.png`。驱动视频选择官方或你自行拍摄的**真人面部居中**短 MP4；以下只是示例路径，文件必须实际存在。

```powershell
$source = "build/dog01_pawpose_round1/input_crops/dog_closed.png"
$driver = "assets/examples/driving/d19.mp4"

python scripts/run_animal_onnx_video.py `
    --source $source `
    --driving $driver `
    --motion-mode full `
    --threads 4 `
    --max-frames 10 `
    --driving-multiplier 1.0 `
    --comparison build/onnx_animal_v070/dog_compare.mp4 `
    --output build/onnx_animal_v070/dog_driven.mp4
```

如果模型的人物头部动作幅度过大，再尝试 **同样输入** 的表情优先实验：

```powershell
python scripts/run_animal_onnx_video.py `
    --source $source `
    --driving $driver `
    --motion-mode expression-only `
    --max-frames 10 `
    --threads 4 `
    --output build/onnx_animal_v070/dog_expression_only.mp4
```

两个实验的 `driving`、`source` 和帧数应一致，方便判断 `full` 与 `expression-only` 的差异。**不能假定冻结 R/t 就一定只动嘴和眼睛**，因为 M 的 `exp` 仍是隐式表达。

首轮成功后改用 `--max-frames 60` 或更大值。传 `--max-frames 0` 会执行完整视频，因此 CPU 下可能需要很长时间。

## 输入裁剪（非常重要）

默认 `--source-crop center-crop`、`--driving-crop center-crop`：先从照片/视频帧正中心截成正方形，再缩放到 256×256。这只是**空间中心裁剪**，不检测鼻尖、眼睛或跟踪人脸。若源图本身是有效的 256×256 头部裁剪，则不会额外损失视野。

- **不要直接用全身 1080×1920 的狗狗照片指望自动对齐狗脸**。请使用现成头部裁剪图，或预先保证狗脸位于正方形图像中。
- 真人视频建议先在其他视频软件中居中裁好头部；如人脸偏画面一角，中心裁剪可能失败。
- `--source-crop stretch` 和 `--driving-crop stretch` 仅用于旧版 V0.6 的拉伸对照，会改变长宽比，不推荐正式质量评估。

## 性能与验证指标

输出目录下的同名 JSON 报告包含：

- `frame_count` / `fps` / `fps_source`；`fps` 继承驱动视频元数据，`--fps` 只是重标输出帧率，不进行时间重采样。
- `ort_session_load_ms`、`source_and_anchor_cache_init_ms`、`video_generation_ms`、`effective_generation_fps`。
- `model_metrics.F_calls_expected=1`、`M_calls_expected=frame_count+1`、`W_calls_expected=frame_count`、`G_calls_expected=frame_count`。这些是缓存模式的预期次数，已由模拟 Session 测试核对。
- `model_metrics.per_frame_mean_ms` 的 `M_target`、`W`、`G` 等项。第一帧 `M_target=0` 因为直接复用首帧 M。
- `rss_sampled_peak_mib`：周期采样的进程 RSS 峰值，**不是操作系统意义上的真实峰值**；如果没有 psutil，该值为 null。

**验收**：先检查 10 帧 MP4 能正常播放、狗脸身份保持、人物张口/闭口和轻微摇头有对应变化、是否异常扭曲；再跑较长视频检验内存稳定性和实际生成速度。当前还不支持口型与音频音素级同步。

## 风险与回退

- 没有声称通用的人→动物表情迁移一定自然；部分源图片上的口腔细节可能无法由 G 可靠生成。
- `--overwrite` 只允许成功编码后替换最终输出文件；遇到解码/编码/模型异常，临时文件会被清理，不把损坏的 MP4 当成完成品。多文件结果不能保证跨文件原子替换。
- 模型和真人视频均需本地授权与合规使用。
- 若需撤回，在未产生依赖更改时执行 `git apply --reverse --check` 然后 `git apply --reverse` 对本补丁回退；用户已经生成的 `build/` 视频不会自动删除。
