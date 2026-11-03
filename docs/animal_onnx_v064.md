# LivePortrait V0.6.4 — 多参考帧 + 嘴部几何 / 纹理分层（实验版）

基线：`perf/low-vram-animal-inference` 的 **V0.6.3**（GitHub commit `815fa9d1393ba68c7839375fcacca2785b26f4ad`）。本阶段**仅新增文件**，不改 V0.6.1/0.6.2/0.6.3 原有实现、模型、参数或测试。

> **目标**：使用同一只狗的闭嘴/张嘴参考，加上真实的中间视频帧，减轻两端照片直接 RGB 混合时牙齿/舌头重影；把 *外部口鼻轮廓* 和 *内部口腔纹理* 分层处理，并用**未参与输入的帧**做留出验证。
>
> **限制**：仅 CPU 2D 视觉实验，没有训练、ONNX 推理、自动狗脸定位、三维下颌、音素、口型驱动、视频合成。质量需要人工验收。

## 一、两种新策略

1. `layered_nearest`：相邻真实参考帧通过 V0.6.3 风格的 Delaunay 分片仿射网格对齐。外侧口鼻区域进行几何变形和颜色插值；嘴腔内部采用**最近参考帧的一份清晰纹理**，而不是双重牙齿的全区域 RGB 混合。优点是口腔细节清楚；缺点是跨纹理选择阈值仍可能**突然跳变**。
2. `layered_flow`：相同的外部几何变形，但嘴腔内部先进行局部 **Farneback 双向光流**再插值。可能减轻位置错位；在张嘴时新露出的牙齿、舌头和遮挡边界处，光流本身不可靠，仍可能重影或拉伸。

对比图保留 V0.6.3 的 `soft`、`dominant` 两行基线；新增两行分别显示上述两种策略。输出 `mouth_outer_mask.png` 和各目标开口状态对应的 `mouth_interior_mask_NNN.png`，用于检查两层实际覆盖区域。

**中间参考帧原始图片**位于原 `prepare_dog_open_closed.py` 生成的 `build/onnx_animal_v062/yawn_pair/frame_01.png`、`frame_02.png`。要使用与 V0.6.2 `original` 模式一致的图像参考，而不是把原图中间帧与 `onnx` 重建的两端图像直接混合；脚本会检查 `report.json` 并拒绝后一种不公平输入组合。

## 二、环境与准备

推荐既有 `liveportrait-onnx` Conda 环境；要求 `numpy`、`opencv-python-headless`（或 `opencv-python`）、`Pillow`、`scipy`。**不会调用 ONNX Runtime 或 F/M/W/G**，也不会下载资源。

在仓库根目录，先检查中间帧、V0.6.2 两端照片和 V0.6.3 已验证标注：

```powershell
conda activate liveportrait-onnx
$pair = "build/onnx_animal_v062/yawn_pair"
$ref  = "build/onnx_animal_v062/yawn_preview"
$lm   = "build/onnx_animal_v063/yawn_annotation/mouth_landmarks_verified.json"
Test-Path "$pair/frame_01.png"
Test-Path "$pair/frame_02.png"
Test-Path "$ref/closed_reference.png"
Test-Path "$ref/open_reference_aligned.png"
Test-Path $lm
```

如果已有的 `mouth_landmarks_verified.json` 不是针对当前 `yawn_preview` 两张图片生成的，SHA256 校验会拒绝使用。请先在 V0.6.3 中重新完成标注。

## 三、最快运行：无新增人工标注（代理点实验）

`alpha=0,0.333333,0.666667,1` 代表原 GIF 4 帧的**时间进度**，不是精确测量的嘴巴张开幅度；请勿等同于说话音素。

```powershell
python scripts/run_animal_mouth_multiref.py `
  --reference-dir $ref `
  --landmarks $lm `
  --same-dog `
  --mid "0.333333=$pair/frame_01.png" `
  --mid "0.666667=$pair/frame_02.png" `
  --method both `
  --alphas "0,0.25,0.5,0.75,1" `
  --output-dir build/onnx_animal_v064/yawn_multiref
```

运行过程中会用 **256px 上半脸 ECC 平移配准** 把中间帧对齐到闭嘴参考，再生成 512px 渲染。上下唇等中间关键点暂按 V0.6.3 闭嘴/张嘴端点的线性插值估计；`report.json` 中明确写入 `endpoint_linear_proxy_UNVERIFIED`。它不代表真实下颌轨迹。

```powershell
mspaint.exe "build/onnx_animal_v064/yawn_multiref/comparison_sheet.png"
```

生成目录还包含：`mid_01_aligned.png`、`mid_02_aligned.png`、`mouth_layered_nearest_*.png`、`mouth_layered_flow_*.png`、`mouth_outer_mask.png`、`mouth_interior_mask_*.png`、`report.json`。同名文件存在时不会覆盖，除非添加 `--overwrite`。

## 四、精确实验：为中间帧人工标注五个对应点

如希望进一步提高网格变形准确度，推荐对中间帧也人工标注。浏览器标注工具是**完全离线的单个 HTML 文件**，不会上传照片，也不会自动把“预测值”当成真实点击。

```powershell
python scripts/annotate_animal_mouth_multiref.py `
  --reference-dir $ref `
  --landmarks $lm `
  --same-dog `
  --mid "0.333333=$pair/frame_01.png" `
  --mid "0.666667=$pair/frame_02.png" `
  --output-dir build/onnx_animal_v064/yawn_annotation
```

打开页面：

```powershell
$html = (Resolve-Path "build/onnx_animal_v064/yawn_annotation/annotate_mid_mouth.html").Path
Start-Process chrome.exe -ArgumentList ([System.Uri]$html).AbsoluteUri
```

在每一帧上分别手动点击 5 个点：`chin` 下巴外轮廓、`left_corner` 图像左嘴角、`lower_lip` 下唇、`right_corner` 图像右嘴角、`upper_lip` 上唇。**不要把舌尖当成下唇，不要为不可见的关键点编造确切位置**。所有帧 × 所有点均点击完成后，网页才允许下载 `mouth_mid_landmarks_verified.json`。

把下载文件移到 `build/onnx_animal_v064/yawn_annotation/`，然后重新运行上节的渲染命令，另外添加：

```powershell
  --mid-points "build/onnx_animal_v064/yawn_annotation/mouth_mid_landmarks_verified.json" `
  --output-dir "build/onnx_animal_v064/yawn_manual"
```

示例这里是命令参数续行，需放进上一节命令中，**不要单独执行**。图片源与标注 JSON 的 SHA256 会一一核对。

## 五、独立留出帧验证（避免数据泄漏）

**不能**一边将 Frame 1 用作 `--mid` 参考、一边用 Frame 1 当 ground truth 给自己打分。建议做两个互换试验：

**实验 A：Frame 2 作为参考，Frame 1 留出**

```powershell
python scripts/run_animal_mouth_multiref.py `
  --reference-dir $ref --landmarks $lm --same-dog `
  --mid "0.666667=$pair/frame_02.png" `
  --holdout "0.333333=$pair/frame_01.png" `
  --method both `
  --output-dir build/onnx_animal_v064/holdout_frame01
```

**实验 B：Frame 1 作为参考，Frame 2 留出**

```powershell
python scripts/run_animal_mouth_multiref.py `
  --reference-dir $ref --landmarks $lm --same-dog `
  --mid "0.333333=$pair/frame_01.png" `
  --holdout "0.666667=$pair/frame_02.png" `
  --method both `
  --output-dir build/onnx_animal_v064/holdout_frame02
```

报告 `holdout_evaluation.roi_mae` 比较 **V0.6.2 RGB、V0.6.3 soft、V0.6.3 dominant、V0.6.4 nearest/flow** 对**同一张未参与拟合的真实帧**的嘴部 ROI RGB 平均绝对像素误差。数字越低代表像素误差较小，但不等于更自然；需结合 `holdout_aligned.png` 和对比图检查纹理、轮廓、头部配准。

留出实验故意不使用为两帧一同创建的 `--mid-points` 文件；该 JSON 必须与实际 `--mid` 集合完全匹配，避免错误套用标注。必要时为不同实验各导出独立的中间帧标注。

## 六、验收准则

- `python -m unittest discover -s tests -p 'test_animal_*py' -v` 全量测试通过（需在完整的本地仓库和 Python 环境中验证）。
- `outside_roi_changed_all_zero == true`；每种算法非口部区域与 Closed 参考 **逐像素相等**。
- 至少 1 张真实额外参考帧参与模型，不能只做 V0.6.3 旧版复跑。
- 观察 25%→50%→75%：牙齿/舌头重影、纹理跳变、嘴角和下颌连续性。`layered_nearest` 不保证时间平滑，`layered_flow` 不保证处理遮挡正确。
- Frame 1/2 用作输入锚点时，不得把“拟合原图准确”当成独立效果；留出测试分别执行。
- 完全闭嘴到张嘴、音频同步、真正下颌几何、时序闪烁抑制均不在本阶段验收范围内。

## 七、回退 / 边界

本阶段只新增：

- `src/utils/animal_mouth_multiref.py`
- `src/utils/animal_mouth_multi_annotator.py`
- `scripts/run_animal_mouth_multiref.py`
- `scripts/annotate_animal_mouth_multiref.py`
- `tests/test_animal_mouth_multiref.py`
- `tests/test_animal_mouth_multiref_cli.py`
- `docs/animal_onnx_v064.md`

不修改其他源码或默认分支。失败后可使用 `git apply -R` 回退 Git Patch（前提是这些新增文件未进一步编辑）；或者用 GitHub Desktop 查看变更并撤销。所有 `build/onnx_animal_v064/...` 的导出由用户自行管理。
