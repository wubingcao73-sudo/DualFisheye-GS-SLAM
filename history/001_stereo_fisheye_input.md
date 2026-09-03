# 001：双目鱼眼原始数据输入改造记录

- 日期：2026-09-02
- 阶段：S0，双目鱼眼数据契约与输入验证
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 状态：已完成并通过真实数据测试

## 修改目标

将 `demo.py` 原来的单目图片目录和针孔内参输入替换为同步双目 Double Sphere 鱼眼数据输入。本阶段只建立输入契约和验证入口，不把鱼眼图像送入尚未适配 DS 几何的跟踪、BA 或 Gaussian mapping。

所有 RGB、range 和标定均保持 2880×2880 原始分辨率；没有缩放、裁剪、去畸变或修改原始图像。

## 修改位置

### `demo.py`

- 删除旧的 `--imagedir`、`--calib`、`--gtdepthdir` 单目输入流程。
- 增加 `--data-root`，并提供 classroom 数据集默认路径。
- 增加 CPU 独立的 `--validate-input` 和可选的 `--json` 报告。
- 默认配置改为 `config/classroom_stereo_fisheye.yaml`。
- DS 跟踪尚未实现时禁止普通 SLAM 运行，避免把原始鱼眼数据错误地当成针孔数据。

### `hislam2/data/frame_types.py`

新增不可变数据类型 `StereoFisheyeFrame`，字段约定如下：

```text
index                  int
frame_number           int
timestamp              float, seconds
rgb                    uint8   [2,3,H,W], RGB order
camera_model           "double_sphere"
camera_params          float32 [2,6], xi alpha fx fy cx cy
image_size             int64   [2,2], width height
T_rig_from_camera      float64 [2,4,4]
gt_T_world_from_rig    float64 [4,4]
gt_range               float32 [2,H,W], meters
gt_range_valid         bool    [2,H,W]
```

### `hislam2/data/stereo_fisheye_dataset.py`

- 新增 `StereoFisheyeDataset` 和默认数据路径。
- 严格配对左右 RGB、EXR range、DepthVis、时间戳和三套 pose 的帧 ID。
- 读取 `calibration/cam0.txt`、`cam1.txt` 的 8 个 DS 标定值。
- 参数文件只支持英文逗号或空白分隔；中文逗号会明确报错。
- 读取并检查 `T_rig_from_cam0`、`T_rig_from_cam1` 和 `T_cam1_from_cam0`。
- 检查 `T_world_from_rig @ T_rig_from_camera` 与相机 GT pose 的一致性。
- 使用 OpenEXR 直接读取 Blender EXR 的单通道 `V`，不使用 `depth_png` 代替真值。
- 使用 DS 逆投影定义域、有限正值条件以及 Blender 的 `1e10` 无效值共同生成 range mask。
- 通过 `validate_dataset()` 对 manifest 和首、中、末帧做真实解码检查。

### 其他工程文件

- `hislam2/data/__init__.py`：导出数据集、帧类型和验证函数。
- `config/classroom_stereo_fisheye.yaml`：记录 DS 模型、文件布局、参数顺序、变换方向、range 语义和单位。
- `tests/test_stereo_fisheye_dataset.py`：增加标定、分隔符、DS 投影域和外参解析测试。
- `requirement.txt`、`requirements-cu128.txt`：增加 OpenEXR 依赖；当前 `hislam2` 环境已安装 `OpenEXR 3.4.0`。

### 数据目录中的参数修正

以下两个文件最后一个分隔符由中文逗号改成英文逗号，数值没有变化：

- `classroom/calibration/cam0.txt`
- `classroom/calibration/cam1.txt`

两份文件均统一为：

```text
# xi alpha fx fy cx cy width height
xi, alpha, fx, fy, cx, cy, width, height
```

## 数据检查结果

- 左右 RGB：各 300 帧。
- 左右 EXR range：各 300 帧。
- 左右 DepthVis：各 300 帧，只用于一致性检查。
- rig、cam0、cam1 pose：各 300 帧。
- 时间范围：0.00 至 11.96 秒；相邻帧间隔 0.04 秒。
- 左右原始尺寸：均为 2880×2880。
- 双目基线：0.050000012 m。
- pose/extrinsics 组合最大误差：`1.227e-07`，小于 `1e-6`。
- EXR 中发现鱼眼视域外使用 `1e10` 作为无效值；加入 mask 后，抽样帧有效率约为 75.4% 至 77.0%。
- 首帧有效 range：cam0 约 0.761–6.943 m，cam1 约 0.764–6.685 m。
- EXR 与 DepthVis 抽样相关系数绝对值均大于 0.83，通过一致性检查。

## 测试命令与结果

### 自动化测试

```bash
conda run -n hislam2 python -m unittest discover -s tests -p 'test_*.py' -v
```

结果：5 项测试全部通过。

### 语法检查

```bash
conda run -n hislam2 python -m py_compile \
  demo.py \
  hislam2/data/__init__.py \
  hislam2/data/frame_types.py \
  hislam2/data/stereo_fisheye_dataset.py \
  tests/test_stereo_fisheye_dataset.py
```

结果：通过。

### 默认路径真实数据验证

```bash
conda run -n hislam2 python demo.py --validate-input
```

### 显式路径真实数据验证

```bash
conda run -n hislam2 python demo.py \
  --validate-input \
  --data-root /media/nonchalance/data/Google_Downloads/sim/sim_data/classroom
```

结果：两种路径都能够读取并验证完整 manifest，以及第 0、150、299 帧的双目 RGB 和 EXR range。

## 当前边界和下一步

- 当前代码已经能够可靠读取原始双目鱼眼数据，但还不能运行完整 SLAM。
- 下一阶段应实现统一 DS `project/unproject/valid_mask`，并为网络输入单独决定显存可承受的分辨率。
- 若后续缩放图像，只能等比例同步缩放 `fx/fy/cx/cy`；`xi/alpha` 保持不变，且该预处理不能回写原始数据。
