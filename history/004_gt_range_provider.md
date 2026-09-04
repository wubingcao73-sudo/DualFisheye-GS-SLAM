# 004：GT Euclidean RangeProvider 开发记录

- 日期：2026-09-04
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 最终状态：通过
- 数值报告：`debug/range_provider/report.json`
- 可视化目录：`debug/range_provider/visualization/`

## 1. 本阶段目的

建立独立 GT `RangeProvider`，把原始 EXR 欧氏 range 统一转换为：

```text
range_m
inverse_range
observation_valid
confidence
```

本阶段只整理观测数据语义和有效性，不把 range 当成 camera Z-depth，不修改 `CameraModel`、`projective_ops.py`、tracking、`DepthVideo`、BA、CUDA BA 或 Gaussian mapping。

## 2. 文件修改列表

- `hislam2/range/base.py`：定义 `RangeProvider` 和 `RangeObservation`。
- `hislam2/range/ground_truth.py`：实现 `GroundTruthRangeProvider`。
- `hislam2/range/__init__.py`：公开 Provider API。
- `hislam2/data/frame_types.py`：移除 Dataset 生成的 `range_observation_valid` 字段。
- `hislam2/data/stereo_fisheye_dataset.py`：Dataset 只读取原始 GT range，数据验证改用 Provider。
- `scripts/validate_camera_geometry.py`：点云和跨帧重投影统一通过 Provider 获取有效观测。
- `scripts/validate_range_provider.py`：新增数值与可视化验证器。
- `tests/test_range_provider.py`：新增 synthetic、真实数据和 CUDA 测试。

## 3. 公共接口

```python
provider = GroundTruthRangeProvider()
observation = provider.provide(frame)
```

输出：

```python
@dataclass(frozen=True)
class RangeObservation:
    range_m: torch.Tensor
    inverse_range: torch.Tensor
    observation_valid: torch.Tensor
    confidence: torch.Tensor
```

当前双目真实帧形状：

```text
range_m             float32 [2,2880,2880]
inverse_range       float32 [2,2880,2880]
observation_valid   bool    [2,2880,2880]
confidence          float32 [2,2880,2880]
```

Provider 保持输入 tensor 的 dtype 和 device。`range_m` 引用原始 tensor，不复制原始 range；其他输出按请求惰性生成，不长期缓存。

## 4. Range 语义

当前 EXR 继续使用 Camera Geometry V0 已验证的定义：

```text
range_m = ||P_camera||
P_camera = range_m * unit_ray
inverse_range = 1 / range_m
```

它不是 camera Z-depth，也不再使用含义模糊的 `depth` 或 `disparity` 名称。

## 5. Observation valid

唯一实现位于 `GroundTruthRangeProvider`：

```text
isfinite(range_m)
& range_m > 0
& range_m < 1e10 * (1 - 1e-6)
```

无效位置：

```text
inverse_range = 0
confidence = 0
```

GT 有效位置：

```text
confidence = 1
```

原始 `range_m` 中的 sentinel、NaN 或 Inf 不被修改，以便诊断数据问题。消费者必须使用 `observation_valid`。

Provider 不计算 Double Sphere `model_valid`，也不计算 `image_valid`。真正消费观测时仍应组合：

```text
valid = model_valid & image_valid & observation_valid
```

## 6. 数值验证

Synthetic 测试显式覆盖：

```text
正常正 range
0
负值
NaN
Inf
1e10 sentinel
低于 sentinel 阈值的有限正值
```

真实 frame 0 结果：

### Front

```text
有效像素：6,257,351 / 8,294,400
有效比例：0.7544066906
sentinel 像素：2,037,049
非有限像素：0
非正像素：0
有效 range：0.7612628341 m ～ 6.9426460266 m
range * inverse_range 最大误差：5.9604644775e-08
```

### Back

```text
有效像素：6,283,240 / 8,294,400
有效比例：0.7575279474
sentinel 像素：2,011,160
非有限像素：0
非正像素：0
有效 range：0.7643648982 m ～ 6.6852102280 m
range * inverse_range 最大误差：5.9604644775e-08
```

两颗相机的无效 inverse-range 全为零，GT confidence 与 observation-valid 完全一致。

## 7. 可视化验证

对 front/back 分别输出：

```text
*_rgb.png
*_range.png
*_inverse_range.png
*_valid_mask.png
*_panel.png
```

range 和 inverse-range 使用各自有效像素 p01～p99 进行伪彩显示，只影响显示，不修改数值。无效像素显示为黑色。四联 panel 从左到右为：

```text
RGB | Euclidean range | inverse range | observation valid
```

人工检查结果：

- front/back range 均随墙面、地面、桌椅等表面连续变化；
- inverse-range 的远近关系相对 range 正确反转；
- valid mask 与镜头外区域以及无 GT 表面的窗格对应；
- RGB、range 和 mask 没有左右镜像、上下翻转或像素错位。

## 8. 内存行为

单个双目 2880×2880 float32 观测中，`range_m` 复用 Dataset 原 tensor。Provider 新生成：

```text
inverse_range：66,355,200 bytes
valid bool：   16,588,800 bytes
confidence：   66,355,200 bytes
合计：        149,299,200 bytes
```

Provider 不建立 cache，观测对象释放后内存可以回收。

## 9. 验证命令

从项目根目录执行：

```bash
conda activate hislam2
python tests/test_range_provider.py
python -m unittest discover -s tests
python scripts/validate_range_provider.py
```

只运行数值验证、不输出 PNG：

```bash
python scripts/validate_range_provider.py --skip-visualization
```

成功输出：

```text
GT RangeProvider validation: passed
Report: debug/range_provider/report.json
Visualization: debug/range_provider/visualization
```

本次结果：22 项单元测试全部通过，GT RangeProvider 报告为 `passed`。

迁移后还完整运行：

```bash
python scripts/validate_camera_geometry.py
```

Camera Geometry V0 的真实点云和跨帧重投影回归仍为 `passed`。

## 10. 下一阶段建议

下一阶段建立独立双鱼眼跨帧重投影模块，并在该模块实现 pose Jacobian：

```text
J_pixel_pose = J_DS_projection @ J_SE3_point_action
```

相机固定外参必须进入 source/target 变换链；分别验证 source rig pose、target rig pose 和 inverse-range Jacobian。此项通过后，再改造 `DepthVideo` 和旧 pinhole tracking 数据流。
