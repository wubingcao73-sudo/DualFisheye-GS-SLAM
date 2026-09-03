# 002：Double Sphere Camera Geometry V0 开发记录

- 日期：2026-09-03
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 最终状态：通过
- 自动报告：`debug/camera_model/report.json`

## 1. 本阶段目的

建立与跟踪、BA 和 Gaussian 完全独立的 Double Sphere 相机几何模块，实现：

```text
pixel -> DS unproject -> unit ray
3D camera point -> DS project -> raw fisheye pixel
pixel + GT range -> camera-frame point
camera-frame point -> fixed extrinsic -> GT rig pose -> world point
```

本阶段没有修改 `projective_ops.py`、FactorGraph、前后端、BA、CUDA BA kernel、MotionFilter、GSBackEnd 或 Gaussian rasterizer，也没有实现 tracking、mapping、PGO、深度网络或跨相机匹配。

## 2. 文件修改列表

- `hislam2/camera/base.py`：统一 `CameraModel` 接口和惰性 LUT cache。
- `hislam2/camera/double_sphere.py`：标准 DS project、unproject 和数学有效域。
- `hislam2/camera/__init__.py`：公开相机接口。
- `hislam2/data/frame_types.py`：将深度有效性字段明确为 `range_observation_valid`。
- `hislam2/data/stereo_fisheye_dataset.py`：移除重复 DS 数学公式，增加 Blender 相机坐标到 DS 坐标的固定转换。
- `config/classroom_stereo_fisheye.yaml`：记录原始文件与运行时坐标约定。
- `scripts/validate_camera_geometry.py`：完整数学、LUT、点云和跨帧验证器。
- `tests/test_double_sphere_camera.py`：DS 单元测试。
- `tests/test_stereo_fisheye_dataset.py`：增加坐标转换和 pose 组合测试。

## 3. Double Sphere 参数定义

两目参数顺序统一为：

```text
xi alpha fx fy cx cy width height
```

运行时 `DoubleSphereCamera.parameters` 返回：

```text
xi alpha fx fy cx cy
```

cam0：

```text
xi=-0.1246412611
alpha=0.5654061503
fx=681.4980808766
fy=682.1175935080
cx=1445.7319706354
cy=1444.2557621085
width=2880
height=2880
```

cam1：

```text
xi=-0.15331803456262516
alpha=0.5574571673595061
fx=656.7849642114762
fy=655.7019232213135
cx=1440.9390155232030
cy=1443.3308917898670
width=2880
height=2880
```

## 4. Transform convention

全模块采用：

```text
T_destination_from_source
```

原始 Blender 文件保存：

```text
T_world_from_rig
T_world_from_blender_camera
T_rig_from_blender_camera
```

DS ray 使用视觉相机坐标，因此运行时转换为：

```text
C_blender_camera_from_ds_camera = diag(1, -1, -1, 1)

T_rig_from_ds_camera =
    T_rig_from_blender_camera
    @ C_blender_camera_from_ds_camera

T_world_from_ds_camera =
    T_world_from_blender_camera
    @ C_blender_camera_from_ds_camera
```

并验证：

```text
T_world_from_rig @ T_rig_from_ds_camera
== T_world_from_ds_camera
```

最大矩阵误差为 `1.2270505945e-07`。

## 5. Camera coordinate convention

DS/算法相机坐标：

```text
X：图像右方
Y：图像下方
Z：相机前方
```

Blender 相机局部坐标为 X 右、Y 上、相机看向 -Z，因此必须使用上述固定转换。转换后 cam0 的前方对应 rig +Y，cam1 的前方对应 rig -Y。

## 6. Image coordinate convention

```text
u：向右增加
v：向下增加
整数坐标：像素中心
图像范围：0 <= u < width, 0 <= v < height
```

Camera Geometry V0 只支持标定原生 2880×2880。请求其他 LUT 尺寸会报错，不隐式缩放内参。

## 7. Depth/range 真实语义

EXR 来自 Blender Cycles Z pass。生成逻辑和真实跨帧测试均确认它表示相机光心到可见表面的欧氏距离，单位为米：

```text
range = sqrt(X^2 + Y^2 + Z^2)
P_camera = range * unit_ray
```

真实 frame 0→1 的候选语义比较：

```text
range + Blender/DS坐标转换：range residual p50/p90/p99
约 0.026% / 0.067% / 0.249%

错误地当成 camera Z-depth：中位相对误差超过 100%
```

因此后续不得使用 `range == Z`。

## 8. Project 公式

对于 `P=(X,Y,Z)`：

```text
d1 = sqrt(X^2 + Y^2 + Z^2)
z1 = xi*d1 + Z
d2 = sqrt(X^2 + Y^2 + z1^2)
denom = alpha*d2 + (1-alpha)*z1
u = fx*X/denom + cx
v = fy*Y/denom + cy
```

投影域使用 DS 论文的 `w1/w2` 条件，不使用 `Z>0`。零范数、非有限输入和非正 denominator 均为无效。没有 clamp Z、角度、根号或 denominator。

## 9. Unproject 公式

```text
mx = (u-cx)/fx
my = (v-cy)/fy
r2 = mx^2+my^2

mz = (1-alpha^2*r2)
     / (alpha*sqrt(1-(2*alpha-1)*r2) + 1-alpha)

k = (mz*xi + sqrt(mz^2+(1-xi^2)*r2))
    / (mz^2+r2)

ray = [k*mx, k*my, k*mz-xi]
unit_ray = ray / ||ray||
```

所有根号域和 denominator 在计算前检查。无效项用零占位并返回 `model_valid=False`，占位值不能进入后续几何。

## 10. Model valid 定义

CameraModel 只负责：

- DS projection/unprojection 数学定义域；
- 有限输入；
- 非零点范数和合法 denominator；
- unit ray 的有限性。

`ray_z<0` 不是无效条件。

## 11. Image valid 与 observation valid

三类有效性完全分离：

```text
model_valid       CameraModel数学定义域
image_valid       像素是否位于图像宽高内
observation_valid 当前帧range是否是有限正值且不接近1e10 sentinel
```

消费 range 时：

```text
valid = model_valid & image_valid & observation_valid
```

Dataset 不再包含任何 DS 投影公式。

## 12. cam0 测试结果

```text
model valid pixels：8,199,996
model valid ratio：0.9886183449
pixel round-trip float64 p99：6.4310987108e-13 px
pixel round-trip float32 p99：3.4526697709e-04 px
3D reconstruction float64 p99：7.4732318960e-16
ray norm float64 max error：2.2204460493e-16
```

## 13. cam1 测试结果

```text
model valid pixels：8,253,165
model valid ratio：0.9950285735
pixel round-trip float64 p99：6.4310987108e-13 px
pixel round-trip float32 p99：3.5622002906e-04 px
3D reconstruction float64 p99：7.2692141469e-16
ray norm float64 max error：2.2204460493e-16
```

## 14. ray_z < 0 比例

```text
cam0：2,706,780 / 8,199,996 = 0.3300952830
cam1：2,756,473 / 8,253,165 = 0.3339898088
```

这只是当前标定的统计结果，不是 DS 模型常量。

## 15. LUT 内存占用

单相机 rays 加 bool valid：

```text
float64：207,360,000 bytes
float32：107,827,200 bytes
```

LUT 只在首次请求时创建；cache key 包含 resolution、device type/index 和 dtype。cam0/cam1 独立缓存，可用 `clear_ray_lut_cache()` 释放。

## 16. CPU float64 结果

两目所有规定阈值均通过：

- pixel round-trip p99 远小于 `1e-3 px`；
- point reconstruction p99 远小于 `1e-5`；
- ray norm 最大误差远小于 `1e-6`；
- 显式覆盖主点、中心、边缘、无效角点、`ray_z≈0`、`ray_z<0` 和大入射角。

## 17. CPU float32 结果

```text
cam0 pixel round-trip p99：0.000345267 px
cam0 max ray difference vs float64：8.046627e-06

cam1 pixel round-trip p99：0.000356220 px
cam1 max ray difference vs float64：2.622604e-06
```

## 18. CUDA float32 结果

GPU：NVIDIA GeForce RTX 5060 Ti。

```text
cam0 valid mask equal CPU：true
cam0 ray difference CPU p99/max：1.788139e-07 / 1.764297e-05
cam0 GPU pixel round-trip p99：0.000305176 px
cam0 ray norm max error：1.192093e-07

cam1 valid mask equal CPU：true
cam1 ray difference CPU p99/max：2.309680e-07 / 3.147125e-05
cam1 GPU pixel round-trip p99：0.000345267 px
cam1 ray norm max error：1.192093e-07
```

最坏单像素 float32 分量差高于最初临时设置的 `1e-5`，但全量 p99 约 `2e-7`，valid 完全一致且 GPU pixel round-trip 低于 `1e-3 px`。没有通过 clamp 或修改有效域隐藏该差异。

## 19. 单鱼眼点云结果

frame 0：

```text
front：6,257,351 points，93,860,446 bytes
back： 6,283,240 points，94,248,781 bytes
```

产物：

- `debug/camera_model/front_frame_0000.ply`
- `debug/camera_model/back_frame_0000.ply`

均为带原始 RGB 的 binary little-endian PLY。

## 20. 双鱼眼世界点云结果

```text
总点数：12,540,591
文件大小：188,109,047 bytes
```

产物：

- `debug/camera_model/dual_frame_0000_world.ply`
- `debug/camera_model/dual_frame_0000_world_preview.png`

预览中地面、墙体、顶面和室内结构方向合理；front/back 构成近全向场景，没有发现整体镜像、上下翻转或固定刚体错位。

## 21. GT 跨帧 reprojection 结果

目标 range 相对误差 p90：

```text
front 0000->0001：0.000656510
back  0000->0001：0.000678746
front 0150->0151：0.000681000
back  0150->0151：0.000709764
```

全部低于 `1%`。可见区 8-bit RGB photometric error median 为约 3.67–4.33 灰度级。所有原图、warp、valid mask 和 error heatmap 位于：

```text
debug/camera_model/reprojection/
```

人工检查 warp 与目标帧整体对齐；细网格状空洞来自 forward splat 离散化，不是 DS 坐标错位。

## 22. 当前仍存在的问题

- 本阶段没有解析 projection Jacobian。
- CameraModel 尚未接入 projective ops、tracking、BA 或 CUDA BA kernel。
- LUT 当前只允许原始标定分辨率，不做隐式缩放。
- forward reprojection 是验证工具，使用最近像素 z-buffer，不是后续训练所用的可微 warp。
- `debug/camera_model` 完整产物约 574 MB。

## 23. 下一阶段建议

先完成 Camera Geometry V1：实现解析 `duv/dXYZ`，与 float64 finite difference 和 autograd 对照。通过后实现 GT RangeProvider，统一输出 `range/inv_range/observation_valid/confidence`，再开始替换定位链路中的针孔几何。

## 测试命令

```bash
conda run -n hislam2 python -m unittest discover -s tests -p 'test_*.py' -v
conda run -n hislam2 python scripts/validate_camera_geometry.py
```

最终结果：13 项单元测试通过；完整 Camera Geometry V0 报告状态为 `passed`。
