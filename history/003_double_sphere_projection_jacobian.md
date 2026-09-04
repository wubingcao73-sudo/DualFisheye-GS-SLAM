# 003：Double Sphere Projection Jacobian（Camera Geometry V1）开发记录

- 日期：2026-09-03
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 最终状态：通过
- 自动报告：`debug/camera_model_v1/report.json`

## 1. 本阶段目的

为 Double Sphere 正向投影实现解析：

```text
d(u,v) / d(X,Y,Z)
```

它是后续重投影 tracking 和 BA 的几何基础。本阶段保持 CameraModel 独立，没有修改 `projective_ops.py`、`MotionFilter`、`DepthVideo`、FactorGraph、BA、CUDA BA kernel 或 Gaussian mapping。

## 2. 文件修改列表

- `hislam2/camera/base.py`：统一接口增加 `project_jacobian(points)`。
- `hislam2/camera/double_sphere.py`：实现 DS 解析投影 Jacobian，并让普通投影和 Jacobian 共用投影中间量与数学有效域。
- `tests/test_double_sphere_camera.py`：增加解析式、autograd、有限差分、无效点和 CUDA 一致性测试。
- `scripts/validate_camera_jacobian.py`：增加 cam0/cam1 独立 V1 验证器。
- `history/003_double_sphere_projection_jacobian.md`：本记录。

## 3. 公共接口

```python
pixels, model_valid, jacobian = camera.project_jacobian(points)
```

形状为：

```text
points       [...,3]
pixels       [...,2]
model_valid  [...]
jacobian     [...,2,3]
```

其中：

```text
jacobian[...,i,j] = d pixel_i / d point_j
pixel_0 = u
pixel_1 = v
point = (X,Y,Z)
```

无效点的 pixel 和 Jacobian 使用零占位，必须通过 `model_valid` 排除，不能消费占位值。

## 4. 解析公式

沿用 Camera Geometry V0：

```text
d1 = ||P||
z1 = xi*d1 + Z
d2 = sqrt(X^2 + Y^2 + z1^2)
D  = alpha*d2 + (1-alpha)*z1
u  = fx*X/D + cx
v  = fy*Y/D + cy
```

链式求导：

```text
dd1/dP = P/d1
dz1/dP = xi*dd1/dP + [0,0,1]

dd2/dP =
    ([X,Y,0] + z1*dz1/dP) / d2

dD/dP =
    alpha*dd2/dP + (1-alpha)*dz1/dP

du/dP = fx * ([1,0,0]/D - X*dD/dP/D^2)
dv/dP = fy * ([0,1,0]/D - Y*dD/dP/D^2)
```

`project()` 与 `project_jacobian()` 均调用 `_projection_terms()`，因此参数、投影公式和 `model_valid` 只有一个 source of truth。

## 5. 数学有效域和 epsilon 修正

继续使用 DS 原数学域：

```text
finite(P)
d1 > numerical tiny
Z > -w2*d1
D > 16*machine_epsilon*d1
```

允许合法的 `Z<0`，没有用 `Z>0`、角度 clamp、denominator clamp 或人为旋转来改变有效域。

V1 测试发现 V0 的 denominator epsilon 使用 `max(d1,1)`，会令同一条射线在 range 小于 1 时可能改变有效性。现改为与 `d1` 同比例的相对阈值，使投影有效性保持齐次。零点仍通过 `torch.finfo(dtype).tiny` 明确排除。

## 6. 测试方法

每颗相机使用真实标定，默认随机采样 20,000 个原图像像素，并显式加入：

- principal point 及其附近；
- 图像水平、垂直轴；
- 图像角点及数学无效角点；
- `ray_z≈0`；
- 合法的 `ray_z<0` 和 `Z<0`；
- DS 正向投影数学域边缘，margin 为 `1e-2`、`1e-4`、`1e-6`；
- 0.25 m 到 20 m 的不同 range。

解析 Jacobian 分别与以下 reference 比较：

1. CPU float64 PyTorch autograd；
2. CPU float64 中心有限差分，步长 `1e-6*max(1,||P||)`；
3. CPU float32 解析结果；
4. CUDA float32 解析结果。

若有限差分的正负扰动跨出数学有效域，该样本不进入有限差分误差统计，但在报告中保留 crossing count。

## 7. cam0 结果

覆盖：

```text
有效随机像素：19,784 / 20,000
总有效三维点：19,870
Z < 0：6,605
ray_z≈0：32
数学域边缘：32
数学无效图像角点：4 / 4
```

误差：

```text
float64 analytic vs autograd relative p99：4.0389574737e-16
float64 analytic vs finite difference relative p99：2.8125532431e-10
CPU float32 vs float64 Jacobian relative p99：3.2993227557e-07
CUDA float32 vs CPU float32 Jacobian relative p99：1.3641653155e-07
CUDA pixel max absolute error：4.8828125e-04 px
CUDA valid mask equal：True
```

## 8. cam1 结果

覆盖：

```text
有效随机像素：19,916 / 20,000
总有效三维点：20,003
Z < 0：6,747
ray_z≈0：32
数学域边缘：32
数学无效图像角点：4 / 4
```

误差：

```text
float64 analytic vs autograd relative p99：4.0076707825e-16
float64 analytic vs finite difference relative p99：2.8603857066e-10
CPU float32 vs float64 Jacobian relative p99：3.2632455478e-07
CUDA float32 vs CPU float32 Jacobian relative p99：1.5810577992e-07
CUDA pixel max absolute error：4.8828125e-04 px
CUDA valid mask equal：True
```

CUDA 测试设备：`NVIDIA GeForce RTX 5060 Ti`。

## 9. 验证命令

相机单元测试可从项目根目录直接执行：

```bash
conda activate hislam2
python tests/test_double_sphere_camera.py
```

完整 V1 验证：

```bash
python scripts/validate_camera_jacobian.py
```

自定义数据和采样量：

```bash
python scripts/validate_camera_jacobian.py \
    --data-root /path/to/classroom \
    --samples 20000 \
    --output debug/camera_model_v1
```

成功时终端输出：

```text
Camera Geometry V1 validation: passed
Report: debug/camera_model_v1/report.json
```

## 10. 当前边界和下一阶段

- 当前只实现 `d(u,v)/d(X,Y,Z)`，没有实现位姿、内参或 unprojection Jacobian。
- 当前解析 Jacobian 尚未接入 tracking 或 BA。
- 下一阶段建立独立 GT `RangeProvider`，输出欧氏 range 和 `observation_valid`。
- RangeProvider 验证后，再设计双目观测状态，最后才替换旧 pinhole `projective_ops` 和接入 tracking。
