# 005：双鱼眼 Rig 重投影与 Pose/Inverse-Range Jacobian 开发记录

- 日期：2026-09-04
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 最终状态：通过
- 自动报告：`debug/fisheye_reprojection/report.json`
- 可视化目录：`debug/fisheye_reprojection/`

## 1. 本阶段目的

建立与在线 tracking 和 BA 解耦的双鱼眼 rig 重投影 reference，实现：

```text
source pixel
-> Double Sphere unit ray
-> inverse-range camera point
-> fixed source camera-to-rig extrinsic
-> source/target rig poses
-> fixed target rig-to-camera extrinsic
-> target camera point
-> Double Sphere target pixel
```

同时实现预测 target pixel 对 source rig pose、target rig pose 和 source inverse-range 的解析 Jacobian。

本阶段没有修改旧 `projective_ops.py`、`DepthVideo`、MotionFilter、FactorGraph、BA、CUDA BA kernel 或 Gaussian mapping。

## 2. 文件修改列表

- `hislam2/geom/fisheye_reprojection.py`：新增 rig 重投影 reference、有效性结构和解析 Jacobian。
- `tests/test_fisheye_reprojection.py`：增加 identity、四种 camera pair、twist 顺序、有限差分、无效性和 CUDA 测试。
- `scripts/validate_fisheye_reprojection.py`：增加 float64/autograd/float32/CUDA/真实数据及可视化验证器。
- `history/005_fisheye_rig_reprojection_jacobians.md`：本记录。

## 3. Pose 与 twist 约定

位姿状态固定为：

```text
G = T_rig_from_world
```

这与当前 DROID/lietorch 的 world-to-camera 状态方向一致，但相机被提升为一帧共享的 rig。

更新使用 lietorch 左扰动：

```text
G_new = Exp(delta) @ G
```

twist 顺序严格沿用 lietorch：

```text
delta = [tx,ty,tz,rx,ry,rz]
        [ translation | rotation ]
```

没有在模块边界进行列置换。验证器逐个扰动六个 basis vector，并与 lietorch `SE3.exp()` 对照：

```text
point-action Jacobian 最大绝对误差：8.2266637946e-11
前三列对单位平移块误差：0
```

## 4. 公共接口

```python
reprojector = FisheyeRigReprojector(
    cameras,
    T_rig_from_camera,
)

result = reprojector.reproject(
    source_pixels,
    source_inverse_range,
    T_rig_from_world_source,
    T_rig_from_world_target,
    source_camera_index,
    target_camera_index,
    compute_jacobians=True,
)
```

输入形状：

```text
source_pixels          [...,2]
source_inverse_range   [...]
rig poses              [4,4]
T_rig_from_camera      [camera_count,4,4]
```

输出：

```text
pixels                         [...,2]
target_range                   [...]
source_pose Jacobian           [...,2,6]
target_pose Jacobian           [...,2,6]
inverse_range Jacobian         [...,2,1]
```

Jacobians 定义为：

```text
d predicted_target_pixel / d variable
```

它们不是 `target-prediction` 或 `prediction-target` 残差的 Jacobian；后续构造残差时必须在调用端处理符号。

固定外参在构造函数中复制并 detach，本阶段不优化外参。

## 5. 完整变换链

定义：

```text
E_s = T_source_rig_from_source_camera
E_t = T_target_rig_from_target_camera
G_s = T_source_rig_from_world
G_t = T_target_rig_from_world
rho = source inverse-range
r_s = source unit ray
```

则：

```text
P_Cs = r_s / rho
P_Rs = E_s * P_Cs
P_W  = inverse(G_s) * P_Rs
P_Rt = G_t * P_W
P_Ct = inverse(E_t) * P_Rt
pixel_t = camera_t.project(P_Ct)
target_range = ||P_Ct||
```

同一接口支持：

```text
cam0 -> cam0 temporal
cam1 -> cam1 temporal
cam0 -> cam1 stereo
cam1 -> cam0 stereo
```

## 6. 解析 Jacobian

lietorch 左扰动下，点作用 Jacobian 为：

```text
J_action(P) = [I, -skew(P)]
```

目标位姿：

```text
dP_Ct/d(delta_t) =
    R_Ct_Rt [I, -skew(P_Rt)]
```

源位姿包含 `inverse(G_s)`：

```text
dP_Ct/d(delta_s) =
    R_Ct_Rs [-I, +skew(P_Rs)]
```

inverse-range：

```text
dP_Cs/d(rho) = -r_s / rho^2

dP_Ct/d(rho) =
    R_Ct_Cs (-r_s/rho^2)
```

最终统一左乘 Camera Geometry V1 的解析 DS Jacobian：

```text
d pixel_t / d variable =
    d pixel_t / d P_Ct
    @ d P_Ct / d variable
```

## 7. Valid 职责分离

`ReprojectionValidity` 分别保存：

```text
source_model_valid
source_image_valid
range_valid
target_model_valid
target_image_valid
geometric_valid
```

其中：

```text
geometric_valid =
    source_model_valid
    & source_image_valid
    & range_valid
    & target_model_valid
    & target_image_valid
```

`observation_valid` 不进入几何模块，仍由 RangeProvider 提供。真实数据调用端使用：

```text
final_valid = geometric_valid & observation_valid
```

合法的 source/target `ray_z<0` 不会被排除。没有使用 `Z>0`、最小 Z、角度 clamp 或 denominator clamp。无效 pixel、target-range 和 Jacobian 使用零占位，必须由 mask 排除。

## 8. Identity 测试

使用同一 rig pose 和同一 source/target camera：

```text
front pixel p99：1.9699340841e-12 px
front target-range max：2.1316282073e-14 m

back pixel p99：5.8006424669e-12 px
back target-range max：9.5923269328e-14 m
```

两颗相机均远低于 `1e-3 px` 门限。

## 9. Float64 有限差分结果

每种 pair 使用 8,019 个样本，包括 12 个显式 DS 数学域边缘像素。

| Pair | Source pose p99 | Target pose p99 | Inverse-range p99 |
|---|---:|---:|---:|
| temporal front | `3.6612e-10` | `3.5212e-10` | `1.3127e-08` |
| temporal back | `3.6945e-10` | `3.5776e-10` | `1.4399e-08` |
| front→back | `3.7241e-10` | `3.7259e-10` | `1.3930e-08` |
| back→front | `3.6609e-10` | `3.5977e-10` | `1.9547e-08` |

全部远低于 p99 `<1e-3` 门限。当前扰动步长下没有有效样本跨出数学域；报告仍保留 crossing count。

四组有效样本中 source/target 的负 `ray_z` 数量均超过 2,000，证明测试没有退化为针孔正前方区域。

## 10. Float64 autograd 结果

| Pair | Source pose p99 | Target pose p99 | Inverse-range p99 |
|---|---:|---:|---:|
| temporal front | `4.4617e-16` | `4.5254e-16` | `5.1188e-14` |
| temporal back | `4.2222e-16` | `3.6609e-16` | `5.4896e-14` |
| front→back | `3.8110e-16` | `3.9292e-16` | `1.1193e-13` |
| back→front | `4.5755e-16` | `4.1641e-16` | `2.5124e-13` |

autograd 使用可微 `torch.matrix_exp()` 构造同一 `[translation,rotation]` 左扰动。

## 11. CPU/CUDA float32 结果

CPU float32 对 float64：

```text
pose Jacobian p99：不超过 4.64e-07
inverse-range Jacobian p99：不超过 8.23e-05
valid mask mismatch：0
```

CUDA float32 对 CPU float32：

```text
pose Jacobian p99：不超过 4.17e-07
inverse-range Jacobian p99：不超过 8.28e-05
pixel p99：不超过 4.8828125e-04 px
valid mask：四组全部完全一致
```

测试设备：`NVIDIA GeForce RTX 5060 Ti`。

报告使用 p99 验收，同时保留 max。单个边缘样本的 pixel max 可能略高于 `1e-3 px`，没有删除样本或修改结果来隐藏它。

## 12. 真实 temporal 重投影

frame 0→1：

```text
front range relative p50/p90/p99：
0.0002619 / 0.0006565 / 0.0017514

back range relative p50/p90/p99：
0.0001519 / 0.0006231 / 0.0017759

front/back photometric median：4.0 / 4.0（8-bit）
```

两组 range p90 均远低于 `1%`。

## 13. 真实 stereo 重投影

同一 frame 0：

```text
front -> back range relative p50/p90/p95：
0.0047215 / 0.0094282 / 0.0112157

back -> front range relative p50/p90/p95：
0.0047210 / 0.0093778 / 0.0109426

两个方向可见像素：约 1.382 million
两个方向 photometric median：6.67（8-bit）
```

stereo p95 约 `1.1%`，低于 `10%` 工程门限。较大的 p99 主要来自遮挡边界、离散 forward splat 和 GT 表面切换，实际值保留在报告中。

## 14. 可视化检查

每组输出：

```text
source_rgb.png
target_rgb.png
warped_source.png
valid_mask.png
range_error.png
photometric_error.png
overlay.png
```

人工检查结果：

- front/back temporal overlay 与 target 整体对齐；
- stereo front↔back 有效区形成符合两颗背向 200° 鱼眼相机的环形重叠带；
- 环形重叠中的墙、窗、天花板和桌椅方向一致；
- 没有左右镜像、上下翻转、额外 180° 旋转或人工 offset；
- range/photometric 高误差主要集中在物体边缘、灯具、窗格和 splat 空洞附近。

## 15. 验证命令

```bash
conda activate hislam2
cd ~/open-project/3DGS/HI-SLAM2

python tests/test_fisheye_reprojection.py
python -m unittest discover -s tests -v
python scripts/validate_fisheye_reprojection.py
```

只运行数学、autograd、float32 和 CUDA 验证：

```bash
python scripts/validate_fisheye_reprojection.py --skip-visualization
```

成功输出：

```text
Fisheye rig reprojection validation: passed
Report: debug/fisheye_reprojection/report.json
```

## 16. 当前边界与下一阶段

- 当前 pose 输入只支持单个 `[4,4]` 变换；后续接入 tracking 时再增加 `[B,N,4,4]` vectorization。
- 当前实现是 PyTorch correctness reference，不是最终在线 CUDA kernel。
- 当前 forward splat 和 z-buffer 只用于验证，不是可微 warp。
- 当前固定 GT inverse-range 尚未作为 tracking 状态写入 `DepthVideo`。
- 下一阶段先将 `DepthVideo` 改为“双目观测 + 每时刻单 rig pose”，再用本 reference 对接新的原生鱼眼 `projective_ops`。
