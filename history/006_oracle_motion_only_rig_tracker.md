# 006：Oracle Motion-only 双鱼眼 Rig Tracker 开发记录

- 日期：2026-09-04
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- 最终状态：通过
- 自动报告：`debug/oracle_motion_tracker/report.json`
- 可视化目录：`debug/oracle_motion_tracker/`

## 1. 本阶段目的

在 `005` 已验证双鱼眼 rig 重投影和解析 Jacobian 的基础上，建立完全独立的 motion-only LM reference：

```text
固定 Oracle pixel correspondence
+ source GT inverse-range
+ 固定 source rig pose
+ 错误 target rig pose 初值
-> Front/Back 重投影与 target-pose Jacobian
-> 联合 6DoF LM
-> target rig pose
```

本阶段用于验证 `005` 的 target-pose Jacobian 不仅与有限差分/autograd 一致，而且能够正确驱动真实数据上的非线性位姿优化。

## 2. 文件修改列表

- `hislam2/tracking/__init__.py`：导出独立 tracking reference 接口。
- `hislam2/tracking/oracle_motion_only.py`：Oracle problem builder、固定 correspondence 类型、motion-only LM optimizer 和结果记录。
- `tests/test_oracle_motion_tracker.py`：增加数据边界、采样、正负单轴、固定 cost、balanced weighting 和 CUDA 测试。
- `scripts/validate_oracle_motion_tracker.py`：增加三个真实帧对、三种相机模式、精度对照、50 帧递推和可视化验证。
- `history/006_oracle_motion_only_rig_tracker.md`：本记录。

没有修改：

```text
DepthVideo
MotionFilter
TrackFrontend
FactorGraph
projective_ops.py
CUDA BA
Gaussian Mapping
```

## 3. Oracle 信息边界

“Oracle”严格定义为：

1. GT range 和 source/target GT pose 只在 problem builder 内使用；
2. 它们只用于生成、遮挡检查和筛选固定的 subpixel correspondence；
3. `OracleMotionProblem` 不保存 target GT pose；
4. optimizer 只接收固定 problem 和调用端提供的 target pose 初值；
5. target GT pose 仅由验证脚本持有，用于外部误差统计。

单帧扰动测试固定 source GT pose，以隔离验证 target-pose Jacobian。

50 帧递推只在 frame 0 使用 GT pose。每个 `i -> i+1` 使用上一帧估计：

```text
fixed source pose = G_i_est
target initial    = G_i_est
```

中间帧从不使用 source GT pose 重锚定。

## 4. 公共接口

```python
problem, gt_target_pose = build_oracle_motion_problem(
    source_frame,
    target_frame,
    reprojector,
    range_provider,
    fixed_source_pose=...,  # 单帧默认 source GT；递推传 G_i_est
)

tracker = OracleMotionOnlyTracker(reprojector, config)
result = tracker.optimize(problem, initial_target_pose)
```

`OracleMotionProblem` 保存：

```text
T_rig_from_world_source
front: Front -> Front fixed correspondence
back:  Back  -> Back  fixed correspondence
```

每组 correspondence 保存：

```text
source_pixels              [N,2]
source_inverse_range       [N]
observed_target_pixels     [N,2]
source/target camera index
fixed_validity             [N]
base_weights               [N]
```

`OracleMotionOnlyResult` 保存 pose、状态、初始/最终 cost、有效点数、固定 camera scale、Hessian eigenvalues/condition number、最终 damping 和逐 retry 历史。

返回状态：

```text
converged
max_iterations
insufficient_observations
numerical_failure
```

## 5. Oracle correspondence 构造

默认直接在 2880×2880 原图上以 stride=16 采样，网格起点为 `stride//2`。相机参数没有缩放。

Builder 使用 `GroundTruthRangeProvider` 获取：

```text
range_m
inverse_range
observation_valid
confidence
```

target GT range 在 Oracle target subpixel 上使用双线性采样。`align_corners=True` 对应的归一化严格为：

```text
x_n = 2*u/(W-1) - 1
y_n = 2*v/(H-1) - 1
```

四个双线性邻点必须全部 observation-valid。整数像素 identity 单测已通过，没有半像素偏移。

遮挡过滤为：

```text
abs(predicted_target_range - sampled_target_gt_range)
    <= max(0.01 m, 0.01 * sampled_target_gt_range)
```

target pixel 还必须通过 128 px safety margin：中心点加八方向偏移后仍同时位于 DS 数学有效域和图像域。

三个正式帧对的固定 correspondence 数：

| frame pair | Front | Back | 合计 |
|---|---:|---:|---:|
| 0→1 | 22,177 | 22,516 | 44,693 |
| 50→51 | 21,456 | 21,702 | 43,158 |
| 150→151 | 23,531 | 21,597 | 45,128 |

Builder 完成后，correspondence、validity 和 base weight 全部固定并 detach。优化迭代中不会重新生成或改变 Oracle observation。

## 6. Pose、twist 和残差约定

位姿状态：

```text
G = T_rig_from_world
```

左扰动和 twist 顺序：

```text
G_new = Exp(delta) @ G
delta = [tx, ty, tz, rx, ry, rz]
```

残差：

```text
r = observed_target_pixel - predicted_target_pixel
```

`005` 返回预测像素 Jacobian `J_t=d(predicted_pixel)/d(delta_t)`，因此本阶段直接求：

```text
J_t * delta ~= r
H = sum(J_t^T W J_t)
b = sum(J_t^T W r)
```

真实正负六自由度测试证明该符号与左扰动方向一致。

## 7. Huber 与 Front/Back 联合

二维 pixel residual 共用一个 Huber 权重：

```text
e_i = sqrt(r_u^2 + r_v^2)
w_huber = min(1, 5/e_i)
```

默认 `all` 模式直接联合所有 correspondence。

`balanced` 模式在优化开始前只根据固定 base weight 计算一次：

```text
S_F = sum(w_F_base)
S_B = sum(w_B_base)
s_F = (S_F + S_B) / (2*S_F)
s_B = (S_F + S_B) / (2*S_B)
```

整个优化期间 `s_F/s_B` 不变：

```text
w_final = s_camera * w_base * w_huber
```

因此 Huber 权重可以随 residual 变化，但优化目标的 Front/Back 基础比例不会逐轮改变。

## 8. 固定分母 nonlinear cost

所有 current/candidate cost 都在 builder 的同一固定集合上比较：

```text
E(G) = sum(C_i(G)) / N_fixed
```

合法点：

```text
C_i = s_camera * w_base * huber(e_i)
```

非法点：

```text
C_i = s_camera * w_base * C_invalid
```

`C_invalid` 为 100 px residual 对应的 Huber cost。禁止用 `N_valid` 作分母，因此 candidate 无法通过丢失 correspondence 人为降低平均 cost。

candidate invalid 点不进入 Hessian；超过固定集合的 1% 变为几何无效时直接拒绝。

## 9. LM 配置和接受规则

默认参数：

```text
Huber threshold                 5 px
initial lambda                  1e-3
accepted lambda scale           0.3
rejected lambda scale           10
maximum retries                 6
maximum iterations              20
maximum translation step        0.1 m
maximum rotation step           5 deg
minimum total observations      2000
minimum per enabled camera      500
maximum candidate invalid       1%
translation convergence         1e-6 m
rotation convergence            1e-6 rad
relative cost convergence       1e-8
```

LM 求解：

```text
(H + lambda*D) delta = b
D = diag(max(diag(H), 1e-12))
```

每个 candidate 重新执行完整 reprojection 并计算真实 fixed-set Huber cost。只有 cost 不增且 invalid 比例不超过门限才接受。

未加阻尼的对称 Hessian 每轮通过 `eigvalsh` 记录 eigenvalues 和 condition number。condition number 仅用于诊断，不改变目标。

## 10. 单帧数值验证

正式矩阵包括三个 frame pair、Front-only/Back-only/Front+Back 三种模式，以及：

```text
GT initial
tx/ty/tz 各 ±5 cm
rx/ry/rz 各 ±2 deg
固定方向 5 cm + 2 deg
固定方向 10 cm + 5 deg
```

共 9 个 GT initial、117 个标准扰动和 9 个困难扰动，全部状态为 `converged`。

GT initial：

```text
最大 initial/final cost   1.5662e-29
最大 final translation    2.7195e-16 m
最大 final rotation       0 deg
最大迭代数                 1
```

全部 Front/Back/Both 标准扰动：

```text
最大 final translation              1.2176e-10 m
最大 final rotation                 1.2074e-06 deg
有初始平移时的最小误差下降比例       0.9999999976
有初始旋转时的最小误差下降比例       0.9999993963
最大迭代数                           3
最大 Hessian condition number        11.8527
```

正式 Front+Back 标准扰动：

```text
39/39 converged
最大 final translation              7.7414e-12 m
最大 final rotation                 1.2074e-06 deg
最大 final robust cost              8.6565e-18
最大迭代数                           3
```

困难扰动：

```text
Front-only   3/3 成功
Back-only    3/3 成功
Front+Back   3/3 成功
最大 final translation   1.9978e-11 m
最大 final rotation      1.2074e-06 deg
最大迭代数                4
```

所有 accepted nonlinear cost 均单调不增；Hessian、delta、pose 和 cost 无 NaN/Inf。

## 11. CPU/CUDA float32 对照

使用同一固定 problem 和同一 5 cm/2° mixed initial pose：

```text
CPU float32 vs float64 translation   6.5214e-08 m
CPU float32 vs float64 rotation      1.3570e-07 deg

CUDA float32 vs float64 translation  6.2889e-08 m
CUDA float32 vs float64 rotation     3.1340e-07 deg

CUDA vs CPU float32 translation      1.1987e-07 m
CUDA vs CPU float32 rotation         1.8676e-07 deg
```

设备：`NVIDIA GeForce RTX 5060 Ti`。CPU/CUDA float32 均为 `converged`，远低于 1 mm / 0.01° 对照门限。本阶段仍未实现 CUDA optimizer kernel；CUDA 结果来自同一 PyTorch reference 在 GPU 上运行。

## 12. 50 帧无重锚定递推

序列使用 frame 0–49：

```text
G_0_est = G_0_GT
fixed source pose for i->i+1 = G_i_est
target initial for i->i+1    = G_i_est
```

49 个 pair 全部 `converged`，每个 pair 都使用上一帧估计，不读取 source GT pose 重锚定。

世界轨迹位置没有错误使用 `G[:3,3]`，而是：

```text
T_world_from_rig = inverse(T_rig_from_world)
p_rig_world = T_world_from_rig[:3,3]
```

frame 0 已用 GT 锚定且 range 为米制，因此评价不做 SE(3) 或 Sim(3) alignment。

结果：

```text
ATE RMSE                              5.2029e-07 m
absolute translation max             8.9708e-07 m
absolute rotation max                4.1841e-07 deg
translation RPE mean / max            8.9061e-08 / 1.9903e-07 m
rotation RPE mean / max               3.8731e-08 / 9.7313e-08 deg
iterations mean / max                 3 / 3
valid correspondence mean / range     43,612.7 / up to 44,809
Hessian condition mean / max          6.6119 / 8.3412
```

极小旋转误差使用稳定的 `atan2(sin(theta), cos(theta))` 计算，避免 `acos(trace)` 在接近单位阵时放大舍入误差。

## 13. 可视化检查

每个正式 frame pair 的 Front-only、Back-only 和 Front+Back mixed-standard 场景输出：

```text
initial_overlay.png
optimized_overlay.png
initial_warp.png
optimized_warp.png
target_rgb.png
convergence.png
```

overlay 中绿色点为固定 Oracle target pixel，红色箭头从当前 predicted pixel 指向 Oracle observation。人工检查结果：

- initial overlay 可见与 5 cm/2° 扰动一致的系统性 residual vector；
- optimized overlay 中预测点与 Oracle 点重合；
- Front/Back 同时收敛，没有一颗相机方向相反；
- convergence 曲线显示真实 robust cost 单调下降。

50 帧额外输出：

```text
debug/oracle_motion_tracker/sequence/trajectory.png
debug/oracle_motion_tracker/sequence/trajectory_error.png
debug/oracle_motion_tracker/sequence/sequence_diagnostics.png
debug/oracle_motion_tracker/sequence/trajectory.csv
```

GT/estimated trajectory 在三维和 XY/XZ/YZ 投影中重合。为了避免亚微米误差下后绘制曲线完全遮住另一条曲线，图中 Estimated 使用下层橙色实线，GT 使用上层蓝色虚线，并标出绿色起点和红色终点。CSV 逐帧保存 pose error、RPE、cost、迭代数、有效点数和 Hessian condition number。

## 14. 验证命令

```bash
conda activate hislam2
cd ~/open-project/3DGS/HI-SLAM2

python tests/test_oracle_motion_tracker.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/validate_oracle_motion_tracker.py
```

不生成 frame-pair 可视化：

```bash
python scripts/validate_oracle_motion_tracker.py --skip-visualization
```

高密度补充验证：

```bash
python scripts/validate_oracle_motion_tracker.py --stride 8
```

本次结果：新增测试 11 项通过，全量回归 39 项通过，正式 Oracle Tracker 报告为 `passed`。同时回归：

```text
Fisheye rig reprojection validation: passed
GT RangeProvider validation: passed
```

## 15. 当前边界与下一阶段

- 当前 correspondence 是由 GT range 和 GT pose 预生成的固定 Oracle observation，不是图像匹配结果。
- 当前 source inverse-range 固定，不优化深度状态。
- 当前每次只优化一个 target rig pose，不是多帧 BA。
- 当前实现是 CPU/PyTorch correctness reference；CUDA 只验证相同 float32 实现，没有独立 optimizer kernel。
- warp 是稀疏 correspondence 可视化，不是最终可微 image warp。
- 本阶段证明 `005` target-pose Jacobian 的符号、坐标系、双目联合和非线性更新均正确。

下一阶段再将 Oracle correspondence 替换为真实 DROID、optical-flow 或 feature pixel correspondence，然后逐步接入正式 Tracking、DepthVideo 和 FactorGraph。
