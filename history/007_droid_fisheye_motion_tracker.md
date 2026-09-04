# 007：DROID Correspondence + 双鱼眼 Motion-only Rig Tracking 开发记录

- 日期：2026-09-04
- 默认数据：`/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom`
- DROID 权重：`pretrained_models/droid.pth`
- 最终状态：通过
- 自动报告：`debug/droid_fisheye_motion/report.json`
- 可视化目录：`debug/droid_fisheye_motion/`

## 1. 本阶段目的与结论

在 `006` 已验证 motion-only 双鱼眼 LM 后，把 Oracle target pixel correspondence 替换成冻结 DROID 网络产生的 target pixel 和置信度：

```text
当前 DS pose projection
+ DROID local correlation/update
-> 固定 target pixel / weight
-> 006 motion-only fisheye LM
-> 更新 target rig pose
-> 下一轮 DROID/update + LM
```

DROID 在这里不是 pose 回归器。只使用 `fnet/cnet/update` 产生 correspondence；pose 仍由 Double Sphere 重投影、解析 target-pose Jacobian 和 LM 解出。

三个真实 frame pair 的 Front-only、Back-only、Front+Back 全部满足 `<5 mm / <0.05 deg`。50 帧递推的 Front+Back 结果为：

```text
ATE RMSE                    0.01469062 m
translation RPE RMSE       0.00053305 m
rotation RPE RMSE          0.01586635 deg
失败状态                    0 / 49
```

与 frame-0 固定不动的 zero-motion baseline 相比，ATE 从 `1.64784171 m` 降至 `0.01469062 m`。

## 2. 文件修改列表

- `hislam2/tracking/droid_fisheye_motion.py`：独立 DROID correspondence frontend、feature DS camera、source geometry 与 DROID/LM 交替求解。
- `hislam2/tracking/__init__.py`：导出 007 公共接口。
- `tests/test_droid_fisheye_motion.py`：半像素映射、ray identity、信息边界、相机模式、最小闭环和真实三帧对验收。
- `scripts/validate_droid_fisheye_motion.py`：三帧对、三相机模式、correspondence、50 帧递推、数值报告和可视化。
- `history/007_droid_fisheye_motion_tracker.md`：本记录。

保持未修改：

```text
DepthVideo
MotionFilter
TrackFrontend
FactorGraph
projective_ops.py
CUDA BA kernel
Gaussian Mapping
```

## 3. 信息边界

Tracker 的 target 输入是 `DroidFrameFeatures`，该类型只保存 `frame_index/feature_maps/context_hidden/context_input`，不保存 target frame 或 target GT pose。

每个 `i -> i+1`：

```text
fixed source pose = G_i_est
target initial    = G_i_est
source range      = source GT Euclidean range
target GT pose    = tracker 不可见，仅外部评价使用
target GT range   = 完全不使用
```

50 帧中只有 frame 0 用 GT 锚定。中间帧不使用 GT source/target pose 重锚定。`DroidSourceGeometry` 只由 source frame 构建，保存 feature pixel、source inverse-range 和 source validity。

当前阶段仍使用 GT source range，因此结果验证的是“真实 DROID correspondence + 已知 source metric range + motion-only pose”，不是完整无 GT SLAM。

## 4. 网络与相关性实现

冻结加载 `droid.pth`，去除 checkpoint 的 `module.` 前缀，并按现有 HI-SLAM2 规则把 update 的 delta/weight 输出裁到 2 通道。网络设为 `eval()` 和 `requires_grad_(False)`。

输入顺序固定为：

```text
source Front, target Front, source Back, target Back
```

只建立同相机时间边：

```text
Front: 0 -> 1
Back:  2 -> 3
```

不做 Front↔Back 同帧匹配。原生图像用 bilinear、`align_corners=False`、antialias resize 到 `720x720`，encoder feature 为 `90x90`。

使用 `AltCorrBlock` 的局部 CUDA correlation，不使用 `CorrBlock` 的全对全相关体。后者在每相机 `90x90` feature 上会产生约 `8100^2` 个相关元素，并不适合作为该独立 reference 的默认实现。

## 5. resize、feature camera 与半像素约定

原图为 `2880x2880`，resize 比例 `a=0.25`，encoder stride `s=8`。feature Double Sphere 参数为：

```text
fx_F = a*fx/s
fy_F = a*fy/s
cx_F = (a*(cx+0.5)-0.5)/s
cy_F = (a*(cy+0.5)-0.5)/s
```

`xi/alpha` 不变。feature pixel 到原图 range pixel：

```text
u_N = (s*u_F + 0.5)/a - 0.5
v_N = (s*v_F + 0.5)/a - 0.5
```

到 720 图像可视化坐标：

```text
p_720 = s*p_F + 0.5
```

单测确认 feature camera ray 与映射后的 native camera ray 在 float64 精度下一致。GT source range 用 `align_corners=True` 的已有 reference 双线性采样，并要求四邻点均 observation-valid；整数像素采样 identity test 已通过。

## 6. DROID/update 与 LM 交替

每个 frame pair 默认执行四个 outer iteration，hidden state、context、correlation pyramid 和前一轮 target 持续保留：

1. 用当前 target pose 做 DS reprojection，得到 `coords1`；
2. `AltCorrBlock(coords1)` 查询局部相关性；
3. motion 输入为 `[coords1-coords0, previous_target-coords1]`；
4. DROID update 输出 `delta` 和二维 `weight`；
5. 固定当前 outer iteration 的 `target=coords1+delta`；
6. scalar base weight 为 `sqrt(weight_u*weight_v)`；
7. 筛除 source range/model 无效、target model/image 无效、非有限、weight 不足和不满足 4 feature-px safety margin 的点；
8. 构建与 006 兼容的固定 `OracleMotionProblem` 容器；容器名为兼容保留，此时 observation 已不是 Oracle；
9. 调用 006 LM 优化一个 target rig pose；
10. 用新 pose 进入下一轮 DROID/update。

同一次 LM 内 correspondence、base weight、balanced camera scale 和 `N_fixed` 都保持固定。DROID 可以在下一个 outer iteration 生成新 correspondence，但不会破坏单次 LM candidate cost 的可比性。

## 7. Pose、残差与求解配置

沿用 006：

```text
G = T_rig_from_world
G_new = Exp(delta) @ G
delta = [tx, ty, tz, rx, ry, rz]
residual = observed target pixel - predicted target pixel
```

007 默认参数：

```text
DROID outer iterations              4
camera weighting                    balanced
Huber threshold                     1 feature px
invalid residual penalty            20 feature px
target safety margin                4 feature px
minimum network weight              1e-4
minimum total observations          2000
minimum per enabled camera          500
maximum candidate invalid           1%
LM maximum iterations               20
LM maximum retries                  6
translation step limit              0.1 m
rotation step limit                 5 deg
```

Balanced camera scale 仍只由每次 problem 的固定 base weight 计算一次。candidate cost 仍使用固定 `N_fixed` 分母和 invalid penalty；每个 accepted LM step 的真实 robust cost 单调不增。

## 8. 三个真实 frame pair 的 correspondence 结果

数值单位是 `90x90` feature pixel。这里使用第四轮 DROID/LM 后固定集合，与 GT pose 产生的 target pixel 只在验证脚本外部比较：

| pair | camera | median | p90 | `<1 px` inlier |
|---|---|---:|---:|---:|
| 0→1 | Front | 0.04749 | 0.17935 | 99.82% |
| 0→1 | Back | 0.06797 | 0.29153 | 98.74% |
| 50→51 | Front | 0.05556 | 0.19944 | 99.29% |
| 50→51 | Back | 0.07218 | 0.32202 | 97.90% |
| 150→151 | Front | 0.04701 | 0.19431 | 99.28% |
| 150→151 | Back | 0.06399 | 0.29311 | 98.81% |

全部满足：median `<0.5 px`、p90 `<1 px`、`<1 px` inlier `>90%`。

## 9. 三个真实 frame pair 的 pose 结果

所有测试都以 source GT pose 固定 source，并以同一 source pose 作为 zero-motion target 初值；GT target pose 只用于结果统计。

| pair | mode | translation | rotation |
|---|---|---:|---:|
| 0→1 | Front | 0.6468 mm | 0.009962° |
| 0→1 | Back | 0.6116 mm | 0.032777° |
| 0→1 | Both | 0.5441 mm | 0.014356° |
| 50→51 | Front | 0.7249 mm | 0.012741° |
| 50→51 | Back | 0.4803 mm | 0.016259° |
| 50→51 | Both | 0.6756 mm | 0.015602° |
| 150→151 | Front | 0.2350 mm | 0.025193° |
| 150→151 | Back | 0.2896 mm | 0.021108° |
| 150→151 | Both | 0.3000 mm | 0.022900° |

9/9 状态均为 `converged`，全部满足 `<5 mm / <0.05 deg`，无 NaN/Inf。各 outer iteration 固定 correspondence 数在 `4,758–11,154` 之间；LM 最多 5 次迭代。

单帧对 Hessian 诊断：

```text
minimum eigenvalue across solves       24,352.47
maximum eigenvalue across solves       5,029,303.0
condition number range                 4.7741 – 22.9969
```

## 10. 50 帧递推结果

轨迹位置严格使用：

```text
T_world_from_rig = inverse(T_rig_from_world)
p_rig_world = T_world_from_rig[:3,3]
```

frame 0 使用 GT 锚定；frame 1–49 都以 `G_i_est` 同时作为 fixed source pose 和下一帧 target 初值。不做 SE(3)/Sim(3) trajectory alignment。

| mode | ATE RMSE | translation RPE RMSE | rotation RPE RMSE | bad status |
|---|---:|---:|---:|---:|
| Front | 0.01700537 m | 0.00074851 m | 0.01770540° | 0/49 |
| Back | 0.01482539 m | 0.00070136 m | 0.02227679° | 0/49 |
| Both | 0.01469062 m | 0.00053305 m | 0.01586635° | 0/49 |
| Zero-motion baseline | 1.64784171 m | 0.05939630 m | 0.37161639° | — |

Front+Back 的最大 absolute translation error 为 `0.02493320 m`。最终 frame 49 的误差为 `0.02474557 m / 0.657274 deg`。这说明逐帧相对运动很准，但纯递推仍有累积 drift；它没有达到 006 Oracle 轨迹的近零误差，符合从真值 correspondence 换成学习 correspondence 后的预期。

序列中最少固定 correspondence 为 `4,718`，最大 Hessian condition number 为 `33.5991`。147 个相机模式/pair 组合全部 `converged`，没有 NaN/Inf。

正式序列验收在实现前固定为：Both 无坏状态；ATE 优于 zero-motion；translation/rotation RPE 均相对 zero-motion 至少降低 50%。四项全部通过。

## 11. 可视化与数值产物

单帧对：

```text
debug/droid_fisheye_motion/pair_000000_000001/
  droid_correspondence_overlay.png
  droid_epe_heatmap.png
  weight_heatmap.png
  initial_pose_overlay.png
  optimized_pose_overlay.png
  convergence.png
```

序列：

```text
debug/droid_fisheye_motion/sequence/
  trajectory.png
  trajectory_error.png
  sequence_diagnostics.png
  trajectory.csv
```

`trajectory.png` 是单个三维世界坐标轨迹图，三个坐标轴使用相同比例；GT 蓝色虚线最后绘制，并在图例标明 `GT (drawn on top)`，因此即使估计轨迹与 GT 高度重合也可以看见真值。绿色/红色圆点分别标记起点/终点。已有 CSV 可用以下命令直接重绘，无需重新执行网络和 LM：

```bash
conda run -n hislam2 python scripts/validate_droid_fisheye_motion.py \
  --regenerate-trajectory-only
```

后续将序列扩展到 300 帧后，已从 `trajectory.csv` 直接生成三维轨迹。该次报告的 Both 指标为 ATE `0.05937728 m`、translation RPE `0.00053342 m`、rotation RPE `0.01461120 deg`；`187→188` 出现一次 `numerical_failure`，因此 300 帧报告状态为 `failed`。轨迹和数值仍完整保留，后续需单独分析该 candidate/retry 失败点。

完整配置、每次 outer/LM、cost、weight、Hessian、pair metrics、sequence metrics 和验收布尔值都写入 `debug/droid_fisheye_motion/report.json`。

## 12. 测试与验收命令

```bash
conda run -n hislam2 python -m unittest discover -s tests -v
conda run -n hislam2 python scripts/validate_droid_fisheye_motion.py --sequence-length 50
```

最终完整测试为 `46/46 passed`。其中 007 包含 6 个纯 CPU 单测和 1 个真实 CUDA 验收测试；真实测试覆盖三个 frame pair、三种相机模式和相同正式 pose 门限。正式验证报告状态为 `passed`，全部 19 项 acceptance check 为 true。

## 13. 遗留边界与下一阶段

本阶段已完成“Oracle correspondence → DROID correspondence”的替换，但还有这些明确边界：

1. source Euclidean range 仍为 GT；下一步需要替换为在线深度/逆深度估计。
2. 只有相邻帧递推，没有 keyframe graph、回环或全局 BA，因此 50 帧的旋转和平移会积累 drift。
3. DROID 权重直接作为 base weight 使用，尚未重新标定为 fisheye/当前数据集上的统计不确定度。
4. 网络仍是原 pinhole 数据训练的权重，但输入没有展开、裁剪或改成 pinhole；当前结果证明其局部特征与 update 在原始 fisheye resize 上可工作，不代表已经完成 fisheye finetune。
5. CUDA 只用于现有 DROID correlation/network 和 PyTorch float32 reference；没有新增 CUDA optimizer kernel。

下一阶段应先把在线 depth/inverse-range 接入这个独立 tracker，验证完全不依赖每帧 GT range 的 50 帧结果，再考虑接入 `DepthVideo/MotionFilter/TrackFrontend/FactorGraph` 正式链路。
