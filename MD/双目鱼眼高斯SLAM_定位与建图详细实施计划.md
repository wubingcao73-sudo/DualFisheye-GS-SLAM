# HI-SLAM2 到双目鱼眼 Gaussian SLAM：定位与建图详细实施计划

> 这是后续真正执行的主计划。它将适配工作明确分成“共享基础”、“定位”、“Gaussian 建图”、“深度网络”和“系统联调”五部分。
>
> 已知数据：同步左右原始鱼眼 RGB、左右内参、固定双目外参、真值 pose、真值深度图，左右 FoV 约 200°。
>
> 硬性原则：跟踪、BA、Gaussian 初始化、渲染和输出始终工作在原始鱼眼像素域，不生成中间投影图。
>
> 当前只规划，不修改算法代码。实施时一次只完成一个编号小步骤。

---

## 1. 为什么必须分成定位和建图

是的，必须分开。两者使用相同的鱼眼相机模型，但解决的问题不同：

### 1.1 定位部分解决什么

```text
输入：左右 RGB + 标定 + 固定外参 + range 观测
  -> 关键帧选择
  -> 左目时间匹配
  -> 同时刻左右匹配
  -> 鱼眼重投影因子
  -> 局部 BA
  -> 回环和全局 BA
输出：rig pose、关键帧、优化后 inv-range、pose/scale correction
```

定位必须能在不启动 Gaussian 建图的情况下独立测试。

### 1.2 建图部分解决什么

```text
输入：左右 RGB + rig pose + 固定外参 + range + CameraModel
  -> range 反投影点云
  -> Gaussian 初始化
  -> 原生鱼眼 Gaussian rasterizer
  -> RGB/range/normal loss
  -> densification / pruning
  -> 回环 pose correction 后的地图更新
  -> 离线 refinement 和表面输出
输出：Gaussian PLY、左右渲染、range、mesh 和优化后相机位姿
```

建图必须先在“真值 pose + 真值 range”下独立测试。如果这时都无法正确渲染，就不能将问题归因于定位误差。

### 1.3 定位与建图的唯一主边界

当前工程中，主边界是：

```text
hislam2/hi2.py:Hi2.call_gs()
```

现在它将 `poses/images/normals/1./disps_up/intrinsics` 传给 `GSBackEnd`。改造后必须传递一个有明确语义的 `MappingPacket`：

```python
MappingPacket = {
    "frame_id": ...,                 # [N]
    "timestamp": ...,                # [N]
    "rgb": ...,                      # [N,2,3,H,W]
    "T_rig_world": ...,              # [N,4,4]，world -> rig
    "T_camera_rig": ...,             # [2,4,4]
    "camera_model": ...,             # 左右模型名称+参数
    "range": ...,                    # [N,2,H,W]，m
    "range_valid": ...,              # [N,2,H,W]
    "range_confidence": ...,         # [N,2,H,W]
    "range_source": "gt",           # 后续变为 network
    "pose_correction": ...,          # 可选
    "scale_correction": ...,         # 只作诊断，不修改真值观测
}
```

`MappingPacket` 是两部分的合同。定位不应直接修改 Gaussian 内部变量，建图也不应自己复制一套跟踪 pose。

### 1.4 文中测试指标的统一定义

- `p99`：99% 的有效样本误差不超过该值；
- `scale_ratio`：估计轨迹相邻帧平移长度之和，除以 GT 轨迹对应平移长度之和；
- stereo metric 轨迹的 ATE 只允许 SE(3) 对齐，不允许 Sim(3) 对齐隐藏尺度误差；
- RGB/range 指标只在 `valid mask` 内统计，同时报告有效覆盖率；
- 本文数值是第一轮工程验收门限。如数据本身噪声使其无法达到，必须先用 GT/参考程序测得噪声下限，记录原因后才能调整，不能为了“通过”直接放宽。

---

## 2. 四种运行模式

| 模式 | pose | range | 启动定位 | 启动建图 | 用途 |
|---|---|---|---:|---:|---|
| `oracle_mapping` | GT pose | GT range | 否 | 是 | 单独验证建图 |
| `localization_gt_range` | SLAM pose | GT range | 是 | 可关闭 | 单独验证定位 |
| `full_gt_range` | SLAM pose | GT range | 是 | 是 | 真值深度的完整 pipeline |
| `full_pred_range` | SLAM pose | 网络 range | 是 | 是 | 最终系统 |

只有 `oracle_mapping` 可以在线路中使用 GT pose。其他模式中 GT pose 只用于计算误差。

---

## 3. 一眼看懂的总执行顺序

```text
S0 数据契约
  -> S1 鱼眼 CameraModel
  -> S2 GT DepthProvider
       │
       ├─> L0-L5 定位线（可独立运行）
       │       └─> 输出 rig pose + inv-range + correction
       │
       └─> M0-M7 建图线（先用 GT pose 独立运行）
               └─> 输出 Gaussian + render + mesh

L 线独立通过 + M 线独立通过
  -> I0-I3 合并为 full_gt_range
  -> D0-D3 双目鱼眼深度网络
  -> I4 full_pred_range
```

推荐真正执行次序：

1. `S0 -> S1 -> S2`；
2. `M0 -> M1 -> M2 -> M3 -> M4`，用 GT pose/range 先证明渲染与建图几何正确；
3. `L0 -> L1 -> L2 -> L3 -> L4 -> L5`，用 GT range 打通定位；
4. `M5 -> M6 -> M7`，完成在线双目 mapping；
5. `I0 -> I3`，完成真值深度 pipeline；
6. 执行 `D0 -> D3`，将 GT range 替换成网络 range；
7. 最后执行 `I4`，验收网络深度的完整 pipeline。

---

# A. 共享基础

## S0：双目鱼眼数据契约

### S0.1 阅读哪里

- `demo.py:mono_stream()`：当前只读一张图和 `[fx,fy,cx,cy]`；
- `demo.py` 主循环：queue 现在传递 `(t,image,intrinsics,is_last)`；
- `scripts/preprocess_owndata.py`：现有自有数据处理；
- `hislam2/hi2.py:track()`：系统入口。

### S0.2 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `demo.py` | 将 `mono_stream()` 替换为 stereo stream，同时返回左右 RGB、帧号、时间戳和 `is_last` |
| `[拟新增] hislam2/data/stereo_fisheye_dataset.py` | 读取左右 RGB、GT range、GT pose，做时间对齐和 invalid mask |
| `[拟新增] hislam2/data/frame_types.py` | 定义 `StereoFisheyeFrame`，避免使用过长 tuple |
| `[拟新增] config/<dataset>_stereo_fisheye.yaml` | 记录左右目录、camera model、标定、外参、pose/depth 格式和单位 |

统一帧形状：

```text
rgb                 [2,3,H,W]
camera_params       [2,P]
T_camera_rig        [2,4,4]
gt_T_rig_world      [4,4]
gt_range            [2,H,W]
gt_range_valid      [2,H,W]
```

如果只有左目深度，右目 `valid=False`，不伪造数据。

### S0.3 测试怎么做

| 测试 | 输入 | 检查 | 通过标准 | 结果文件 |
|---|---|---|---|---|
| 数量对齐 | 完整数据目录 | RGB/pose/depth 数量与 ID | 没有未解释的缺帧；左右成对率 100% | `debug/data_manifest.csv` |
| 时间对齐 | 随机 100 对 | 左右时间差 | 小于数据集规定的同步容差 | `debug/timestamp_report.txt` |
| 标定方向 | `T_right_left` | 基线长度和求逆 | `T_lr*T_rl` 与单位阵最大误差 `<1e-6` | `debug/calibration_report.txt` |
| pose 语义 | 前 20 帧 GT pose | 轴方向、四元数顺序、单位 | 相邻运动与数据实际运动一致 | `debug/pose_path.ply` |
| depth 语义 | 中心/边缘深度 | range 或 Z-depth | 官方定义已记录；可反投影成合理点云 | `debug/gt_range_cloud.ply` |

### S0.4 这一步的结果

- 程序能读出完整 `StereoFisheyeFrame`；
- 已明确 camera model、参数顺序、外参方向、pose 方向、深度定义和单位；
- 还没有进入跟踪和建图。

### S0.5 需要理解的语法

`dataclass`、Tensor shape、bool mask、YAML 字典、4×4 齐次变换、`torch.utils.data.Dataset`。

---

## S1：统一原生鱼眼 CameraModel

### S1.1 阅读哪里

- `hislam2/geom/pinhole.py`：当前 Python 针孔投影；
- `hislam2/geom/projective_ops.py`：当前帧间重投影；
- `hislam2/gaussian/utils/slam_utils.py:depths_to_points()`：当前深度反投影。

### S1.2 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `[拟新增] hislam2/geom/camera_models.py` | 定义统一接口 `project/unproject/project_jacobian/valid_mask` |
| `[拟新增] hislam2/geom/<actual_model>.py` | 根据标定文件实现真实鱼眼模型 |
| `[拟新增] hislam2/geom/ray_lut.py` | 生成左右 `ray_lut[H,W,3]` 和有效圆 mask |
| `hislam2/geom/projective_ops.py` | 后续通过 CameraModel 调用几何，不再直接调 `pinhole.py` |

200° FoV 下内部统一使用：

```text
unit_ray = unproject(u,v)
X_camera = range * unit_ray
inv_range = 1 / range
```

### S1.3 测试怎么做

| 测试 | 通过标准 |
|---|---|
| `uv -> ray -> uv` | 双精度下 p99 像素误差 `<1e-3 px` |
| `X -> uv+range -> X` | 有效区域 p99 相对误差 `<1e-5` |
| ray 单位长度 | `max(abs(norm(ray)-1)) < 1e-6` |
| Jacobian 有限差分 | 有效非奇异点相对误差 `<1e-3` |
| 边缘区域 | `ray_z < 0` 的后向边缘射线仍能正确往返 |
| 左右标定对照 | 与标定工具参考结果误差 `<0.1 px` |

结果保存到 `debug/camera_model/`：误差直方图、ray 可视化、边缘样本和 Jacobian 报告。

### S1.4 这一步的结果

定位和建图得到同一个经过单测的 CameraModel，以后不允许在其他文件里重新手写一套投影公式。

### S1.5 需要理解的语法

类与统一接口、broadcasting、`torch.where`、安全除法、有限差分、Jacobian。

---

## S2：DepthProvider，第一版只返回 GT range

### S2.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `[拟新增] hislam2/depth/depth_provider.py` | 定义 `DepthPacket` 和 provider 基类 |
| `[拟新增] hislam2/depth/gt_depth_provider.py` | 读 GT depth，在数据边界转成 m 制 range/inv-range/mask/confidence |
| `hislam2/motion_filter.py` | GT 模式不再调用现有 OmniData depth 推理 |
| `hislam2/depth_video.py` | 分开存储 `range_measurement`、`inv_range_state`、`range_valid`、`range_source` |

### S2.2 测试怎么做

- provider 输出与原始 GT 在 valid 区域对比，转换后最大误差 `<1e-6 m`；
- `range * inv_range` 在 valid 区域的最大误差 `<1e-5`；
- invalid 区域不参与求逆，输出不含 NaN/Inf；
- 1/8 分辨率下采样不将 invalid=0 当成真实远景。

结果是后续定位和建图不再关心 GT 文件格式，只消费 `DepthPacket`。

---

# B. 定位部分

## L0：读懂当前单目定位链路

### L0.1 阅读顺序和要回答的问题

| 顺序 | 文件 | 只回答什么 |
|---:|---|---|
| 1 | `hislam2/hi2.py:track()` | MotionFilter、Frontend、PGBA、GS 按什么顺序调用 |
| 2 | `hislam2/motion_filter.py:track()` | 什么情况添加候选关键帧 |
| 3 | `hislam2/track_frontend.py` | 如何建图边、做局部 BA、删关键帧 |
| 4 | `hislam2/factor_graph.py` | `ii/jj`、feature correlation、target/weight 是什么 |
| 5 | `hislam2/depth_video.py` | pose、disparity、feature、intrinsics 存在哪里 |
| 6 | `hislam2/geom/projective_ops.py` | Python 重投影的输入输出 |
| 7 | `src/droid_kernels.cu` | 真正在线 BA 使用的投影和 Jacobian 在哪里 |

### L0.2 这一步的结果

产出一张定位数据流图和一张 Tensor 表，暂不修代码。

---

## L1：将 DepthVideo 改成“双目观测 + 单 rig pose”

### L1.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/depth_video.py:__init__()` | `fmaps` 的 camera 维由 1 扩展为 2；保留一份 rig pose；增加左右 camera params/外参/range mask |
| `hislam2/depth_video.py:__item_setter()/append()/shift()` | 每次写入、移动、删除必须同步处理左右数据 |
| `hislam2/motion_filter.py` | 左右 RGB 共享 encoder，写入两组 feature/context；关键帧 ID 仍是 rig 帧 ID |
| `hislam2/modules/droid_net.py` | 检查 feature extractor 输入维度，不将左右相机当两个时刻 |

目标状态：

```text
poses             [N,7]              # 每个时刻一个 rig pose
fmaps             [N,2,128,h,w]
camera_params     [N,2,P] 或 [2,P]
T_camera_rig      [2,4,4]
inv_range_state   [N,h,w]            # 第一版以左目为主状态
gt_range          [N,2,H,W]          # CPU observation，如果数据提供
```

### L1.2 测试怎么做

- 对 3 个人工帧执行 `append -> shift -> __getitem__`；
- 所有帧 ID、左右 RGB hash、camera id、range mask 在操作后与期望一致；
- 左右 feature 形状一致但数值不应完全相同；
- 任意时刻只有一个可优化 rig pose；
- 运行 100 次 append/shift 不出现错位或越界。

结果：`DepthVideo` 成为正确的双目 rig 状态容器，但尚未加 stereo factor。

---

## L2：左目时间跟踪，使用 GT range

### L2.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/geom/projective_ops.py:projective_transform()` | 将 `iproj_pinhole/proj_pinhole` 替换为 CameraModel，深度语义改为 inv-range |
| `hislam2/depth_video.py:reproject()/distance()` | 传递 camera model 和 valid mask |
| `hislam2/factor_graph.py:update()` | 第一轮使用 GT inv-range，先开 `motion_only` 只优化 pose |
| `src/droid_kernels.cu` | 替换 `iproj/proj`、pose Jacobian、inv-range Jacobian、frame/covis distance |
| `src/droid.cpp` | 如 camera params/外参维度变化，同步修改 CUDA 入口 |

### L2.2 测试怎么做

1. Python identity reprojection：相同 pose 下 p99 `<1e-3 px`；
2. Python 已知位姿：人工三维点投影结果与直接 CameraModel 计算一致；
3. CUDA/Python 对照：像素坐标最大误差 `<1e-3 px`，valid mask 完全一致；
4. pose Jacobian：有限差分相对误差 `<1e-2`；
5. BA 扰动测试：GT pose 加 5 cm/2° 扰动，固定 GT range，优化后 pose 误差至少降低 90%；
6. tiny sequence：20–50 帧只开左目时间边，跟踪完成率 100%，无 NaN/Inf。

结果保存：`debug/localization_l2/trajectory.txt`、`reprojection_overlay/`、`ba_perturbation_report.json`。

### L2.3 这一步的结果

得到“单左目时间边 + GT range”的鱼眼位姿跟踪。尺度此时由 GT range 提供，还不能宣称已完成双目 SLAM。

---

## L3：同时刻左右 stereo factor

### L3.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/factor_graph.py:add_factors()` | 不再仅用 `ii==jj` 暗示 stereo；显式增加 `source_cam/target_cam/edge_type` |
| `hislam2/factor_graph.py` | 定义 `TEMPORAL_LEFT`、`STEREO_LR`，后续可消融 `TEMPORAL_RIGHT` |
| `hislam2/modules/corr.py` | stereo edge 使用左 feature 与右 feature 的 correlation |
| `hislam2/geom/projective_ops.py` | 实现 `left uv+range -> X_l -> T_right_left -> right uv` |
| `src/droid_kernels.cu` | BA 中按 camera id 使用左右模型和固定外参；stereo edge 不增加新 pose |
| `[拟新增] hislam2/geom/stereo_fisheye.py` | CPU 射线三角化和左右几何参考实现 |

### L3.2 测试怎么做

| 测试 | 通过标准 |
|---|---|
| 左-右-左往返 | 人工可见点 p99 `<1e-3 px` |
| 三角化 | 人工点 range p99 相对误差 `<1e-3` |
| GT range 对照 | 若左右都有 GT range，遮挡过滤后三维一致性误差 p50 `<2%`、p95 `<10%`；若只有左目 GT，本项只产生 overlay，不伪造数值门限 | 
| stereo BA 扰动 | range 加 10% 比例扰动后，尺度误差至少降低 80% |
| 外参固定 | 优化前后 `T_right_left` 逐元素完全不变 |
| edge 检查 | 每个关键帧至少有指定的 stereo edge，不被重复边过滤器误删 |

结果保存：`debug/localization_l3/stereo_overlay/`、`triangulation_report.json`、`edge_dump.txt`。

### L3.3 这一步的结果

定位因子图同时包含左目时间约束和左右空间约束，且左右相机始终共享一个 rig pose。

---

## L4：米制局部 BA 和深度状态

### L4.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/depth_video.py:normalize()` | stereo metric 模式禁止单目尺度归一化 |
| `hislam2/depth_video.py:cuda_ba()` | 先 pose-only，再开放 inv-range 状态；传递 range confidence/mask |
| `hislam2/geom/ba.py:JDSA()` | GT range 模式暂停现有单目 scale grid，后续只允许它作为弱先验 |
| `hislam2/factor_graph.py:update()` | 分开 temporal/stereo/GT-range 残差和权重日志 |

### L4.2 分三次测试

1. `pose_only + fixed_gt_range`：只确认位姿求解；
2. `pose_and_range + gt_range_anchor`：允许 inv-range 小幅优化；
3. `pose_and_range + no_gt_anchor`：关闭 GT range loss，只依靠时间边和 stereo edge 检查米制尺度。

通过标准：

- 三次实验都无 NaN/Inf；
- GT range 观测 Tensor 在 BA 前后 bitwise 不变；
- 第 3 次中估计轨迹与 GT 的中位尺度比保持在 `[0.98,1.02]`（tiny sequence 工程门限）；
- stereo reprojection median 不比优化前大，且主要迭代中持续下降；
- 输出 `temporal_loss/stereo_loss/range_anchor_loss/scale_ratio` 曲线。

### L4.3 这一步的结果

得到可独立运行的 `localization_gt_range` 局部定位版本。

---

## L5：回环、PGBA、离线 BA 和轨迹

### L5.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/pgo_buffer.py` | 回环边调用原生鱼眼重投影 |
| `hislam2/factor_graph.py:update_pgba()` | 支持双目 rig 状态和 metric range |
| `hislam2/depth_video.py:cuda_pgba()/distance_covis()` | 替换剩余针孔几何；禁止缩放 GT/stereo 观测 |
| `src/droid_kernels.cu` | 替换 PGBA、frame distance、covis distance 中的投影 |
| `hislam2/track_backend.py` | 确认全局因子使用双目 rig 数据 |
| `hislam2/trajectory_filler.py` | 非关键帧位姿填充使用同一 CameraModel |
| `demo.py:save_trajectory()` | 分别输出 rig/left/right 轨迹，标明 pose 方向 |

### L5.2 测试怎么做

- 用一段真正回到起点的 validation sequence；
- 对比 loop off/on 的闭环误差、ATE、RPE 和 scale ratio；
- loop on 后闭环误差必须下降，scale ratio 不能出现超过 2% 的突变；
- 先保留 Sim(3) scale 作为诊断；若长序列 scale 始终接近 1，再将 stereo 模式收紧为 SE(3)；
- 抽查 20 帧，`T_right_world = T_right_left * T_left_world` 恒成立，误差 `<1e-6`；
- 输出 `traj_rig.txt/traj_left.txt/traj_right.txt/localization_metrics.json`。

### L5.3 定位线最终结果

在完全关闭 Gaussian 建图时，系统仍能完成关键帧选择、左右匹配、局部 BA、回环、全局 BA 和轨迹评价。

---

# C. Gaussian 建图部分

## M0：建立独立 oracle mapping 入口

### M0.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `[拟新增] scripts/run_oracle_mapping.py` | 直接读取 `StereoFisheyeFrame`，使用 GT pose/range 构造 `MappingPacket` |
| `hislam2/gs_backend.py:process_track_data()` | 接收统一 `MappingPacket`，不要求定位线已运行 |
| `config` | 增加 `mode: oracle_mapping`、`mapping.enable`、`localization.enable` |

### M0.2 测试与结果

- 一帧和 10 帧数据能构造 `MappingPacket`；
- packet 中左右 pose 基线固定、range 单位为 m、mask 无错位；
- 定位模块全部关闭时程序仍可进入 `GSBackEnd`；
- 输出 `debug/oracle_packet.pt` 和字段报告。

---

## M1：鱼眼 Gaussian Camera

### M1.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/gaussian/utils/camera_utils.py:Camera` | 保存 camera id、CameraModel 参数、ray LUT、rig pose 和固定外参 |
| `hislam2/gaussian/utils/camera_utils.py:init_from_tracking()` | 输入 range/mask，不再依赖 perspective projection matrix |
| `hislam2/gaussian/utils/graphics_utils.py` | 建图主路径不再调用 `getProjectionMatrix2()` |
| `hislam2/gs_backend.py:process_track_data()` | 为左右视图创建两个 Camera，但它们共享一个 rig pose |

### M1.2 测试与结果

- 给定一个 rig pose，left/right Camera 中心距离等于标定 baseline，误差 `<1e-6 m`；
- 改变 rig pose 后左右相对位姿不变；
- `Camera` 能返回任意像素 ray，与 S1 参考误差 `<1e-6`；
- 输出 `debug/mapping_m1/camera_frustums.ply`。

---

## M2：用 range 初始化 Gaussian

### M2.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/gaussian/scene/gaussian_model.py:create_pcd_from_image_and_depth()` | 删除主路径中 `Open3D PinholeCameraIntrinsic` 反投影，改为 `X_c=range*ray_lut` |
| `hislam2/gaussian/scene/gaussian_model.py:create_pcd_from_image()` | 传入 range/mask/camera model，根据原始鱼眼像素取颜色 |
| `hislam2/gaussian/utils/slam_utils.py:depths_to_points()` | 替换固定 `fx/fy/cx/cy`、`W/2/H/2` 逻辑 |
| `hislam2/gaussian/utils/slam_utils.py:depth_to_normal()` | 基于鱼眼三维点计算法线，跳过 invalid/深度边界 |

### M2.2 测试怎么做

1. 单帧反投影点数等于 valid mask 经采样后的点数；
2. GPU 点云与 S1 CPU 参考点云的 p99 距离 `<1e-5 m`；
3. 将世界点投回初始帧，p99 像素误差 `<0.1 px`；
4. `ray_z<0` 后向边缘的有效点未被错误删除；
5. 点云不出现 NaN、长针或尺度突变。

结果保存：`debug/mapping_m2/frame_left.ply`、`frame_right.ply`、`reprojection_overlay.png`。

---

## M3：原生鱼眼 Gaussian rasterizer forward

### M3.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/gaussian/renderer/__init__.py:render()` | raster settings 传 CameraModel 参数/ray LUT，不再传 `tanfovx/tanfovy/projmatrix` 作为主投影 |
| `thirdparty/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | Python autograd 入口增加 camera model 数据 |
| `thirdparty/diff-gaussian-rasterization/ext.cpp` | C++ 绑定增加鱼眼参数 |
| `thirdparty/diff-gaussian-rasterization/rasterize_points.cu` | 转发新参数 |
| `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | Gaussian 中心用 CameraModel 投影；使用 `J_project_X` 投影协方差；输出 range |
| `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu` | 鱼眼有效区域、bounding box、tile culling 和按 range 排序 |
| `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h` | 新增设备端鱼眼投影/Jacobian 帮助函数 |

### M3.2 测试怎么做

| 测试 | 输入 | 通过标准 |
|---|---|---|
| 单 Gaussian 中心 | 已知世界点 | 渲染中心与 CameraModel 预期 `<0.5 px` |
| 二维协方差 | 小尺度 Gaussian | CUDA ellipse 与数值采样边界的主轴/半径误差 `<2%` |
| range | 单 Gaussian | 渲染 range 与相机中心到 Gaussian 的距离误差 `<1e-4 m` |
| 后向边缘 | `ray_z<0` 上的 Gaussian | 可见且中心位置正确 |
| tile 边界 | 跨 tile Gaussian | 无裂缝、重复、整块消失 |
| CPU/CUDA 小场景 | 10–100 Gaussian | 非边界像素 RGB 最大绝对误差 `<1e-4`，range p99 误差 `<1e-4 m`，visibility mask 一致 |

输出 `debug/rasterizer_forward/`，包含每个人工场景的预期图、实际图和 difference image。

### M3.3 这一步的结果

原生鱼眼 rasterizer 能生成 RGB、range、visibility、radii 和 `n_touched`，但暂不进行反向优化。

---

## M4：rasterizer backward 和单帧 mapping

### M4.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` | 增加鱼眼 project/Jacobian 链式求导，返回 xyz/scale/rotation/opacity/color/pose 梯度 |
| `hislam2/gaussian/utils/slam_utils.py` | RGB/range/normal loss 全部使用 valid mask |
| `hislam2/gs_backend.py:initialize_map()` | 先固定 GT pose，只优化 Gaussian |
| `hislam2/gaussian/scene/gaussian_model.py` | densification 统计使用鱼眼可见性和 radii |

### M4.2 测试怎么做

- 对 xyz/scale/opacity 各选若干参数做中心有限差分，相对梯度误差 `<1e-2`；
- pose 平移/旋转梯度与有限差分方向一致；
- 单帧 200 次优化中 total loss 相比前 10 次平均至少下降 50%；
- 优化全程无 NaN/Inf，Gaussian 数量、可见数量和梯度范数有日志；
- 输出初始/优化后 RGB、range、difference 和 loss curve。

### M4.3 这一步的结果

`oracle_mapping` 能在一帧上运行，证明鱼眼 Gaussian forward/backward 基本正确。

---

## M5：多帧左目 oracle mapping

### M5.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/gs_backend.py:process_track_data()` | 管理关键帧窗口和 GT rig pose |
| `hislam2/gs_backend.py:add_next_kf()` | 使用鱼眼 range 增加 Gaussian |
| `hislam2/gs_backend.py:map()` | 对多个左目 Camera 累加 RGB/range/normal loss |
| `hislam2/gaussian/utils/eval_utils.py` | 保存原始鱼眼渲染和分项指标 |

### M5.2 测试怎么做

- 用 10 帧 GT pose/range，其中 8 帧建图、2 帧留作 held-out view；
- mapping loss 能稳定下降，新关键帧加入时不出现全局数值爆炸；
- held-out RGB PSNR 相比未优化初始地图提升至少 3 dB；
- held-out range AbsRel 相比初始地图降低至少 30%；
- 输出 `3dgs_oracle_left.ply`、held-out render 和 `mapping_metrics.json`。

---

## M6：左右双视图 Gaussian mapping

### M6.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/gs_backend.py` | 每个 rig 关键帧保存 left/right Camera，两者共享 rig pose delta |
| `hislam2/gs_backend.py:map()` | 分开计算 `rgb_left/rgb_right/range_left/range_right`，再加权求和 |
| `hislam2/gaussian/scene/gaussian_model.py` | 左右 range 都可增加 Gaussian，通过空间距离/可见性避免无限重复 |
| `hislam2/gaussian/utils/slam_utils.py` | 右目没有 GT range 时只使用 RGB loss，不对 invalid 区域算 range loss |

### M6.2 测试怎么做

1. 同一 Gaussian 分别渲染到左右图，中心与 CameraModel 预期各自 `<0.5 px`；
2. 运行左目 mapping 基线与左右 mapping；
3. 双视图版右目 held-out PSNR 应明显优于只左目版，并报告具体差值；
4. 左目质量不应因加入右目大幅下降，PSNR 退化工程门限 `<1 dB`；
5. 共享 rig pose delta 更新后左右 baseline 误差 `<1e-6 m`；
6. 输出 `3dgs_oracle_stereo.ply` 和左/右分项报告。

---

## M7：在线 mapping、回环更新、离线 refinement 和 mesh

### M7.1 改哪里、改什么

| 文件 | 具体改动 |
|---|---|
| `hislam2/hi2.py:call_gs()` | 构造完整 `MappingPacket`，不再用 `1./disps_up` 隐式解释 depth |
| `hislam2/gs_backend.py:process_track_data()` | 接收定位 pose correction，更新相机和 Gaussian |
| `hislam2/gs_backend.py:color_refinement()` | 每个 rig 关键帧只有一份 pose delta，左右 Camera 共享；曝光参数可以每相机独立 |
| `hislam2/gaussian/utils/eval_utils.py` | 输出左右 RGB/range/mask 和指标 |
| `tsdf_integrate.py` | 将针孔融合入口替换为 ray+range 的原生鱼眼表面融合 |

### M7.2 测试怎么做

- 人工给所有 pose 一个已知 SE(3) correction，更新后 Gaussian 与相机相对投影保持不变；
- 如仍保留 Sim(3) 诊断，仅可缩放可优化地图状态，GT/stereo range observation 的 hash 前后相同；
- 50 帧在线 mapping 的 packet 帧号严格递增，无重复/漏帧/过期 pose；
- refinement 后 held-out PSNR 不低于 refinement 前，range 误差不恶化超过 2%；
- mesh 与 GT 点云在米制坐标中对齐，输出 `mesh.ply`、`3dgs_final.ply`、左右 render 和 mapping metrics。

### M7.3 建图线最终结果

在 GT pose/range 的 oracle 模式下，原生鱼眼 Gaussian 建图可以独立完成；在线模式下，可以消费定位线输出的 rig pose 和 correction。

---

# D. 双目鱼眼深度网络

## D0：保持 DepthProvider 接口不变

最终网络必须返回与 S2 完全相同的 `range/inv_range/valid/confidence/source`。后端不能通过 `if network` 到处分叉。

## D1：数据和基线网络

| 项目 | 内容 |
|---|---|
| 输入 | 左右原始鱼眼 RGB、两套 CameraModel 参数、固定外参 |
| 输出 | 左目 inv-range + confidence；后续扩展右目 |
| 监督 | GT range/inv-range + valid mask |
| 分区 | stereo overlap/non-overlap、中心/边缘、`ray_z>=0`/`ray_z<0` |
| 对照 | 单目鱼眼网络 vs 双目鱼眼网络 |

## D2：几何损失

- GT range/inv-range 监督；
- 左右 CameraModel 重投影一致性；
- 遮挡感知 photometric/feature loss；
- 左右三维点一致性；
- 边界感知平滑项；
- confidence calibration loss。

## D3：接入 SLAM

### 改哪里

- `[拟新增] hislam2/depth/stereo_fisheye_depth_provider.py`；
- `hislam2/motion_filter.py`：调用 provider，不再固定调用 OmniData depth；
- `config`：`depth.provider: gt | stereo_network`。

### 测试怎么做

1. provider contract test：GT/network 两种 provider 的 key、shape、dtype、device、单位完全一致；
2. offline depth test：报告 MAE/RMSE/AbsRel/有效覆盖率；
3. 单目/双目消融：分区报告改善，用数据证明 200° 宽视场和 stereo 输入的作用；
4. 只改配置一项就能从 `full_gt_range` 切到 `full_pred_range`；
5. 第一版接入时冻结深度网络，不做端到端反向。

结果：深度来源可替换，定位和建图代码不需要重写。

---

# E. 系统联调与明确测试

## I0：一帧 oracle mapping

| 项目 | 内容 |
|---|---|
| 输入 | 1 对 RGB + GT pose + GT range |
| 开启 | CameraModel + Gaussian mapping |
| 关闭 | 定位、回环、深度网络 |
| 必须输出 | 点云、Gaussian PLY、左右 RGB/range render、loss curve |
| 通过 | 没有几何错位；loss 下降至少 50%；后向边缘渲染正确 |

## I1：定位独立测试

| 项目 | 内容 |
|---|---|
| 输入 | 20–50 对 RGB + GT range，GT pose 只评价 |
| 开启 | temporal edge + stereo edge + local BA |
| 关闭 | Gaussian mapping、回环、深度网络 |
| 必须输出 | rig/left/right trajectory、reprojection loss、scale ratio |
| 通过 | 跟踪完成率 100%；无 NaN；scale ratio 在 `[0.98,1.02]`；左右 baseline 不变 |

## I2：建图独立测试

| 项目 | 内容 |
|---|---|
| 输入 | 10–50 对 RGB + GT pose + GT range |
| 开启 | stereo Gaussian mapping + refinement |
| 关闭 | SLAM 定位、回环、深度网络 |
| 必须输出 | Gaussian PLY、左右 held-out render、range、mesh |
| 通过 | held-out PSNR 相比初始提升 `>=3 dB`；range AbsRel 降低 `>=30%`；无大面积几何错位 |

## I3：真值深度完整 pipeline

| 项目 | 内容 |
|---|---|
| 输入 | validation sequence 左右 RGB + 标定 + GT range，GT pose 只评价 |
| 开启 | 定位 + stereo BA + mapping + loop + refinement |
| 关闭 | 深度网络 |
| 必须输出 | 轨迹、ATE/RPE、Gaussian、左右 render、mesh、运行时间和显存 |
| 通过 | 整序列完成；无尺度跳变；loop 后闭环误差下降；mapping 未使用 GT pose |

## I4：网络深度完整 pipeline

| 项目 | 内容 |
|---|---|
| 输入 | 左右 RGB + 标定，GT pose/range 只离线评价 |
| 开启 | 全部最终模块 |
| 必须输出 | I3 的全部输出 + 网络 range/confidence 指标 |
| 对照 | I3 是上限参考；同时对比单目鱼眼 depth network |
| 通过 | 不读取 GT pose/range 也能运行；误差变化能由 depth confidence/跟踪/mapping 分项解释 |

---

## F. 最终必须产出的结果文件

```text
output/
  config_used.yaml
  run_log.txt
  data_manifest.csv
  traj_rig.txt
  traj_left.txt
  traj_right.txt
  localization_metrics.json
  depth_metrics.json
  mapping_metrics.json
  runtime_metrics.json
  3dgs_final.ply
  mesh.ply
  renders/
    left_rgb/
    right_rgb/
    left_range/
    right_range/
    left_error/
    right_error/
    valid_masks/
  debug/
    reprojection_overlay/
    stereo_overlay/
    loss_curves/
    scale_curve/
```

如果某个阶段没有它应该产生的结果文件，就不算验收通过。

---

## G. 新手每个小步骤的执行方法

每次只做一个编号，例如只做 `L2`：

1. 阅读该小步骤列出的文件；
2. 标出需要修改的函数；
3. 写出每个 Tensor 的 shape、单位和坐标系；
4. 先写人工数据测试；
5. 修改当前文件，不顺手改下一模块；
6. 运行人工测试、单帧真实测试、tiny sequence；
7. 保存本步规定的报告和图像；
8. 通过标准全部满足后，才进入下一编号。

建议记录表：

| 函数 | 输入 shape | 输出 shape | 坐标系 | 单位 | valid mask | 测试 |
|---|---|---|---|---|---|---|

---

## H. 第一次实际开工只做什么

只做 `S0`，不修改跟踪、CUDA 或 Gaussian rasterizer。需要先确认：

1. 左右 RGB 的完整目录规则和时间戳；
2. 左右 camera model 名称、参数顺序和标定分辨率；
3. `T_right_left` 的原始数值和方向；
4. GT pose 是 `world -> rig` 还是 `rig -> world`；
5. GT depth 是 ray range、Z-depth 还是其他定义，单位和 invalid 值是什么；
6. GT depth 是只有左目还是左右都有。

`S0` 的最终产物是一个可检查的 `StereoFisheyeFrame`、数据 manifest、pose 轨迹可视化和 GT range 点云。这些都正确后再做 `S1`。
