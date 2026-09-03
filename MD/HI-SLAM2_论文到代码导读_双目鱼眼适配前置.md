# HI-SLAM2：从论文到代码的中文导读

> 目标：先建立对整个工程的统一认识，再为后续“双目鱼眼 + Gaussian SLAM”适配做准备。本文只解释现有实现，不修改算法代码。
>
> 对照论文：[HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction](https://arxiv.org/abs/2411.17982)。当前在线论文已经更新到 v3（2026-02-02），而本仓库主体文件时间为 2025-05，因此论文与代码存在少量差异，本文会明确指出。
>
> 当前项目决策：系统从输入到跟踪、Gaussian 渲染和表面融合始终工作在原始双目鱼眼像素域。实际执行以 `HI-SLAM2_新手源码阅读路线.md` 的原生鱼眼阶段为准。

---

## 0. 先用一句话看懂这个项目

HI-SLAM2 不是“用高斯地图直接跟踪相机”的 map-centric SLAM，而是一套混合式系统：

1. DROID-SLAM 风格的学习式稠密跟踪器，从单目 RGB 序列估计关键帧位姿和稠密逆深度；
2. OmniData 提供单目深度、法线先验，JDSA 把尺度不一致的深度先验对齐到 BA 深度；
3. 关键帧位姿和深度送入 3DGS 后端，初始化、优化并增密高斯地图；
4. 检测到回环后，使用 `Sim(3)` PGBA 修正位姿和单目尺度漂移，并按关键帧锚点直接变形高斯；
5. 序列结束后插入补充关键帧、执行全局 BA，再联合优化高斯、相机位姿和曝光，最后渲染深度并进行 TSDF 融合得到网格。

最重要的系统边界是：

```text
RGB 图像
   │
   ├── DROID 特征/光流关联 ──> SE(3) 位姿 + 稠密逆深度
   │                                ▲
   └── OmniData ──> 深度先验 ──JDSA─┘
              └──> 法线先验 ──────────────────┐
                                              ▼
位姿 + 深度 + RGB + 法线 ───────────────> 3D Gaussian Map
        ▲                                     │
        └──── Sim(3) PGBA / Full BA ──> 锚定高斯变形
                                              │
                                              └──> RGB/Depth 渲染 ──> TSDF Mesh
```

对双目鱼眼适配而言，这个边界意味着：跟踪前端和高斯后端虽然解耦，但二者都各自写死了针孔模型，不能只改图像读取或标定文件。

---

## 1. 论文第一章：为什么采用混合式 Gaussian SLAM

论文要解决的是单目稠密重建中的三个矛盾：

- 单目没有绝对尺度，深度和尺度容易漂移；
- 纯几何稠密 SLAM 在弱纹理、遮挡区域的深度不可靠；
- Neural SLAM 往往在几何精度、渲染质量和速度之间取舍。

HI-SLAM2 的核心思想不是让高斯承担所有职责，而是分工：

| 子系统 | 擅长的事情 | 主要输出 |
|---|---|---|
| DROID 式稠密跟踪 | 跨帧匹配、位姿和多视几何深度 | 关键帧 `SE(3)` 位姿、逆深度、置信度 |
| OmniData + JDSA | 补足弱纹理区域，纠正单目先验的空间尺度畸变 | 对齐后的先验和更稳定的 BA 深度 |
| 3DGS | 显式场景表示、快速可微渲染、增量建图 | 高斯均值/尺度/旋转/颜色/不透明度 |
| PGBA / Full BA | 消除全局位姿和尺度漂移 | 更新后的位姿、深度、尺度 |
| 锚定高斯变形 | 不重新训练整张图就能应用回环修正 | 全局一致的高斯地图 |

代码总入口在 `demo.py`，系统编排器在 `hislam2/hi2.py`。

---

## 2. 论文第二章：相关工作在代码里分别来自哪里

论文第二章本身没有对应的独立实现，但有助于理解代码来源。

### 2.1 深度估计

- 多视几何深度：来自 DROID-SLAM 的稠密光流关联和 BA，主要在：
  - `hislam2/modules/droid_net.py`
  - `hislam2/factor_graph.py`
  - `hislam2/depth_video.py`
  - `src/droid_kernels.cu`
- 单目深度与法线先验：OmniData，主要在：
  - `hislam2/motion_filter.py:65-83`
  - `hislam2/midas/omnidata.py`
- 两种深度的融合：JDSA，主要在：
  - `hislam2/geom/ba.py:161-218`
  - `src/droid_kernels.cu` 中 `proj_trans`、`bi_inter` 等 CUDA 实现。

### 2.2 表面重建

- 3DGS 数据结构和增密/裁剪来自原始 Gaussian Splatting：
  - `hislam2/gaussian/scene/gaussian_model.py`
  - `thirdparty/simple-knn/`
- 深度可微栅格化来自修改后的 rasterizer，并引入 RaDe-GS 风格的无偏深度：
  - `thirdparty/diff-gaussian-rasterization/`
- 最终显式网格不是直接从 Gaussian 椭球提取，而是：
  - 先用最终高斯地图逐帧渲染深度；
  - 再由 `tsdf_integrate.py` 融合成 TSDF 网格。

### 2.3 稠密视觉 SLAM

- 跟踪骨架来自 DROID-SLAM；
- 高斯建图组织方式受 MonoGS 等工作影响；
- 整体创新点位于“DROID 深度 + 空间尺度对齐 + 几何监督 3DGS + 可变形地图”之间的接口。

因此，理解本工程时不要从 `GaussianModel` 一头扎进去。正确顺序是先看 `demo.py -> Hi2 -> MotionFilter/TrackFrontend -> DepthVideo/FactorGraph`，再看 `GSBackEnd`。

---

## 3. 论文第三章：方法与代码逐节对应

## 3.1 III-A：相对 HI-SLAM 的变化

论文总结了三项升级，在本仓库中的落点如下。

### 变化一：从单尺度先验对齐升级到 2×2 尺度网格

每个关键帧保存一个 `2×2` 尺度网格：

```python
self.dscales = torch.ones(buffer, 2, 2, ...)
```

位置：`hislam2/depth_video.py:48`。

`hislam2/geom/ba.py:get_prior_depth_aligned()` 在整幅图上双线性插值这个网格，然后逐像素乘到单目先验上。这样图像左上、右上、左下、右下可以具有不同尺度，不再假设整幅预测深度只差一个统一比例。

需要特别注意：论文把变量写成 depth，代码的优化主变量实际是 disparity/inverse depth。`DepthVideo.disps`、`disps_prior`、`disps_up` 都是逆深度；传给高斯后端前才在 `Hi2.call_gs()` 中执行 `1. / disps_up` 变回深度。

### 变化二：从隐式场景表示升级到显式 3DGS

核心数据在 `GaussianModel`：

- `_xyz`：世界坐标中的高斯中心；
- `_scaling`：对数域尺度，读取时经过 `exp`；
- `_rotation`：四元数，读取时归一化；
- `_features_dc`：颜色的 0 阶 SH/DC 表示；
- `_opacity`：logit，不透明度读取时经过 `sigmoid`；
- `unique_kfIDs`：每个高斯所属的锚定关键帧，这是快速地图变形的关键。

系统用 `sh_degree=0` 创建模型，因此本质上只保留与视角无关的常量颜色，而不是完整的高阶球谐外观。

### 变化三：分层优化

真实代码执行顺序为：

```text
在线局部 BA + JDSA
        ↓
在线 Sim(3) PGBA（配置可关闭）
        ↓
结束时补充关键帧
        ↓
离线全局 BA
        ↓
Gaussian + Pose + Exposure 联合 Adam 优化
```

对应编排集中在 `hislam2/hi2.py:86-165`。

---

## 3.2 III-B：Online Tracking 在线跟踪

### 3.2.1 输入读取与相机模型

`demo.py:30-63` 的 `mono_stream()` 完成：

1. 从目录按文件名排序读取图像；
2. 从标定文本读取 `[fx, fy, cx, cy, distortion...]`；
3. 当前原工程包含一个可选的普通相机预处理分支，原生鱼眼改造不使用该分支；
4. 把图像等比例缩放到约 `341×640` 像素量级，并把宽高裁到 8 的倍数；
5. 同比例修改内参；
6. 通过队列将 `(时间戳, 图像, 内参, 是否最后一帧)` 送给主进程。

此处所谓时间戳 `t` 默认只是从 0 开始的图像序号。保存轨迹时才从文件名中提取数字作为输出时间戳。

当前输入模型的本质限制：

- 只接收一幅图像；
- 标定核心只有 4 个针孔参数；
- 后续优化器无法接收完整的非线性鱼眼相机模型；
- 后续优化器完全不知道畸变参数。

### 3.2.2 关键帧前置筛选：MotionFilter

`hislam2/motion_filter.py` 并不是最终关键帧管理器，而是第一层“候选帧筛选”。

每帧都会：

1. 用 `fnet` 提取 1/8 分辨率特征；
2. 与上一个已接收关键帧构建相关体；
3. 只执行一次 recurrent update，估计平均光流幅度；
4. 超过 `motion_filter.thresh` 才写入 `DepthVideo`；
5. 如果启用 `skip_blur`，在最近缓存帧中优先选择拉普拉斯清晰度更好的帧；
6. 对被接受的帧运行 OmniData，生成全分辨率深度和法线先验；
7. 保存特征图、上下文隐状态和输入特征，供后续因子图复用。

关键输出尺寸，设输入为 `H×W`：

| 字段 | 典型形状 | 含义 |
|---|---:|---|
| `images` | `[N,3,H,W]` | uint8 关键帧 RGB |
| `fmaps` | `[N,1,128,H/8,W/8]` | 相关特征 |
| `nets` | `[N,128,H/8,W/8]` | GRU 隐状态 |
| `inps` | `[N,128,H/8,W/8]` | 上下文输入 |
| `disps` | `[N,H/8,W/8]` | 优化中的逆深度 |
| `disps_up` | `[N,H,W]` | 凸上采样后的逆深度 |
| `disps_prior` | `[N,H/8,W/8]` | OmniData 逆深度先验 |
| `normals` | `[N,3,H,W]` | OmniData 法线先验 |
| `poses` | `[N,7]` | `tx,ty,tz,qx,qy,qz,qw` 的世界到相机位姿 |

统一状态容器就是 `hislam2/depth_video.py:DepthVideo`。

### 3.2.3 DROID 网络在这里做什么

`hislam2/modules/droid_net.py` 包含三部分：

- `fnet`：提取相关特征；
- `cnet`：产生 GRU 初始隐状态 `net` 和上下文 `inp`；
- `UpdateModule`：根据相关特征和当前几何运动特征，反复输出：
  - `delta`：目标像素坐标修正；
  - `weight`：每个像素匹配的置信度；
  - `eta`：BA 阻尼；
  - `upmask`：逆深度的学习式凸上采样权重。

网络不直接输出最终位姿。它预测稠密对应点和置信度，位姿与深度由几何 BA 求解。这是理解整个跟踪器的关键。

### 3.2.4 因子图与论文公式（1）

`hislam2/factor_graph.py` 中：

- 图节点是关键帧；
- `ii[k] -> jj[k]` 表示第 `k` 条有向重投影边；
- `target` 是 recurrent optical flow 网络修正后的目标坐标；
- `weight` 是目标坐标的置信度；
- `corr` 是边两端特征的多尺度相关体；
- 过老的活动边会转存为 inactive edge，供后续 PGBA 构造相对位姿约束。

论文公式（1）的实际循环是：

```text
当前 pose/depth 重投影得到 coords1
       ↓
读取 corr(coords1) + 当前光流/残差
       ↓
GRU 输出 delta 和 weight
       ↓
target = coords1 + delta
       ↓
CUDA BA 最小化 target 与几何重投影的加权误差
       ↓
更新 pose 和 per-pixel inverse depth
```

Python 的投影链在：

- `hislam2/geom/projective_ops.py`
- `hislam2/geom/pinhole.py`

实时求解使用的高性能版本在：

- `src/droid_kernels.cu`
- `src/droid.cpp`

这里存在非常明确的针孔假设：

```text
u = fx * X/Z + cx
v = fy * Y/Z + cy
X = (u-cx)/fx
Y = (v-cy)/fy
```

而且 CUDA 中 BA 的投影 Jacobian也按针孔公式手写。后续若走原生鱼眼模型，这里是必须整体替换的一层，而非只替换 `demo.py`。

### 3.2.5 初始化与局部 BA

`TrackFrontend` 的固定 warmup 是 12 个关键帧，与论文的 `N_init=12` 一致。

初始化阶段：

1. 对相隔不超过 3 帧的关键帧建双向边；
2. 先执行不使用单目先验的 BA；
3. 为每帧初始化尺度网格；
4. 再交替执行 BA 与 JDSA；
5. 删除运动过小、过于接近的关键帧；
6. 调用 `DepthVideo.normalize()` 归一化全局单目尺度；
7. 固定前两个位姿，阻止后续 BA 自由改变系统尺度。

初始化完成后，每来一个候选关键帧：

- 根据重叠距离添加邻接边和近邻边；
- 执行多轮 update + BA；
- 后几轮启用 JDSA；
- 如果中间关键帧与邻帧运动太小且共视性良好，就删除它；
- 将更新过的关键帧索引送给高斯后端。

### 3.2.6 JDSA 与论文公式（2）—（5）

JDSA 的任务不是直接用单目深度替换 BA 深度，而是让二者互相约束：

- BA 深度受多视重投影约束，在纹理丰富区域可信；
- OmniData 先验在弱纹理区域仍给出形状，但每帧、甚至同一帧不同区域存在尺度畸变；
- 2×2 scale grid 用双线性插值生成逐像素尺度；
- JDSA 联合更新逆深度与尺度网格，但不在同一步更新位姿；
- 系统在普通 BA 和 JDSA 之间交替，降低尺度漂移和数值不稳定。

代码路径：

```text
FactorGraph.update(use_mono=True)
  -> DepthVideo.cuda_ba()
     -> droid_backends.ba()             # 先更新 pose + disparity
     -> geom.ba.JDSA()                   # 再更新 disparity + 2×2 scales
        -> droid_backends.proj_trans()   # 重投影项对深度的 Hessian/梯度
        -> droid_backends.bi_inter()     # 尺度网格双线性插值及 Jacobian
        -> schur_solve_mono_prior()      # Schur 求解
```

JDSA 仍依赖针孔重投影 CUDA 核心，所以即使双目直接给出米制深度，只要继续保留这套 JDSA，也必须确认鱼眼/虚拟相机投影与深度定义一致。

---

## 3.3 III-C：Online Loop Closing 在线回环

### 3.3.1 为什么使用 Sim(3)

单目系统除 6DoF 位姿漂移外还有尺度漂移。`SE(3)` 只有旋转和平移，无法让不同关键帧获得尺度修正；`Sim(3)` 增加一个尺度自由度，所以在线回环阶段临时把所有位姿从 7 维 `SE(3)` 扩展为 8 维 `Sim(3)`。

### 3.3.2 回环候选检测

`hislam2/pgo_buffer.py:search_lc_candidate()` 对较新的关键帧与历史帧计算 `DepthVideo.distance()`：

- 距离本质是基于当前位姿和深度的平均重投影光流距离；
- 需小于 `pgba_thresh`；
- 相对姿态欧拉角范数需小于约 120°；
- 在线线程还要求足够大的时间间隔，并在累积候选达到条件后才触发 PGBA。

论文中的三个判据“光流距离、方向差、时间间隔”在代码中均存在，但具体阈值和调度写在 `PGOBuffer.spin()` 内，不全在 YAML 中开放。

### 3.3.3 相对位姿因子如何产生

局部窗口中的边失活时，`FactorGraph.rm_factors(store=True)` 会调用 `PGOBuffer.add_rel_poses()`：

1. 保存已经由 recurrent update 多次精化的稠密对应；
2. 在固定深度的条件下，仅优化该边的相对 `SE(3)`；
3. 由残差、Jacobian 和 Hessian 估计相对位姿协方差；
4. 保存 `(rel_ii, rel_jj, rel_pose, rel_cov)`，作为论文公式（6）（7）中的相对位姿因子。

### 3.3.4 PGBA 的优化与输出

PGBA 同时使用：

- 检出的回环边上的稠密重投影因子；
- 历史 inactive edge 压缩得到的相对位姿因子；
- 每关键帧一个 `Sim(3)` 位姿；
- 关键帧逆深度。

路径为：

```text
PGOBuffer._pgba()
  -> FactorGraph.update_pgba()
     -> DepthVideo.cuda_pgba()
        -> global_relative_posesim3_constraints()
        -> droid_backends.pgba()
```

优化完成后：

- `Sim(3)` 位姿转回 `SE(3)`；
- 平移、逆深度、先验尺度网格按尺度修正；
- 计算每个关键帧优化前后的相对变换 `dposes` 和 `dscale`；
- 交给 Gaussian 后端直接变形地图。

### 3.3.5 并行关系

当前真实进程关系是：

- 图像读取：独立 `Process`；
- 主跟踪、局部 BA、Gaussian mapping：由主流程依次调用；
- 回环候选搜索：独立 `Process`；
- 可视化：可选独立 `Process`。

虽然 `GSBackEnd` 继承了 `mp.Process`，本仓库并未调用其 `start()`，而是直接调用 `process_track_data()`、`map()` 和 `finalize()`。所以在当前实现中 Gaussian mapping 与主线程同步，并不是一个真正独立运行的 mapper 进程。

---

## 3.4 III-D：3D Scene Representation

### 3.4.1 高斯参数，对应论文公式（9）

每个高斯包含：

| 论文变量 | 代码字段 | 内部参数化 |
|---|---|---|
| 均值 `mu` | `_xyz` | 直接世界坐标 |
| 尺度 `s` | `_scaling` | 对数域，读取时 `exp` |
| 旋转 `R` | `_rotation` | 四元数，读取时归一化 |
| 颜色 `c` | `_features_dc` | `sh_degree=0` 的常量颜色 |
| 不透明度 `o` | `_opacity` | logit，读取时 `sigmoid` |
| 锚点关键帧 | `unique_kfIDs` | 每个 Gaussian 一个原始帧时间戳 |

3D 协方差由旋转和三轴尺度组合：`GaussianModel.build_covariance_from_scaling_rotation()`。

### 3.4.2 针孔投影与可微栅格化，对应公式（10）（11）

`hislam2/gaussian/renderer/__init__.py` 把相机、Gaussian 参数传给 CUDA rasterizer。该 rasterizer 完成：

1. 世界坐标 Gaussian 变换到相机坐标；
2. 通过针孔透视投影得到二维均值；
3. 用透视投影 Jacobian 把 3D 协方差传播成 2D 协方差；
4. 按 tile 和深度排序；
5. 从近到远进行 alpha blending；
6. 输出 RGB、期望深度、可见性和屏幕半径；
7. 反向传播到 Gaussian 参数以及离线优化中的相机位姿增量。

鱼眼适配必须重视这里：`thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:76-188` 的 2D 协方差和 ray-plane 推导都基于透视针孔；它不是换一张投影矩阵就能变成鱼眼栅格化。

### 3.4.3 无偏深度

普通 3DGS 常用 Gaussian 中心的 `Z` 作为整个 splat 的深度，这会让一个倾斜椭球覆盖区域的深度成为常数或产生偏差。

修改后的 rasterizer 为每个 Gaussian 计算一个局部 ray-plane。在渲染某像素时，根据该像素相对 Gaussian 二维中心的偏移修正交点距离：

```text
t(pixel) = t_center + ray_plane_x * dx + ray_plane_y * dy
```

再以 alpha-transmittance 权重混合并归一化得到期望深度。主要实现在：

- `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:76-188`
- `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:440-509`
- 对应反向传播在 `backward.cu`。

这是 HI-SLAM2 几何精度的重要来源，也是原生鱼眼 rasterizer 最难复用的部分之一，因为 ray 的参数化和投影 Jacobian都会变化。

### 3.4.4 新关键帧如何生成高斯

入口：`GSBackEnd.add_next_kf()` -> `GaussianModel.extend_from_pcd_seq()`。

步骤：

1. 使用关键帧 RGB、跟踪器输出的全分辨率深度和针孔内参；
2. Open3D `create_from_rgbd_image()` 反投影到世界点云；
3. 随机下采样；
4. 点坐标作为 Gaussian 均值；
5. 对应像素 RGB 作为初始颜色；
6. 近邻距离决定初始尺度；
7. 旋转初始化为单位四元数；
8. opacity 初始化为 0.5；
9. 保存该批 Gaussian 的锚点关键帧 ID。

第一张图使用 `pcd_downsample_init`，后续图使用 `pcd_downsample`。

### 3.4.5 在线地图窗口和优化损失，对应公式（14）—（17）

`GSBackEnd.process_track_data()` 维护最近关键帧窗口，实际最多约 11 帧，并在每次有跟踪更新时调用 `map(..., iters=10)`。

每轮 mapping 使用：

- 当前局部窗口全部视角；
- 窗口外随机最多 2 个历史视角，降低遗忘和局部过拟合。

损失实现为：

1. RGB L1：渲染图与关键帧 RGB；
2. 逆深度 L1：`1/rendered_depth` 与 `1/tracker_depth`；
3. 法线余弦损失：从渲染深度反投影点云，再用邻域叉积求法线，与 OmniData 法线先验比较；
4. 各向同性正则：惩罚三轴尺度偏离各自均值，抑制过细长 Gaussian。

主要代码：

- `hislam2/gs_backend.py:204-273`
- `hislam2/gaussian/utils/slam_utils.py:93-121, 210-245`

默认 RGB/逆深度组合在配置没有显式 `alpha` 时取 `alpha=0.95`；尺度正则在代码中乘 10；法线权重由 YAML 的 `lambda_dnormal` 控制。

### 3.4.6 地图管理：增密、裁剪和 opacity reset

`GaussianModel.densify_and_prune()`：

- 屏幕空间梯度大的小 Gaussian 被 clone；
- 梯度大且世界尺度大的 Gaussian 被 split；
- opacity 过低且不属于最新锚定关键帧的 Gaussian 被删除；
- 过大的屏幕/世界空间 Gaussian 也可被删除；
- 定期 reset 非可见 Gaussian 的 opacity。

第一帧会先执行 `init_itr_num=1050` 轮较重的初始化；正常在线阶段每次关键帧更新只做 10 轮，但每 150 轮全局 iteration 执行一次增密/裁剪。

### 3.4.7 回环后如何立即更新地图，对应公式（12）

每个 Gaussian 都记住自己由哪个关键帧初始化。PGBA 或 full BA 返回关键帧修正后，`GSBackEnd.process_track_data()` 根据锚点执行：

- 高斯均值乘相对位姿并按尺度修正；
- 高斯尺度按关键帧尺度修正；
- 高斯旋转左乘关键帧相对旋转。

代码位置：`hislam2/gs_backend.py:78-95`。

这一步不通过成千上万次梯度下降“重新学地图”，而是显式改变每个 Gaussian，因此回环更新很快。后续 mapping 再消除变形后的局部不一致。

---

## 3.5 III-E：Offline Refinement 离线精化

序列结束由 `Hi2.terminate()` 驱动。

### 3.5.1 Post-Keyframe Insertion

论文写的是基于相邻帧视野覆盖率寻找观测不足区域。当前代码采用一种更工程化的近似：

1. 用 `distance_covis()` 评估关键帧相对前后邻帧的出视野比例；
2. 若高于 `covis_thresh`，就在该关键帧与前后关键帧之间各寻找一个原始帧；
3. `PoseTrajectoryFiller.fill()` 先做李群线性位姿插值；
4. 再把补充帧连接到两侧关键帧，通过 motion-only BA 精化其位姿；
5. 对补充帧提取 OmniData 先验并插回 `DepthVideo`；
6. 后续作为完整关键帧送进 Gaussian map。

代码位置：`hislam2/hi2.py:120-145` 和 `hislam2/util/trajectory_filler.py`。

### 3.5.2 Full Bundle Adjustment

`TrackBackend` 重建一个覆盖所有关键帧的低显存因子图：

- 通过 `add_proximity_factors(..., backend=True)` 搜索全局重叠帧对；
- 使用 `AltCorrBlock` 按小批量动态计算相关性，避免为全图保留所有相关体；
- 调用两次全局 BA，先 4 steps 再 8 steps；
- 当前代码调用 CUDA BA 时从 `t0=10` 开始优化，即前 10 个关键帧保持固定以稳定 gauge/尺度。

代码位置：

- `hislam2/track_backend.py`
- `hislam2/factor_graph.py:update_lowmem()`
- `hislam2/hi2.py:147-154`

### 3.5.3 Joint Pose and Map Refinement

全局 BA 后，先把位姿变化显式作用到锚定高斯，再进入 `GSBackEnd.color_refinement()`：

- 每次随机选择一张关键帧；
- 可微渲染 RGB 和 depth；
- 联合优化 Gaussian 参数；
- 优化每相机 3 维旋转增量和 3 维平移增量；
- 可选优化每相机曝光参数；
- Replica 默认 2000 轮，ScanNet/自有数据默认 26000 轮；
- 每次 Adam step 后将相机增量通过 `SE(3)` 指数映射写回相机位姿。

最终保存：

- `3dgs_final.ply`：高斯地图；
- `traj_kf.txt`：最终关键帧轨迹；
- `traj_full.txt`：对非关键帧补全后的完整轨迹；
- `intrinsics.npy`：输入分辨率尺度的针孔内参；
- `renders/`：最终 RGB、depth 及评估产物。

### 3.5.4 TSDF 网格

`tsdf_integrate.py` 读取：

- `renders/depth_after_opt/`；
- `renders/image_after_opt/`；
- `traj_full.txt`；
- `intrinsics.npy`。

然后用 Open3D 的针孔 TSDF 融合提取 `tsdf_mesh_w*.ply`。

所以“最终网格支持鱼眼”也不能只看 SLAM 主体。当前 Open3D 接口只认识针孔 K；原生鱼眼系统必须根据每个像素的真实鱼眼 ray 更新 TSDF，不能直接复用这段融合代码。

---

## 4. 一帧图像在代码中的完整生命周期

下面这条调用链建议在阅读代码时反复对照：

```text
demo.py::mono_stream
  读图、resize、同步更新鱼眼 CameraModel 参数
      ↓ queue
demo.py main loop
      ↓
Hi2.track(t, image, K)
      ↓
MotionFilter.track
  fnet 特征 -> 与上一关键帧做一次 flow update -> 是否接收
  若接收：OmniData depth/normal + cnet state -> DepthVideo.append
      ↓
TrackFrontend.__call__
  第 12 帧初始化；之后维护局部 FactorGraph
  recurrent correspondence update <-> CUDA BA <-> JDSA
  决定保留/删除关键帧，返回被更新索引 viz_idx
      ↓
PGOBuffer（并行）
  搜索回环；若触发，Sim(3) PGBA -> dpose/dscale
      ↓
Hi2.call_gs
  从 tracker 取 RGB、normal、1/disparity、K、w2c
      ↓
GSBackEnd.process_track_data
  应用回环地图变形
  新帧 depth 反投影 -> 添加 Gaussian
  当前窗口 map 10 iterations
      ↓
序列结束：Hi2.terminate
  补关键帧 -> Full BA -> 高斯变形 -> Joint refinement
  -> 全轨迹补全 -> 最终渲染与保存
```

---

## 5. 最重要的数据、坐标和尺度约定

这是后续适配最容易出错的部分。

### 5.1 位姿方向

- `DepthVideo.poses[i]` 是世界到相机 `T_cw`；
- 帧 `i` 到帧 `j` 的变换在代码中为 `T_jw * inverse(T_iw)`；
- Gaussian `_xyz` 在世界坐标；
- 高斯 `Camera.R/T` 也保存世界到相机变换；
- 输出轨迹前会取逆，保存为相机到世界 `T_wc` 的平移和四元数。

如果接入双目标定，必须先明确外参写的是 `T_right_left`、`T_left_right`、`T_cam_body` 还是其逆，不能只根据变量名猜。

### 5.2 深度与逆深度

| 环节 | 表示 |
|---|---|
| OmniData 原始输出 | 相对深度，代码额外乘 50 |
| `disps_prior(_up)` | OmniData 输出的倒数 |
| `disps(_up)` | BA 优化的逆深度 |
| JDSA | 对齐逆深度先验和 BA 逆深度 |
| `Hi2.call_gs()` | `1/disps_up`，转成深度 |
| Gaussian `Camera.depth` | 深度 |
| mapping depth loss | 实际比较二者的倒数，即逆深度 L1 |
| rasterizer 输出 | 沿相机 ray 的期望深度/距离实现，供当前损失及 TSDF 输出使用 |

双目深度接入前必须决定其语义是：相机 Z 深度、欧氏 ray range，还是视差。针孔中心区域三者容易混淆，鱼眼大视场边缘差异会非常明显。

### 5.3 内参分辨率

- 输入 `demo.py` 中的内参先随 resize 缩放；
- 进入 `MotionFilter` 后原地除以 8，存入 `DepthVideo`，对应 1/8 特征图；
- 送给 Gaussian 时在 `Hi2.call_gs()` 乘 8，恢复全分辨率内参；
- `intrinsics.npy` 同样乘 8 保存。

如果后续引入左右相机或虚拟相机，每一路 K 都必须遵守同样的分辨率约定。

### 5.4 关键帧索引与原始时间戳

- tracker 内部图节点使用连续 buffer index；
- `DepthVideo.tstamp` 保存它对应的原始帧序号；
- Gaussian `unique_kfIDs` 使用原始帧序号作为锚点；
- 回环修正传入 Gaussian 时，通过 `tstamp == unique_kfIDs` 把关键帧更新广播到每个 Gaussian。

后续双目中左右图应共享同一采集时间戳，但需要额外的 camera ID，不能简单把左右图都当成不同时间的普通关键帧。

---

## 6. 工程目录地图

### 6.1 顶层

| 路径 | 作用 | 阅读优先级 |
|---|---|---:|
| `demo.py` | 单目数据入口、主循环、轨迹保存 | 最高 |
| `hislam2/` | SLAM 和 Gaussian 主体 | 最高 |
| `config/*.yaml` | 跟踪、PGBA、Gaussian 优化参数 | 高 |
| `calib/*.txt` | 针孔内参与可选 OpenCV 畸变参数 | 高 |
| `src/` | DROID BA/PGBA/距离/JDSA 的 CUDA 后端 | 高，适配阶段必读 |
| `thirdparty/diff-gaussian-rasterization/` | 修改后的可微 RGB + 无偏 depth Gaussian rasterizer | 高，原生鱼眼必读 |
| `thirdparty/simple-knn/` | Gaussian 初始化尺度所需 KNN | 中 |
| `thirdparty/lietorch/` | `SE(3)` / `Sim(3)` 李群运算 | 中 |
| `scripts/` | 数据预处理、批量实验和重建评估 | 中 |
| `tsdf_integrate.py` | 最终渲染深度的 TSDF 网格融合 | 中 |
| `pretrained_models/` | DROID 与 OmniData 权重 | 中 |
| `outputs/` | 运行结果 | 低 |

### 6.2 `hislam2/` 主模块

| 文件 | 一句话职责 |
|---|---|
| `hi2.py` | 系统总调度器，串联所有阶段 |
| `motion_filter.py` | 候选关键帧筛选、DROID 特征和 OmniData 先验提取 |
| `depth_video.py` | 共享关键帧状态仓库和 CUDA 优化入口 |
| `factor_graph.py` | 活动/非活动重投影因子、局部 BA、PGBA、全局 BA 调度 |
| `track_frontend.py` | 初始化、滑窗图维护、局部 BA/JDSA、关键帧剔除 |
| `track_backend.py` | 离线低显存全局 BA |
| `pgo_buffer.py` | 回环搜索、相对位姿因子缓存、Sim(3) PGBA |
| `gs_backend.py` | Gaussian 初始化、在线 mapping、地图变形、离线联合优化 |
| `geom/pinhole.py` | Python 针孔投影/反投影及 Jacobian |
| `geom/projective_ops.py` | 位姿 + 深度的跨帧重投影 |
| `geom/ba.py` | Python BA、JDSA 和 Schur 系统构造 |
| `geom/chol.py` | Cholesky/Schur 求解器 |
| `modules/droid_net.py` | 特征网络、GRU update、置信度和上采样 mask |
| `modules/corr.py` | 常规与低显存相关体 |
| `midas/omnidata.py` | 深度/法线先验模型包装 |
| `util/trajectory_filler.py` | 非关键帧位姿补全及 post-keyframe 位姿求解 |

### 6.3 `hislam2/gaussian/`

| 路径 | 作用 |
|---|---|
| `scene/gaussian_model.py` | Gaussian 参数、点云初始化、Adam 参数组、增密和裁剪、PLY 保存 |
| `renderer/__init__.py` | Python rasterizer 接口 |
| `utils/camera_utils.py` | Gaussian 相机对象和可优化 pose/exposure 参数 |
| `utils/graphics_utils.py` | 针孔投影矩阵与 world/view 变换 |
| `utils/slam_utils.py` | 深度转点/法线、RGB/depth/normal loss、位姿更新 |
| `utils/eval_utils.py` | 最终关键帧/全帧渲染评估和结果保存 |
| `gui/` | Gaussian 可视化，不参与核心优化 |

---

## 7. 配置文件怎么读

以 `config/replica_config.yaml` 为例。

### 7.1 `Dataset`

- `pcd_downsample_init`：第一张关键帧生成点云时的随机下采样倍数；
- `pcd_downsample`：普通关键帧下采样倍数；
- `adaptive_pointsize`：是否按中值深度调整 Gaussian 初始尺度；
- `point_size`：KNN 距离到初始尺度的乘数；
- `scale_multiplier`：初始化时单目系统尺度归一化因子。

### 7.2 `Tracking.motion_filter`

- `init_thresh`：初始化阶段接受候选帧的光流阈值；
- `thresh`：正常阶段阈值；
- `skip_blur`：是否在缓存中优先选清晰帧。

### 7.3 `Tracking.frontend`

- `keyframe_thresh`：判断中间关键帧是否运动过小；
- `frontend_thresh`：允许添加共视边的帧距离阈值；
- `frontend_window`：局部图时间窗口；
- `frontend_radius`：强制相连的时间邻域；
- `frontend_nms`：边选择的非极大抑制半径；
- `mono_depth_alpha`：JDSA 中单目先验项强度。

### 7.4 `Tracking.backend / pgba`

- `backend_thresh/radius/nms`：离线全局 BA 的边搜索；
- `covis_thresh`：序列结束时补关键帧的覆盖阈值；
- `pgba.active`：是否启用在线回环；
- `pgba_thresh`：回环候选的重投影距离阈值。

### 7.5 `Training`

- `init_itr_num`：第一张图 Gaussian 初始化轮数；
- `gaussian_update_every/offset`：增密裁剪频率和相位；
- `gaussian_th`：裁剪 opacity 阈值；
- `window_size`：配置中存在，但当前 `process_track_data()` 实际窗口长度使用硬编码 10/11，未直接读取该值；
- `lambda_dnormal`：法线几何监督权重；
- `compensate_exposure`：离线是否优化曝光。

### 7.6 `opt_params`

这里是 Gaussian、相机位姿、曝光等 Adam 学习率和增密梯度阈值。`position_lr_max_steps` 同时决定离线联合优化总迭代数。

---

## 8. 论文与当前代码的差异和实现细节

后续适配必须以代码为准，再决定是否追随论文新版本。

| 主题 | 论文描述 | 当前代码 |
|---|---|---|
| 曝光补偿 | 每帧 `3×3 A + 3×1 b` | 每帧标量 `exp(a) * I + b`，RGB 三通道共享 |
| 点云下采样 | 论文实现细节写 `psi=32` | 第一帧 32，普通关键帧 YAML 默认 64 |
| opacity reset | 每 500 iterations | 初始化由配置 500；在线代码强制改成 501 |
| post-keyframe | 显式投影分析视野覆盖 | 代码用 `distance_covis()` + 相邻时间中点近似 |
| color 表示 | 直接 RGB | 代码用 0 阶 SH/DC 存储，效果等价于视角无关常量色 |
| depth loss | 论文写渲染 depth 与估计 depth 的 L1 | 代码实际对二者取倒数后做 L1 |
| mapper 并行 | 系统图容易理解为 continuous mapper | `GSBackEnd` 未 `start()`，当前实现同步调用 |
| 局部窗口 | 论文概念上的局部关键帧集合 | 代码列表更新方式可达到约 11 帧，且没有直接使用 YAML `window_size` |
| 法线反投影 | 应使用真实相机内参 | `depths_to_points()` 重新用 FoV 求焦距，并把主点固定在 `W/2,H/2`，忽略实际 `cx,cy` |
| 多帧内参 | `DepthVideo` 看起来逐帧保存 K | CUDA BA、frame distance、covis distance 多处只使用 `intrinsics[0]` |

这张表不是说实现错误，而是说明论文符号、最新论文版本和公开代码之间不能机械一一对应。

---

## 9. 面向“双目鱼眼 Gaussian SLAM”的适配边界

本节不提供代码修改，只给出架构判断。

## 9.1 当前代码中所有相机模型耦合点

### A. 数据输入层

- `demo.py:mono_stream()`：单图、单 K、OpenCV 普通畸变模型；
- 标定文本：只定义 `[fx,fy,cx,cy,...]`；
- resize/crop 后的内参更新。

### B. 跟踪几何层

- `hislam2/geom/pinhole.py`：Python 投影/反投影；
- `hislam2/geom/projective_ops.py`：跨帧重投影；
- `src/droid_kernels.cu`：真实高性能 BA/PGBA/JDSA/距离函数里的投影、反投影和 Jacobian；
- `DepthVideo`：只分配单相机 feature rig 维度 `[N,1,...]`；
- `DepthVideo.cuda_ba()` 等函数只传第一组内参；
- `MotionFilter` 和 `TrajectoryFiller` 默认只编码一幅图。

### C. 深度与先验层

- OmniData 是针对普通透视图像训练的；直接输入强畸变鱼眼图，尤其边缘区域，深度和法线先验的分布外问题较大；
- 2×2 JDSA 网格只能纠正平滑尺度畸变，不能纠正错误的鱼眼 ray 几何；
- 双目输出的米制 depth/disparity 与当前单目逆深度先验需要统一语义和置信度。

### D. Gaussian 初始化层

- Open3D `PinholeCameraIntrinsic` 反投影；
- `Camera` 只保存一组针孔 K 和 FoV；
- `slam_utils.depths_to_points()` 是针孔 ray。

### E. Gaussian 栅格化层

- 透视投影矩阵；
- 透视投影的 covariance Jacobian；
- 基于针孔图像平面的 tile/binning；
- 无偏深度的 ray-plane 推导；
- 相机 pose 的反向 Jacobian。

### F. 输出与网格层

- 最终逐帧渲染使用针孔相机；
- `tsdf_integrate.py` 使用 Open3D 针孔 TSDF；
- `intrinsics.npy` 只能表示一台针孔相机。

## 9.2 唯一适配路线：全链路原生鱼眼

项目直接在 Kannala-Brandt、Unified/Double Sphere 或实际标定对应的鱼眼模型上完成投影、BA、Gaussian splatting 和 TSDF。必须改造：

- Python 与 CUDA 的投影/反投影及解析 Jacobian；
- frame/covis distance；
- BA、PGBA、JDSA；
- 3D Gaussian 到鱼眼图像上的非线性 footprint；
- tile culling、无偏 depth、pose backward；
- 鱼眼深度转法线与 mesh fusion。

原生鱼眼不是“替换 `pinhole.py`”这么简单，而是跟踪、回环、Gaussian 和表面融合中的相机几何都要统一重新定义。

## 9.3 推荐的分阶段理解和验证顺序

后续按以下顺序建立可回归基线：

1. 固定当前单目工程结果，保存轨迹、Gaussian 和 mesh 基线；
2. 实现统一的原生鱼眼 `project/unproject/Jacobian/valid mask`；
3. 修改 DROID Python 与 CUDA 投影，先跑通原始左鱼眼时间跟踪；
4. 加入同一时刻左右鱼眼约束和固定外参，建立米制尺度；
5. 适配 JDSA、`Sim(3)/SE(3)` PGBA 和 full BA；
6. 实现原生鱼眼 Gaussian 初始化、forward、backward 和双目监督；
7. 最后完成原生鱼眼深度、轨迹和 TSDF 融合。

## 9.4 双目带来的算法选择

双目不仅是“多一张图”，它改变了当前系统的一些基本假设：

- 绝对尺度可观：初始化不再必须把平均深度归一化为 1；
- 可用 `SE(3)` 回环保持米制尺度，是否继续用 `Sim(3)` 取决于双目深度稳定性和前端是否真正使用双目约束；
- JDSA 可从“对齐单目先验”改造成“融合 stereo metric depth、单目 prior 和多视 BA depth”；
- 同时刻左右相机之间存在固定基线，应作为 rig constraint，而不是让 BA 独立优化两台相机位姿；
- 左右曝光、白平衡和鱼眼暗角不同，需要相机级甚至空间变化的光度模型；
- 左右图的遮挡不同，stereo depth 必须携带有效 mask/置信度，不能把所有像素作为硬真值送入 Gaussian loss。

---

## 10. 建议的源码阅读顺序

### 第一遍：只理解控制流

1. `demo.py`
2. `hislam2/hi2.py`
3. `hislam2/motion_filter.py`
4. `hislam2/track_frontend.py`
5. `hislam2/gs_backend.py`

目标：能口述“一帧进入后发生什么”和“序列结束后发生什么”。

### 第二遍：理解跟踪数学

1. `hislam2/depth_video.py`
2. `hislam2/factor_graph.py`
3. `hislam2/geom/projective_ops.py`
4. `hislam2/geom/pinhole.py`
5. `hislam2/geom/ba.py`
6. `hislam2/pgo_buffer.py`
7. `src/droid_kernels.cu`

目标：能说清 `target/weight`、pose、inverse depth、JDSA scale grid、inactive edge 和 PGBA 之间的关系。

### 第三遍：理解 Gaussian 地图

1. `hislam2/gaussian/scene/gaussian_model.py`
2. `hislam2/gaussian/renderer/__init__.py`
3. `hislam2/gaussian/utils/camera_utils.py`
4. `hislam2/gaussian/utils/slam_utils.py`
5. `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
6. `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
7. `tsdf_integrate.py`

目标：能说清 Gaussian 如何初始化、如何优化、如何跟着关键帧变形，以及 rendered depth 如何变成 mesh。

### 第四遍：只为双目鱼眼做接口盘点

沿下面五条数据逐层追踪：

1. `image_left/image_right`；
2. `camera model + intrinsics + distortion + T_right_left`；
3. `depth/disparity/range + confidence`；
4. `rig pose + per-camera pose`；
5. `render rays + Gaussian projection + TSDF rays`。

---

## 11. 读完后应能回答的检查问题

如果以下问题都能回答，就已经具备开始设计适配方案的基础：

1. DROID 网络输出的是最终位姿，还是稠密匹配修正和置信度？
2. 为什么系统同时保存 `disps`、`disps_up`、`disps_prior` 和 `dscales`？
3. JDSA 为什么和普通 BA 交替，而不是把 pose/depth/scale 一次联合求解？
4. 为什么在线回环使用 `Sim(3)`，离线 full BA 又回到 `SE(3)`？
5. inactive reprojection edge 如何变成 PGBA 的相对位姿因子？
6. 一个 Gaussian 如何知道回环后应该使用哪个关键帧的位姿修正？
7. Gaussian mapping 的 RGB、depth、normal、scale 四类损失分别约束什么？
8. 为什么无偏深度不是简单输出 Gaussian 中心 Z？
9. 为什么 `GSBackEnd` 当前并没有真正作为独立 mapping 进程运行？
10. 为什么原生鱼眼适配至少同时涉及 DROID CUDA 和 Gaussian rasterizer CUDA？
11. 双目深度究竟表示 Z-depth、ray range 还是 disparity，当前代码希望得到哪一种？
12. 原始左右鱼眼如何共享同一个 rig pose，同时通过固定外参得到各自 camera pose？

---

## 12. 最简总结

把 HI-SLAM2 记成四个状态和三个闭环即可：

四个状态：

- `pose`：关键帧世界到相机位姿；
- `disparity`：每关键帧 1/8 分辨率稠密逆深度；
- `prior scale grid`：每帧 2×2 的单目先验尺度；
- `Gaussian map`：带关键帧锚点的显式场景。

三个闭环：

1. recurrent correspondence 与局部 BA/JDSA 的闭环；
2. `Sim(3)` PGBA 与锚定 Gaussian 变形的闭环；
3. 离线可微渲染与 Gaussian/pose 联合优化的闭环。

双目鱼眼适配的真正工作不是“怎样读两张图”，而是在原生鱼眼域中重新维持这三个闭环：DROID 的投影与 BA、`Sim(3)/SE(3)` 全局一致性、Gaussian 的前向/反向可微渲染。当前项目已经选定原生鱼眼路线，具体执行顺序见配套的新手源码阅读路线。
