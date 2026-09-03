# HI-SLAM2 到原生双目鱼眼 Gaussian SLAM：真值深度优先的分阶段适配计划

> **版本说明：**本文保留为早期技术思路。后续请以 [双目鱼眼高斯SLAM_定位与建图详细实施计划.md](./双目鱼眼高斯SLAM_定位与建图详细实施计划.md) 为主执行计划，新版已将定位和 Gaussian 建图分开，并给出了逐文件改动和定量验收标准。

> 适用于当前数据条件：同步左右原始鱼眼 RGB、左右相机内参、固定双目外参、真值 pose 和真值深度图。左右相机视场角约为 200°。
>
> 实施原则：始终在原始鱼眼像素域工作，不生成中间投影图。先用真值深度降低联调难度，打通整条 pipeline 后，再用双目鱼眼深度网络替换真值深度提供器。
>
> 本文是后续实施的唯一执行清单，不表示当前代码已经完成这些改造。每次只执行一个小步骤，验收后再继续。

---

## 1. 先把最终目标说清楚

当前工程的主要前提是“单目 + 针孔 + 单目深度先验 + 针孔 Gaussian rasterizer”。目标工程要变成：

```text
同步左/右原始鱼眼 RGB
  + 两套鱼眼标定
  + 固定双目外参
  + DepthProvider（第一轮返回真值 range，最终返回网络预测 range）
                    │
                    ├─> 原生鱼眼时间跟踪
                    ├─> 原生鱼眼左右重投影约束
                    ├─> 米制尺度局部 BA / 回环 / 全局 BA
                    └─> 原生鱼眼 Gaussian 初始化、渲染和建图
                                      │
                                      └─> rig 轨迹、左右相机轨迹、Gaussian、深度和 mesh
```

真值 pose 有两个用途：

1. 在最早的 oracle mapping 模式中直接作为输入，用来单独检验鱼眼 Gaussian 建图链路；
2. 跟踪开始后只作为评价真值，不再喂给在线 SLAM。

真值深度的用途是：先把深度网络完全从问题中拿掉，让我们能先确认数据、鱼眼几何、跟踪、BA、Gaussian 渲染和建图是正确的。

---

## 2. 200° 鱼眼必须提前确定的技术约定

### 2.1 内部深度统一用 ray range

200° 视场中会出现与光轴夹角大于 90° 的射线。对这些像素，相机 Z 坐标可以接近 0 甚至为负，因此不能把全画面深度继续理解为针孔 Z-depth。

本计划统一使用：

```text
ray(u,v)       : 像素对应的单位方向，形状 [3]
range(u,v)     : 沿 ray 从相机中心到三维点的欧氏距离，单位 m
inv_range(u,v) : 1 / range，用于代替当前 video.disps 的几何含义
X_camera       : range * ray
```

数据中的“深度图”不能仅凭文件名判断含义。阶段 1 必须验证它是 range、Z-depth，还是其他定义；如果不是 range，只在数据边界转换一次，系统内部不再混用。

### 2.2 位姿和外参统一记号

全文用 `T_ab` 表示“把 b 坐标系中的点变换到 a 坐标系”：

```text
T_Rw : world -> rig（用大写 R 表示 rig）
T_wR : rig -> world
T_lR : rig -> left（如果左目就是 rig 原点，它为单位变换）
T_rR : rig -> right（小写 r 表示 right camera）
T_cw = T_cR * T_Rw
```

第一版建议以左目为 rig 参考系，因此只优化一个 rig pose，右目 pose 由固定外参计算。任何时候都不允许左右位姿被当成两个独立变量优化。

### 2.3 鱼眼模型由标定文件决定

不先入为主地假定一定是 Kannala–Brandt、Double Sphere 或 Unified/MEI。第一阶段从标定文件和生成它的标定工具确认模型，然后 Python、CUDA、Gaussian rasterizer 和深度网络共用同一套公式和参数顺序。

### 2.4 大视场带来的深度网络收益要用实验确认

左右 200° 图像能提供更宽的场景上下文，双目基线还能提供米制几何约束，这是相比单目深度的重要优势。但最终精度还取决于网络是否理解鱼眼投影、左右重叠区、遮挡和 range 定义。因此阶段 9 会用真值深度做定量消融，不只看主观效果。

---

## 3. 三种运行模式：每次只撤掉一个真值条件

| 模式 | pose 来源 | depth/range 来源 | 主要检验什么 | 何时使用 |
|---|---|---|---|---|
| `oracle_mapping` | 真值 pose | 真值 range | 数据、鱼眼几何、Gaussian 初始化/渲染/建图 | 阶段 3 |
| `slam_gt_depth` | SLAM 估计 pose | 真值 range | 跟踪、双目约束、BA、回环和完整 pipeline | 阶段 4–8 |
| `slam_pred_depth` | SLAM 估计 pose | 双目鱼眼网络 | 最终系统 | 阶段 9–10 |

这三种模式必须走同一条主程序，不要为真值另写一套后端。差别只能出现在 `PoseProvider` 和 `DepthProvider` 的输入边界。

建议的统一深度接口：

```python
DepthPacket = {
    "range": ...,       # [B, 2, H, W]，m
    "inv_range": ...,   # [B, 2, H, W]
    "valid": ...,       # bool mask
    "confidence": ...,  # [0,1]，真值模式可由 valid 生成
    "source": "gt" or "network",
}
```

如果当前数据只有左目深度，接口仍然保留相机维度；右目项用 invalid mask 表示，不伪造右目真值。

---

## 4. 总体实施顺序

```text
阶段 0  冻结原工程基线和实验规则
   ↓
阶段 1  打通双目 RGB / 标定 / pose / 真值深度的数据边界
   ↓
阶段 2  实现并单测原生鱼眼 CameraModel
   ↓
阶段 3  用真值 pose + 真值 range 打通 oracle Gaussian mapping
   ↓
阶段 4  用左目时间边打通原生鱼眼在线跟踪，深度仍用真值
   ↓
阶段 5  加入同时刻左右边，建立米制双目 BA
   ↓
阶段 6  将估计轨迹 + 真值 range 接入在线 Gaussian mapping
   ↓
阶段 7  适配回环、PGBA/全局 BA 和米制尺度管理
   ↓
阶段 8  适配离线 refinement、轨迹、评价和表面输出
   ↓
阶段 9  训练/接入双目鱼眼 range 网络，但不改后端接口
   ↓
阶段 10 全系统联调、消融和最终验收
```

---

# 阶段 0：冻结基线和实验规则

## 0.1 小步骤

1. 用现有数据跑通一次当前单目工程；
2. 保存命令、配置、轨迹、Gaussian PLY、渲染 RGB/深度和日志；
3. 固定一段 20–50 帧的小数据作为所有单元测试的 `tiny sequence`；
4. 再固定一段有快速旋转、近物、远物、左右遮挡和回环的 `validation sequence`；
5. 规定每个阶段都要保存数值指标和可视化，不以“程序没崩”作为验收。

## 0.2 阅读文件

- `README.md`：安装、运行和输出；
- `demo.py`：现有单目输入、主循环和轨迹保存；
- `hislam2/hi2.py`：跟踪、回环和 Gaussian 后端的调度；
- `config/replica_config.yaml`：学习配置层级。

## 0.3 本阶段语法

- `argparse`：命令行参数如何变成 `args.xxx`；
- YAML 字典：配置项如何被 Python 访问；
- Python 主入口 `if __name__ == "__main__":`；
- Tensor 形状的记录方式，例如 `[B,C,H,W]`。

## 0.4 验收

- 能画出 `demo.py -> Hi2.track() -> MotionFilter -> TrackFrontend -> GSBackend`；
- 有一套可重复的原工程结果；
- 新适配中出现回归时可以与它比较。

---

# 阶段 1：双目鱼眼数据边界

本阶段只读数据和做一致性检查，不运行 BA 和 Gaussian 优化。

## 1.1 小步骤 1：给每帧建立统一数据包

目标数据形状：

```text
tstamp             scalar
rgb                 [2, 3, H, W]      # 0=左，1=右
camera_params       [2, P]
T_camera_rig        [2, 4, 4]
gt_T_rig_world      [4, 4]
gt_range            [2, H, W]         # 右目无真值时由 mask 声明
gt_range_valid      [2, H, W]
```

后续建议新增的数据结构名称是 `StereoFisheyeFrame`。它的作用不是复制图像，而是防止队列中传递的 tuple 越来越长、下标越来越难理解。

## 1.2 小步骤 2：验证时间同步

- 左右 RGB 必须一一对应；
- pose 时间戳必须对应 rig，而不是未说明的某个相机；
- 深度必须与对应 RGB 处于同一像素采样下；
- 丢帧时整个 stereo pair 丢弃，不拼接不同时刻的左右图。

## 1.3 小步骤 3：验证标定

1. 写出左右鱼眼模型名称和参数顺序；
2. 确认标定分辨率与 RGB 分辨率相同；
3. 确认外参方向，使用已知三维点检查 `T_right_left`；
4. 检查基线长度与物理相机一致，单位统一为 m；
5. 确认 200° 是水平、垂直还是对角 FoV，不把它当成计算投影的唯一参数。

## 1.4 小步骤 4：验证 pose

- 确认文件是 `T_world_rig` 还是 `T_rig_world`；
- 确认四元数顺序是 `xyzw` 还是 `wxyz`；
- 确认坐标轴方向；
- 将相邻两帧的平移和旋转量打印出来，排除单位和方向错误；
- 用左右固定外参推出右目 GT pose，确认基线在世界中始终保持刚性。

## 1.5 小步骤 5：确定深度图定义

- 确认数值单位和 invalid 值；
- 从画面中心、边缘和超过 180° 的区域分别选点；
- 用 `X = value * ray` 生成三维点并与 pose/场景比较；
- 如果是 Z-depth，必须通过射线几何转为 range，并对 `|ray_z|` 过小或方向不可定义的区域设 invalid；
- 画出反投影点云，确认边缘没有被错误拉伸成长针。

## 1.6 本阶段阅读/以后修改的文件

- `demo.py:mono_stream()`：现在只产生单张 RGB 和 4 个针孔内参；
- `scripts/preprocess_owndata.py`：现有自有数据预处理逻辑；
- `hislam2/hi2.py:track()`：主系统的帧输入边界；
- `hislam2/motion_filter.py:track()`：候选关键帧接收的实际数据；
- `hislam2/depth_video.py`：当前只有单目存储和 `[fx,fy,cx,cy]`；
- 后续新增的数据集 adapter 和配置文件（文件名在实施该阶段时确定）。

## 1.7 本阶段语法

- `dataclass`：用有名字段代替过长 tuple；
- `torch.Tensor.shape`：明确批次、相机、通道和像素维；
- `dtype` 和单位：RGB、深度、mask 分别使用什么类型；
- `bool mask`：不用 0 深度冒充真实观测；
- 4×4 齐次变换乘法和求逆。

## 1.8 验收

- 随机抽 100 个时刻，左右 RGB、深度、pose 无错位；
- 已写明鱼眼模型、参数顺序、外参方向、pose 方向、深度定义和单位；
- 可以从任意帧得到规范化的 `StereoFisheyeFrame`；
- 仍未开始修改 BA 和渲染器。

---

# 阶段 2：原生鱼眼 CameraModel

这是所有后续模块的几何基础。先完成 Python/PyTorch 参考版，不先进 CUDA。

## 2.1 小步骤

1. 定义 `project(X_camera, params) -> uv, valid`；
2. 定义 `unproject(uv, params) -> unit_ray, valid`；
3. 定义 `range_to_point(range, ray)` 和 `point_to_range(X)`；
4. 定义 `J_project_X`，即投影坐标对三维点的 Jacobian；
5. 为固定分辨率生成 `ray_lut[H,W,3]` 和 `valid_lut[H,W]`；
6. 明确图像 resize 后哪些内参需要缩放，不在多个文件里分别猜测；
7. 用双精度 CPU 参考结果作为后续 CUDA 的真值。

## 2.2 必须做的测试

- 像素往返：`uv -> ray -> project(ray)`；
- 三维往返：`X -> uv + range -> X_reconstructed`；
- 左右外参往返：`X_l -> X_r -> X_l`；
- 有限差分 Jacobian 对比；
- 画面中心、边缘、180° 附近和超过 180° 区域的边界测试；
- 与标定工具或数据生成器的参考投影对比。

## 2.3 阅读/以后修改的文件

- `hislam2/geom/pinhole.py`：当前 `iproj_pinhole()` 和 `proj_pinhole()`；
- `hislam2/geom/projective_ops.py`：两帧之间的反投影、位姿变换和重投影；
- `hislam2/gaussian/utils/slam_utils.py:depths_to_points()`：当前针孔深度转点；
- 后续新增的 `CameraModel` 参考实现和单元测试。

## 2.4 本阶段语法

- 类、`self` 和统一方法接口；
- Tensor broadcasting，例如 `[H,W,1] * [H,W,3]`；
- `torch.where` 与 bool mask；
- `torch.autograd` 自动微分与有限差分；
- 数值稳定：`eps`、安全除法、反三角函数输入范围。

## 2.5 验收

- 有一个与 SLAM 无关的 CameraModel 测试程序；
- 像素往返误差达到预先设定的亚像素阈值；
- 解析/autograd Jacobian 与有限差一致；
- 原始鱼眼深度反投影点云形状正确。

---

# 阶段 3：oracle Gaussian mapping（真值 pose + 真值 range）

这一阶段先隔离跟踪问题，验证“鱼眼 RGB + range + pose -> Gaussian -> 鱼眼 RGB/range”。它是第一个小型端到端闭环。

## 3.1 小步骤 1：反投影并初始化 Gaussian

1. 用 `X_c = range * ray_lut` 从深度图生成相机坐标点；
2. 用 GT pose 变到世界坐标；
3. 根据 valid mask、深度边界和采样间隔筛点；
4. 从原始鱼眼 RGB 取得初始颜色；
5. 根据相邻射线和 range 估计初始 Gaussian scale，不再调用 Open3D 针孔内参点云路径。

## 3.2 小步骤 2：原生鱼眼 rasterizer forward

1. 将 Gaussian 中心变换到相机坐标；
2. 通过 CameraModel 投影到原始鱼眼像素；
3. 用 `J_project_X` 将三维协方差投影成二维协方差；
4. 为可见 Gaussian 计算原始鱼眼画面中的 bounding box 和 tile；
5. 用 range 而不是 Z 完成深度排序和输出；
6. 对边缘及投影奇异区域使用显式 valid mask；
7. 先完成 forward 数值正确性，再开始 backward。

## 3.3 小步骤 3：backward 和建图优化

- 检查颜色、不透明度、三维中心、scale、rotation 和 pose 的梯度；
- 用有限差分检查至少一类三维中心梯度和 pose 梯度；
- 仅在单帧/少量帧上运行 mapping loss；
- 先固定 GT pose，只优化 Gaussian；
- 用左右两个视图共享同一份 Gaussian，pose 由 rig pose 和固定外参组成。

## 3.4 阅读/以后修改的文件

- `hislam2/gaussian/scene/gaussian_model.py:create_pcd_from_image_and_depth()`；
- `hislam2/gaussian/utils/slam_utils.py:depths_to_points()` 和 `depth_to_normal()`；
- `hislam2/gaussian/utils/camera_utils.py:Camera`；
- `hislam2/gaussian/renderer/__init__.py:render()`；
- `hislam2/gs_backend.py`；
- `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`；
- `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`；
- `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`；
- `thirdparty/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`。

## 3.5 本阶段语法

- `nn.Module` 和 `forward()`；
- PyTorch autograd 中叶子 Tensor、`.grad` 和 `.backward()`；
- CUDA kernel 的 thread index 和内存下标；
- 协方差矩阵 `J * Sigma * J^T`；
- `requires_grad`、`detach()` 和 `no_grad()` 的区别。

## 3.6 验收

- 单点、单 Gaussian、平面和少量真实帧的 forward 正确；
- 渲染的 RGB/range 与 GT 在原始鱼眼像素上对齐；
- 优化 loss 能下降，梯度检查通过；
- `oracle_mapping` 可以完成 tiny sequence 并保存 Gaussian。

---

# 阶段 4：左目原生鱼眼在线跟踪（真值 range）

这一阶段只使用左目的时间边，目标是先证明位姿跟踪中的鱼眼投影、Jacobian 和 BA 正确。深度使用真值 range，不调用深度估计器。

## 4.1 先按顺序阅读

1. `demo.py:mono_stream()` 和主循环；
2. `hislam2/hi2.py:track()`；
3. `hislam2/motion_filter.py:track()`；
4. `hislam2/track_frontend.py`；
5. `hislam2/factor_graph.py:add_factors()` 和 `update()`；
6. `hislam2/depth_video.py:append()/reproject()/distance()/cuda_ba()`；
7. `hislam2/geom/projective_ops.py`；
8. `src/droid_kernels.cu`；
9. `src/droid.cpp` 和 `setup.py`，理解 Python 如何调用 CUDA 扩展。

## 4.2 小步骤 1：让真值 range 进入 DepthVideo

- `DepthProvider=gt` 返回全分辨率 range/inv-range/mask；
- 得到 1/8 分辨率 inv-range 时使用对无效值友好的下采样；
- 区分 `measurement` 和 `state`：真值 range 是观测，优化中的 inv-range 是状态；
- 第一个小测试可以完全固定 inv-range，只优化 pose；
- 第二个小测试允许 inv-range 小幅优化，但通过 GT 残差约束其不漂移。

## 4.3 小步骤 2：替换 Python 时间重投影

```text
帧 i 像素 + range_i
  -> CameraModel_i.unproject
  -> X_i
  -> T_j_i * X_i
  -> CameraModel_j.project
  -> 帧 j 原始鱼眼像素
```

先用 GT pose 对比两帧重投影，再让跟踪优化 pose。有效 mask 至少包含：源深度有效、三维点可投影、目标像素在鱼眼有效圆内、数值有限。

## 4.4 小步骤 3：替换 DROID CUDA 几何

`src/droid_kernels.cu` 中不只有一个投影函数。需要逐个搜索并对照 Python 参考版替换：

- `proj` / `iproj`；
- BA 的 pose Jacobian 和 inv-range Jacobian；
- `frame_distance`；
- `covis_distance`；
- PGBA 使用的重投影；
- JDSA 中的 `proj_trans`；
- CUDA 端内参参数数量和内存布局。

每改一个 kernel，都用相同输入和 Python 参考版对比，不要一次性改完再查错。

## 4.5 小步骤 4：分层恢复在线跟踪

1. GT pose + GT range：只看重投影误差；
2. 轻微扰动 pose + 固定 GT range：BA 应收敛回 GT；
3. 两帧左目跟踪；
4. 20–50 帧左目局部 BA；
5. 允许 inv-range 作为状态优化，但保留 GT range loss；
6. 逐步降低 GT range loss 权重，观察几何是否稳定。

## 4.6 本阶段语法

- Python 切片和高级索引 `tensor[ii, jj]`；
- `@staticmethod` 与普通方法；
- C++/CUDA 指针、`const` 和 kernel launch；
- Jacobian 链式法则；
- `motion_only=True` 为什么意味着只更新 pose。

## 4.7 验收

- CUDA 重投影和 Python 参考版数值一致；
- pose 扰动实验能回到 GT 邻域；
- tiny sequence 能完成左目在线跟踪；
- GT pose 仅用来计算 ATE/RPE，不再作为跟踪输入；
- 还未引入右目 factor，不把这一阶段的尺度稳定性当成最终结果。

---

# 阶段 5：原生双目 factor 和米制 BA

本阶段把同一时刻的左右图像加入因子图。不使用水平视差公式，而是通过两个 CameraModel 和固定 `T_right_left` 完成射线重投影。

## 5.1 小步骤 1：扩展存储和 feature

- RGB、feature map、context 和 camera params 增加左右相机维；
- 检查 `DepthVideo.fmaps` 现有 rig 维度为 1 的假设；
- 左右图使用共享权重的 feature encoder；
- 不复制两份 rig pose；
- 引入 edge type：`TEMPORAL_LEFT`、`STEREO_LR`，必要时再加 `TEMPORAL_RIGHT`。

## 5.2 小步骤 2：左到右重投影

```text
left uv + left range
  -> left ray / X_left
  -> T_right_left * X_left
  -> right CameraModel.project
  -> right uv
```

相反方向也要实现，用于双向一致性和遮挡检查。双目边上 `i == j` 不代表位姿是单位变换；它表示时间相同，左右间仍然有固定外参。

## 5.3 小步骤 3：先做几何参考三角化

- 从左右像素反投影得到两条空间射线；
- 用已知外参把射线表示在同一坐标系；
- 求两射线最近点并得到 range；
- 过滤小夹角、负 range、过大残差和遮挡；
- 与 GT range 对比，将它作为后续双目 factor 的 CPU 参考。

## 5.4 小步骤 4：加入 stereo factor

- `FactorGraph.add_factors()` 显式接收 edge type 和 source/target camera id；
- correlation 从左 feature 查找右 feature；
- 目标坐标是右目原始鱼眼像素；
- BA 优化 rig pose 和 inv-range，双目外参默认固定；
- 第一版建议以“左目时间边 + 同时刻左右边”为主，右目时间边作为后续消融项。

## 5.5 小步骤 5：处理尺度

- 双目基线已用 m 表示，inv-range 也必须为 `1/m`；
- `DepthVideo.normalize()` 在 stereo metric 模式下禁用；
- JDSA 的 2×2 scale grid 不能无条件修改双目真值尺度；
- GT range 还可以作为调试 loss，但要区分“尺度由 stereo 几何建立”和“真值监督帮助收敛”；
- 使用 GT pose/range 检查基线尺度误差，然后将 GT range loss 降到 0 做消融。

## 5.6 阅读/以后修改的文件

- `hislam2/depth_video.py`；
- `hislam2/factor_graph.py:add_factors()/update()`；
- `hislam2/modules/extractor.py`；
- `hislam2/modules/corr.py`；
- `hislam2/modules/droid_net.py`；
- `hislam2/geom/projective_ops.py`；
- `hislam2/geom/ba.py:BA()/JDSA()`；
- `src/droid_kernels.cu`。

## 5.7 本阶段语法

- `enum` 或有名常量表示 edge type；
- `torch.stack` 与 `torch.cat` 的区别；
- 使用 camera id 做 Tensor 高级索引；
- 固定参数与优化变量；
- robust loss、confidence weight 和 occlusion mask。

## 5.8 验收

- GT pose/range 下，左右重投影与真实对应对齐；
- CPU 三角化 range 与 GT range 的误差分布可解释；
- stereo factor 能在 pose/range 扰动后降低残差；
- 去掉 GT range loss 后，系统仍保持基本米制尺度；
- `normalize()` 不会再改变 stereo metric 数据。

---

# 阶段 6：估计轨迹 + 真值 range 的在线 Gaussian mapping

阶段 3 已经证明 mapping 在 GT pose 下可用，阶段 4–5 已经证明跟踪可用。现在将两者连起来，但深度仍来自 `GroundTruthDepthProvider`。

## 6.1 小步骤

1. `Hi2.call_gs()` 不再通过 `1/disps_up` 默认产生针孔 depth，而是显式传递 range 及其 mask/source；
2. 同时传递 rig pose、左右固定外参和两套 CameraModel 参数；
3. 左右视图初始化同一份 Gaussian map；
4. 建图 loss 分开记录 `rgb_left/rgb_right/range_left/range_right/normal/regularization`；
5. 右目没有 GT range 时，右目仍可用 RGB loss，但不对 invalid 深度计算 range loss；
6. densification 和 pruning 的可见性统计合并左右视图；
7. 将 GT pose 只用于评价，输入 Gaussian 的是 SLAM 估计 rig pose。

## 6.2 阅读/以后修改的文件

- `hislam2/hi2.py:call_gs()`；
- `hislam2/gs_backend.py`；
- `hislam2/gaussian/utils/camera_utils.py`；
- `hislam2/gaussian/utils/slam_utils.py`；
- `hislam2/gaussian/scene/gaussian_model.py`；
- `hislam2/gaussian/renderer/__init__.py`。

## 6.3 本阶段语法

- producer/consumer 队列和数据所有权；
- Python dict 字段传递；
- CPU/GPU 间 `.to(device)` 和同步；
- loss 加权求和与分项日志；
- rig pose 如何派生 left/right camera pose。

## 6.4 验收

- `slam_gt_depth` 模式能完成 tiny sequence；
- 左右原始鱼眼渲染均与输入对齐；
- SLAM pose 与 GT pose 的差异会在渲染误差中可视化；
- 可保存 Gaussian PLY、rig/左/右轨迹和左右渲染结果；
- 深度网络仍然完全未接入。

---

# 阶段 7：回环、PGBA 和全局优化

## 7.1 小步骤

1. 检查 loop candidate 距离中的所有投影已替换为 CameraModel；
2. 检查 `frame_distance` 和 `covis_distance` 对鱼眼有效区域正确归一化；
3. 先保留 Sim(3) 作为尺度诊断，记录每次回环的 scale update；
4. 如果 stereo metric 模式稳定，Sim(3) scale 应长期接近 1；
5. 当该现象被实验确认后，再评估将 stereo 模式的全局优化收紧为 SE(3)；
6. 禁止 PGBA scale update 直接缩放真值/stereo range 观测；
7. 回环后只更新位姿状态、可优化的 range state 和 Gaussian 世界坐标，原始观测保持不变。

## 7.2 阅读/以后修改的文件

- `hislam2/loop_detector.py`；
- `hislam2/pgo_buffer.py`；
- `hislam2/factor_graph.py:update_pgba()`；
- `hislam2/depth_video.py:cuda_pgba()`；
- `src/droid_kernels.cu` 的 PGBA/frame/covis 几何；
- `hislam2/hi2.py` 中的 pose/scale 更新传递。

## 7.3 本阶段语法

- SE(3) 与 Sim(3) 的状态维度；
- graph edge 的 `ii/jj`；
- 稀疏图优化中局部和全局变量；
- in-place update 与原始观测不可变性。

## 7.4 验收

- 开启回环后 validation sequence 不崩溃；
- 闭环误差下降，不产生明显的米制尺度跳变；
- 记录的 Sim(3) scale 足以判断是否可转为 SE(3)；
- 回环前后 Gaussian 地图与轨迹同步更新。

---

# 阶段 8：离线 refinement、输出和评价

## 8.1 小步骤

1. `TrajectoryFiller` 和离线 low-memory BA 使用同一鱼眼几何；
2. 离线 Gaussian joint refinement 中，左右视图共享 rig pose delta，外参固定；
3. 分别输出 rig、left camera 和 right camera 轨迹，文件头写明 pose 方向和四元数顺序；
4. 保存左右原始鱼眼 RGB/range render 及 valid mask；
5. 用 ATE/RPE 评价轨迹，stereo metric 主结果不使用 Sim(3) 对齐隐藏尺度误差；
6. 评价 range MAE/RMSE/AbsRel 和有效覆盖率；
7. 评价 RGB PSNR/SSIM/LPIPS，左右分开报告；
8. 表面融合使用原生鱼眼 ray + range，不再调用 `tsdf_integrate.py` 当前的针孔 Open3D 入口。

## 8.2 阅读/以后修改的文件

- `hislam2/trajectory_filler.py`；
- `hislam2/hi2.py:terminate()`；
- `hislam2/gs_backend.py`；
- `hislam2/gaussian/utils/eval_utils.py`；
- `hislam2/gaussian/utils/camera_utils.py`；
- `demo.py:save_trajectory()`；
- `tsdf_integrate.py`。

## 8.3 本阶段语法

- 轨迹时间戳对齐；
- pose matrix 与 quaternion 互换；
- 指标的 valid mask 归一化；
- 配置化实验和结果目录组织。

## 8.4 验收

- `slam_gt_depth` 在完整 validation sequence 上跑通；
- 跟踪、回环、Gaussian mapping、refinement、评价和输出全部完成；
- 结果中不存在未说明的坐标系或深度定义；
- 到这里才可以说“真值深度版双目鱼眼 Gaussian SLAM pipeline 已打通”。

---

# 阶段 9：双目鱼眼深度网络

深度网络不与前面几何改造同时开始。它只需实现与 `GroundTruthDepthProvider` 相同的输出协议，后端不应因此重写。

## 9.1 小步骤 1：定义网络输入输出

```text
输入：left RGB + right RGB + left/right CameraModel 参数 + T_right_left
输出：left/right range 或 inv-range + valid/confidence
```

网络内部可以预测 inv-range 以改善近距离分辨率，但 API 同时返回 range 和 inv-range，且单位始终是 m。

## 9.2 小步骤 2：建立双目鱼眼数据加载器

- 左右增强必须保持几何一致；
- 任何改变像素采样的增强必须同步更新 CameraModel 参数和 range mask；
- 保留与光轴夹角 `theta` 接近 90° 和 `theta > 90°` 的后向边缘射线，单独报告这些区域的误差；
- 区分 stereo overlap 和 non-overlap 区域；
- 训练/验证/测试按场景划分，避免相邻帧泄漏。

## 9.3 小步骤 3：先做最小网络基线

1. 使用左右 RGB 预测左目 inv-range；
2. 主监督使用 GT range/inv-range；
3. 网络必须在原始鱼眼像素上输出；
4. 先在离线测试集上评价，不马上接 SLAM；
5. 与“单目鱼眼输入”做对照，定量确认双目和 200° 上下文的收益。

## 9.4 小步骤 4：增加几何损失

- GT range/inv-range 监督；
- 左右原生鱼眼重投影一致性；
- 遮挡感知的 photometric/feature loss；
- 左右 range 三维一致性；
- 深度边界感知平滑项；
- confidence 标定损失，使 SLAM 知道哪些像素不可信。

## 9.5 小步骤 5：用统一接口接入 SLAM

1. 将 `DepthProvider=gt` 替换为 `DepthProvider=stereo_network`；
2. 第一轮固定网络，SLAM 不反向更新它；
3. 记录网络 confidence 与实际 range 误差的关系；
4. 通过 confidence 调整 depth factor/mapping loss 权重；
5. 先不做端到端联合训练，等推理版稳定后再评估是否值得。

## 9.6 阅读/以后修改的文件

- `hislam2/motion_filter.py`：当前 OmniData depth/normal 先验的入口；
- `hislam2/modules/droid_net.py`：跟踪网络与几何的边界；
- `hislam2/geom/ba.py:JDSA()`；
- `hislam2/depth_video.py` 的 `disps_prior/dscales/mono_depth_alpha`；
- 后续新增的 `DepthProvider` 、双目鱼眼深度网络、dataset 和训练配置。

## 9.7 本阶段语法

- PyTorch `Dataset/DataLoader`；
- `train()/eval()` 和 `torch.no_grad()`；
- optimizer、learning rate、checkpoint；
- supervised loss、geometry loss 和 mask；
- mixed precision 与显存管理。

## 9.8 验收

- 网络离线输出的 range 单位、shape、mask 与 GT provider 完全一致；
- 已定量比较单目鱼眼和双目鱼眼深度；
- 分别报告 overlap/non-overlap、中心/边缘、`ray_z >= 0`/`ray_z < 0` 后向边缘区域指标；
- 只改一个配置项就能在 GT provider 和 network provider 之间切换；
- `slam_pred_depth` 能完成 tiny sequence。

---

# 阶段 10：完整联调和消融

## 10.1 必须比较的实验

| 编号 | 实验 | 用途 |
|---|---|---|
| A | GT pose + GT range | mapping 理想上限 |
| B | 估计 pose + GT range | 隔离跟踪/回环误差 |
| C | 估计 pose + 双目网络 range | 最终系统 |
| D | 只有左目时间边 | 双目 factor 消融 |
| E | 左目时间边 + stereo edge | 验证米制尺度收益 |
| F | 单目鱼眼 depth net | 深度网络对照 |
| G | 双目鱼眼 depth net | 验证双目和宽 FoV 收益 |
| H | 只左目 mapping | 右目 Gaussian 监督消融 |
| I | 左右 mapping | 证明双目覆盖的地图收益 |

## 10.2 最终指标

- 轨迹：ATE、RPE translation、RPE rotation、尺度漂移；
- 深度：range MAE/RMSE/AbsRel、有效覆盖率、confidence calibration；
- 渲染：左/右 PSNR、SSIM、LPIPS；
- 地图：Gaussian 数量、表面几何误差、孔洞/浮点；
- 系统：FPS、延迟、GPU 显存、网络深度耗时、rasterizer 耗时；
- 分区：左/右、overlap/non-overlap、中心/边缘、`ray_z >= 0`/`ray_z < 0` 后向边缘区域。

## 10.3 最终验收

- 输入只需左右原始鱼眼 RGB、标定和必要的运行配置；
- GT pose/depth 可以完全从系统输入中关闭，仅用于离线评价；
- 跟踪、BA、回环、Gaussian 渲染、建图和表面输出都使用同一 CameraModel；
- 左右视图共享 rig pose，固定外参在长序列中不漂移；
- 系统以 m 为单位，不需要事后任意尺度校正；
- `slam_gt_depth` 和 `slam_pred_depth` 只差一个 DepthProvider；
- 完成上表 A–I 消融并能解释性能变化。

---

## 11. 实施时的每日小步骤模板

每次实际开始一个小步骤时，都按下面的次序：

1. 只读本小步骤列出的文件；
2. 把函数的输入 shape、输出 shape、坐标系、单位写在笔记中；
3. 画一张数据流图，只画当前小步骤；
4. 写最小参考测试；
5. 只修改当前模块；
6. 先用人工构造数据，再用一帧真实数据，最后用 tiny sequence；
7. 保存数值误差、可视化和运行命令；
8. 通过本阶段退出条件后停止，不顺手改下一模块。

建议每个小步骤都维护四张表：

### Tensor 表

| 变量 | shape | dtype/device | 单位 | 含义 |
|---|---|---|---|---|
| 例：`range` | `[B,2,H,W]` | float32/CUDA | m | 沿单位鱼眼 ray 的距离 |

### 坐标系表

| 变量 | 变换方向 | 是否优化 | 来源 |
|---|---|---|---|
| 例：`T_right_left` | left -> right | 否 | stereo 标定 |

### 几何调用表

| 文件/函数 | project | unproject | Jacobian | range 语义 | valid mask |
|---|---:|---:|---:|---:|---:|

### 回归表

| 测试 | 修改前 | 修改后 | 阈值 | 结论 |
|---|---:|---:|---:|---|

---

## 12. 现在不应过早做的事

- 在 CameraModel 尚未通过往返和 Jacobian 测试前修改 BA；
- 在 oracle mapping 未通过前将渲染错误归因于跟踪；
- 在左目时间跟踪未通过前同时加入双目边和回环；
- 在 `slam_gt_depth` 未完整跑通前训练深度网络；
- 在 stereo metric 尺度稳定性未验证前直接删除所有 Sim(3) 诊断；
- 用调阈值、扩大 mask 或关闭 loss 来掩盖坐标系、单位或 Jacobian 错误。

---

## 13. 下一次实际开工的唯一任务

下一次不立即修改跟踪或渲染器，只执行“阶段 1：双目鱼眼数据边界”。需要先提供或确认：

1. 一对左右 RGB 样例及完整目录规则；
2. 左右标定文件原文，包括 camera model 名称；
3. 双目外参原文及方向说明；
4. 一个 pose 文件样例及其格式说明；
5. 一张深度图原文件、单位、invalid 值和官方定义；
6. 确认深度是只有左目，还是左右都有。

这些信息确认后，先完成 `StereoFisheyeFrame` 数据契约和可视化检查，然后才进入阶段 2。
