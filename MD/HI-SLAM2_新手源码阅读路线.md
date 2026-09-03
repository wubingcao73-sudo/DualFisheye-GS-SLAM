# HI-SLAM2 分阶段阅读与原生双目鱼眼适配教程

> 目标：原始双目鱼眼图像直接进入跟踪、优化、Gaussian 渲染和表面融合，整个算法始终工作在原始鱼眼像素域。
>
> 方法：一次只完成一个阶段。每阶段都按“读源码 → 理解语法与数据 → 修改限定模块 → 小测试 → 完整测试 → 验收”的顺序进行。上一阶段没有验收，不进入下一阶段。
>
> 本文目前是阅读和改造路线，不代表已经修改工程算法。真正实施时，首先需要根据你的标定文件确定鱼眼模型，例如 Kannala–Brandt、Double Sphere、Unified/MEI 或其他模型。

---

## 0. 最终系统要变成什么

当前 HI-SLAM2 的真实结构是：

```text
单目针孔 RGB
  -> DROID 针孔重投影与局部 BA
  -> 单目先验 JDSA
  -> Sim(3) 回环
  -> 针孔 3DGS rasterizer
  -> 针孔渲染深度
  -> Open3D 针孔 TSDF
```

目标系统是：

```text
同步左/右原始鱼眼 RGB + 两套鱼眼标定 + 固定双目外参
  -> 原生鱼眼特征匹配
  -> 原生鱼眼时间重投影 BA
  -> 同时刻左右鱼眼约束，建立米制尺度
  -> 原生鱼眼回环与全局 BA
  -> 原生鱼眼 Gaussian 初始化和可微 rasterizer
  -> 左右视图共享 rig pose 和固定外参
  -> 原生鱼眼深度/点云融合
  -> mesh、Gaussian、rig 轨迹和左右相机轨迹
```

关键原则：

- 输入、匹配、优化、渲染和融合全部保持原始鱼眼像素几何；
- 所有涉及 ray、投影、反投影和 Jacobian 的地方都必须认识鱼眼模型；
- 左右相机属于同一个刚性 rig，不能各自漂移；
- 双目提供绝对尺度后，不能继续无条件执行单目尺度归一化；
- 不能继续使用只适用于水平针孔双目的 `depth=fx*baseline/disparity` 思维；
- 原生鱼眼像素对应的是非线性 ray，需要由标定模型反投影并三角化。

本文统一采用位姿记号 `T_ab`：它把 `b` 坐标系中的点变换到 `a` 坐标系。例如：

```text
T_cw：world -> camera
T_wc：camera -> world
T_rl：left -> right
T_cr：rig -> camera
T_rw：world -> rig
```

于是同一时刻某个相机的世界到相机位姿为：

```text
T_cw = T_cr * T_rw
```

全文看到位姿乘法时都按这个规则解释。

---

## 1. 总阶段流程

```text
阶段 0：跑通当前单目工程，记录不可破坏的基线
   ↓
阶段 1：读在线跟踪，建立原生鱼眼 CameraModel
   ↓ 修改 Python 投影、反投影、有效 mask 和数值 Jacobian测试
阶段 2：把原生鱼眼模型移入 DROID CUDA，跑通左目时间跟踪
   ↓ 修改 BA、frame distance、covis distance、PGBA 重投影
阶段 3：加入右目同一时刻的固定外参约束，获得米制尺度
   ↓ 修改状态结构、stereo factor、深度和 JDSA
阶段 4：适配在线回环和离线全局 BA
   ↓ 处理 Sim(3)/SE(3)、scale update 和外部米制观测
阶段 5：实现原生鱼眼 Gaussian 初始化和 forward rasterizer
   ↓ 自定义 ray 反投影、2D covariance、tile、RGB/depth forward
阶段 6：实现 backward、在线 mapping 和左右相机联合监督
   ↓ pose/Gaussian 梯度、rig pose、固定外参、loss 和 densification
阶段 7：适配离线 refinement、完整轨迹和原生鱼眼 TSDF
   ↓
完成原生双目鱼眼 Gaussian SLAM
```

每次只读当前阶段列出的文件。文末的语法附录按需查阅，不要求预先学完。

---

# 阶段 0：冻结当前单目基线

## 0.1 为什么必须先做

后续会同时修改 Python、C++、CUDA、状态结构和渲染器。没有基线时，即使程序能运行，也无法判断改造是否让跟踪、深度或地图退化。

## 0.2 只读这些文件

1. `README.md`：Run Demo、Run your own data、输出说明；
2. `config/replica_config.yaml`：先认识配置结构；
3. `calib/replica.txt`：认识当前 `[fx,fy,cx,cy]`；
4. `demo.py:30-76`：输入和轨迹保存；
5. `demo.py:80-138`：主循环；
6. `hislam2/hi2.py:20-57, 86-105`：系统对象和在线调用顺序。

## 0.3 需要理解的语法

### 命令行参数

```python
args = parser.parse_args()
```

运行命令中的 `--imagedir` 等参数会变成 `args.imagedir`。

### Python 主入口

```python
if __name__ == "__main__":
```

只有直接运行 `demo.py` 时执行其下代码。

### 图像形状

```text
OpenCV:          [H,W,3]
PyTorch 单图:   [3,H,W]
送入 Hi2:       [1,3,H,W]
当前内参:       [1,4]
```

## 0.4 本阶段操作

- 跑通已有 Replica 或可用单目数据；
- 保存运行命令和配置；
- 记录输入分辨率、关键帧数、Gaussian 数量和运行时间；
- 保存 `traj_full.txt`、`traj_kf.txt`、`3dgs_final.ply`、一张 RGB render 和 depth；
- 记录当前变量约定：

| 变量 | 当前含义 |
|---|---|
| `video.poses` | 世界到相机 `T_cw` |
| 输出轨迹 | 相机到世界 `T_wc` |
| `video.disps` | 1/8 分辨率逆深度 |
| `video.disps_up` | 全分辨率逆深度 |
| `Hi2.call_gs().depths` | `1/disps_up` |
| Gaussian `_xyz` | 世界坐标 |

## 0.5 不要修改

- 不接双目；
- 不改投影；
- 不改 CUDA；
- 不调阈值掩盖失败。

## 0.6 退出条件

- 相同命令可以重复完成；
- 能画出 `demo.py -> Hi2.track -> Hi2.terminate`；
- 有一组可用于后续对比的结果。

完成后停止，开始阶段 1。

---

# 阶段 1：阅读在线跟踪，并实现原生鱼眼相机模型

这一阶段完成后，先拥有一个独立、可测试的原生鱼眼 `project/unproject`，暂时不追求整个 SLAM 运行。

## 1.1 按顺序阅读在线跟踪

### A. 输入层

阅读：

- `demo.py:30-63`：`mono_stream()`；
- `demo.py:80-125`：reader process、queue 和 `Hi2.track()`；
- `scripts/preprocess_owndata.py`：现有自有数据如何提取单目图像和针孔内参。

只回答：图像在哪里读取、何时 resize、标定如何进入系统、queue 传了哪些字段。

### B. 系统调度

阅读：

- `hislam2/hi2.py:20-57`：所有子模块；
- `hislam2/hi2.py:86-105`：在线顺序；
- `hislam2/hi2.py:72-84`：tracker 到 mapper 的数据边界。

### C. 候选关键帧

阅读：

- `hislam2/motion_filter.py`；
- `hislam2/modules/droid_net.py:149-172`；
- `hislam2/modules/extractor.py:118-198`；
- `hislam2/modules/corr.py` 只看 `CorrBlock`。

先记住：

- `fnet` 提取匹配 feature；
- `cnet` 产生 GRU 隐状态和上下文；
- `update` 输出像素坐标修正和置信度；
- MotionFilter 的一次 update 只用于判断运动量，不是完整 BA。

### D. 局部跟踪

阅读：

- `hislam2/track_frontend.py`；
- `hislam2/factor_graph.py:11-231`；
- `hislam2/depth_video.py` 的字段、`append/reproject/distance/cuda_ba`。

只理解调用，不推导矩阵：

```text
当前 pose/depth -> 几何重投影 coords1
coords1 + correlation -> update network
update network -> delta/weight
target = coords1 + delta
target/weight -> BA 更新 pose/depth
```

### E. 当前针孔实现

精读：

- `hislam2/geom/pinhole.py`；
- `hislam2/geom/projective_ops.py`；
- `src/droid_kernels.cu:58-72`。

当前模型是：

```text
u = fx*X/Z + cx
v = fy*Y/Z + cy
ray = [(u-cx)/fx, (v-cy)/fy, 1]
```

这三行是第一批需要被抽象掉的针孔假设。

## 1.2 必须先确定鱼眼标定模型

不同模型不能只靠相同的 `k1,k2,k3,k4` 名字互换。实施前从标定文件确认：

- 模型名称；
- 参数排列；
- 像素坐标原点定义；
- 投影是否包含 skew；
- 有效成像圆/有效 mask；
- 左右外参方向和单位；
- 时间同步方式。

建议形成统一接口：

```python
class CameraModel:
    def project(self, points_3d):
        """camera 3D points -> pixels + valid mask"""

    def unproject(self, pixels):
        """pixels -> unit camera rays + valid mask"""

    def project_jacobian(self, points_3d):
        """d pixel / d point"""

    def scaled(self, sx, sy):
        """return model for resized image"""
```

这只是建议接口。核心是所有模块调用同一实现，不让 BA、Gaussian 和 TSDF 各写一套稍有不同的公式。

## 1.3 本阶段需要掌握的语法

### 类与 `self`

```python
class CameraModel:
    def __init__(self, params):
        self.params = params
```

`self.params` 属于每台相机对象，因此左右相机可以使用同一类但拥有不同参数。

### 字典

建议双目输入使用有名字的字段：

```python
frame = {
    "timestamp": t,
    "left": left,
    "right": right,
    "camera_left": camera_left,
    "camera_right": camera_right,
    "T_right_left": T_right_left,
}
```

不要用越来越长的位置 tuple，使左右字段难以辨认。

### Tensor 最后一维

```text
pixels: [N,2]
points: [N,3]
rays:   [N,3]
```

`...` 可以代表任意前置 batch 维，接口最好支持 `[...,2]` 和 `[...,3]`。

### 布尔 mask

投影不能只返回像素，还要返回有效性：

```python
pixels, valid = camera.project(points)
pixels = pixels[valid]
```

鱼眼有效性至少考虑模型定义域、成像圆、点是否位于可观测方向和数值稳定性。

## 1.4 原生鱼眼输入改造

本阶段或阶段 2 实施时需要：

- 成对读取左、右原始鱼眼图；
- 根据真实 timestamp 同步，不使用“排序后第 n 张一定对应”的隐含假设；
- 不对输入图像执行改变相机成像模型的重映射；
- 网络和几何模块直接接收原始鱼眼图；
- resize 原始鱼眼图时只更新适用的像素尺度参数；
- 保留畸变模型参数；
- 为图像边界和无效成像区域保存 mask。

注意：对于常见中心鱼眼模型，resize 时 `fx/fy/cx/cy` 随图像缩放，而无量纲畸变系数通常不缩放；最终以实际模型定义为准。

## 1.5 `project/unproject` 单元测试

在接入 SLAM 之前必须通过：

### 往返测试

```text
pixel -> unproject ray -> 取任意正深度生成 point -> project -> pixel'
```

要求有效区域内 `pixel'` 接近原 pixel。

### 三维往返

```text
point -> project pixel -> unproject ray
```

要求 ray 方向与 point 方向一致。

### 标定库对照

随机生成相机前方/有效 FoV 内的 3D 点，与产生标定的相机库投影结果比较。

### 边界测试

- 图像中心；
- 成像圆边缘；
- 超出有效 FoV；
- 接近光轴；
- 接近模型奇异位置；
- 非法深度和 NaN。

## 1.6 Jacobian 测试

BA 和 Gaussian covariance 都需要投影 Jacobian。先用中心差分检查解析 Jacobian：

```text
J_num[:,k] = [project(P + eps*e_k) - project(P - eps*e_k)] / (2*eps)
```

再与 `project_jacobian(P)` 比较。不要仅在图像中心测，必须覆盖边缘大入射角区域。

## 1.7 本阶段不要做

- 不改 CUDA BA；
- 不加 stereo factor；
- 不改 Gaussian；
- 不把 OpenCV fisheye 函数当作运行时唯一实现而跳过 Jacobian；
- 不假设所有 3D 点只需判断 `Z>0`，超过 180° 或特殊鱼眼模型需按模型定义有效性。

## 1.8 退出条件

- 左右原始鱼眼图可同步读取；
- 标定模型和参数排列已确认；
- `project/unproject` 与标定库一致；
- 解析 Jacobian通过数值差分；
- resize 后模型仍通过投影测试；
- 全部测试始终使用原始鱼眼像素坐标。

完成后停止，进入阶段 2。

---

# 阶段 2：把原生鱼眼投影接入 DROID 在线跟踪

这一阶段先让左目原始鱼眼完成时间跟踪。右目暂时只同步保存，不参与 BA。

## 2.1 需要复读的文件

1. `hislam2/geom/projective_ops.py`；
2. `hislam2/geom/pinhole.py`；
3. `hislam2/depth_video.py:169-242`；
4. `hislam2/factor_graph.py:update()`；
5. `hislam2/track_frontend.py`；
6. `src/droid.cpp`；
7. `src/droid_kernels.cu` 中所有 `proj/iproj` 调用；
8. `frame_distance_kernel`；
9. `covis_distance_kernel`；
10. `projective_transform_kernel` 与 `projective_transform2_kernel`。

## 2.2 为什么 Python 改完仍不能运行正确

`DepthVideo.cuda_ba()` 实际调用编译后的 `droid_backends.ba()`。因此：

```text
Python projective_ops 改成鱼眼
≠
实时 CUDA BA 已经变成鱼眼
```

Python 版本主要用于测试、PGO 某些辅助计算和数学参照；真实高速 BA、距离计算和 JDSA 辅助项必须同步修改 CUDA。

## 2.3 深度参数化必须先定义

当前针孔逆深度的反投影隐含 ray 的第三维为 1。原生鱼眼建议明确选择：

### 方案 A：Z-depth

若 unit ray 为 `r=[rx,ry,rz]`，三维点为：

```text
P = (Z/rz) * r
```

优点是与当前 TSDF/深度习惯较接近；大 FoV 边缘 `rz` 很小时数值敏感，超过 90° 时还可能改变符号。

### 方案 B：ray range

```text
P = range * unit_ray
```

更自然地支持超大 FoV，变量是 inverse range。Gaussian 无偏深度和 TSDF 也要统一使用 range。

本项目若鱼眼 FoV 接近或超过 180°，推荐认真评估 inverse range，而不是强行沿用 Z-depth。选定后，DROID、stereo、Gaussian、输出和评估全部使用统一定义或在接口处显式转换。

## 2.4 CUDA 修改边界

至少包括：

- 鱼眼 `project`；
- 鱼眼 `unproject`；
- `d pixel / d point`；
- point 对 inverse-depth/inverse-range 的 Jacobian；
- 有效 FoV mask；
- BA Hessian/gradient 中投影 Jacobian；
- `frame_distance` 中的鱼眼重投影距离；
- `covis_distance` 中的鱼眼图像有效区域判断；
- JDSA 使用的 `proj_trans`；
- PGBA 使用的重投影。

不要在每个 kernel 复制完整模型公式。建议把模型核心写成可内联的 `__device__` 函数，并让所有 kernel 共用。

## 2.5 本阶段需要掌握的 CUDA 语法

```cpp
__device__ float2 project(...)
__global__ void kernel(...)
```

- `__device__`：GPU 内部辅助函数；
- `__global__`：从 CPU 启动的并行 kernel；
- `threadIdx/blockIdx`：当前 GPU 线程索引；
- `__shared__`：同一 block 共享内存；
- `__syncthreads()`：等待同一 block 线程。

`PackedTensorAccessor` 让 CUDA 用 `tensor[i][j]` 访问 PyTorch Tensor。

## 2.6 内参数据结构不能再固定为 4

当前大量代码默认：

```text
intrinsics = [fx,fy,cx,cy]
```

原生鱼眼至少需要：

- camera model type；
- 不定数量或模型固定数量参数；
- 图像宽高；
- 有效区域信息。

第一版可以为确定的单一模型使用固定长度参数 Tensor；若项目需要支持多种模型，再增加 model ID 和统一分派。不要让 Python 使用一种参数顺序、CUDA 使用另一种。

## 2.7 网络与几何的区别

DROID 的卷积特征和相关体本身不显式使用针孔公式，理论上可以在原始鱼眼图上提取 feature。但网络训练数据多为普通透视图，原始鱼眼边缘可能分布外：

- feature 匹配可能变差；
- MotionFilter 的平均 flow 阈值含义可能变化；
- 图像边缘像素对应的角分辨率不同；
- 后续可能需要鱼眼数据微调网络。

第一版先不训练网络，只替换几何并记录边缘与中心的 `weight/residual` 分布。几何正确仍失败时，再考虑微调。

## 2.8 分层测试顺序

1. Python 鱼眼重投影测试；
2. CUDA project/unproject 与 Python 对照；
3. CUDA Jacobian 与数值差分对照；
4. 单条 `i->j` 边的重投影；
5. 固定 pose，只更新 depth；
6. 固定 depth，只更新 pose；
7. 2—3 帧短 BA；
8. 12 帧初始化；
9. 完整左目鱼眼时间跟踪；
10. 原针孔模式回归。

## 2.9 重点观察

- 图像中心和边缘 residual；
- `weight` 是否在边缘全部接近 0；
- BA 是否出现 NaN/Inf；
- inverse-depth/range 是否越界；
- `frame_distance` 是否能正确判断邻近帧；
- 关键帧数量是否异常暴涨或过少；
- 大旋转时跟踪是否比针孔假模型更稳定。

## 2.10 退出条件

- 原始左鱼眼图可完成 DROID 初始化和连续跟踪；
- Python/CUDA 投影一致；
- BA Jacobian经过测试；
- frame/covis distance 使用原生鱼眼有效域；
- 旧针孔模式可以通过配置保留，或至少有清晰的回归分支；
- 右目仍未被当作时间帧错误加入图中。

完成后，原生鱼眼在线单目时间跟踪已适配，进入双目阶段。

---

# 阶段 3：加入原生双目约束，获得米制尺度

原生鱼眼双目不能简单沿水平方向搜索 disparity。左右像素对应关系通常位于鱼眼模型决定的曲线上，最稳妥的几何表达是“两条相机 ray + 固定外参”。

## 3.1 先读这些文件

1. `hislam2/depth_video.py`：需要增加右目 feature/观测和 stereo 状态；
2. `hislam2/modules/droid_net.py`：feature 与 update 的输入；
3. `hislam2/modules/corr.py`：相关查询；
4. `hislam2/factor_graph.py`：时间边和未来 stereo 边；
5. `hislam2/geom/ba.py:161-218`：当前 JDSA；
6. `src/droid_kernels.cu`：BA/JDSA 核心；
7. `hislam2/motion_filter.py`：左右图 feature 在何时计算；
8. `hislam2/hi2.py`：双目 packet 和缓存生命周期。

## 3.2 正确的 rig 状态

每个时间节点只有一个可优化 rig pose。左右相机 pose 由固定外参得到：

```text
T_lw = T_lr * T_rw
T_cw = T_cr * T_rw
```

上式中的 `c` 可以是 left 或 right。如果选择左目作为 rig 参考，`T_lr=I`，并且 `T_rw=T_lw`。右目不能作为另一个独立时间节点自由优化，否则 baseline 会漂移。

## 3.3 两种双目接入方式

### 方式 A：先计算鱼眼 stereo depth/range，再作为观测

外部或独立模块给左像素提供 metric depth/range 和 confidence，BA 增加加权深度残差。

优点：容易分阶段；缺点：匹配和几何在 BA 外部，信息可能损失。

### 方式 B：把左右对应直接作为 stereo reprojection factor

对左像素：

```text
left pixel + inverse depth/range
 -> left unproject ray
 -> 生成左相机 3D point
 -> 固定 T_right_left 变到右相机
 -> right fisheye project
 -> 与网络预测的右像素 target 比较
```

这样同一时刻 stereo edge 直接约束 depth 和 rig scale，而固定外参不参与每帧自由优化。

推荐最终采用方式 B；可以先实现 A 验证单位和尺度，再升级到 B。

## 3.4 原生鱼眼三角化

若先外部计算深度：

1. 左像素 `u_l` 反投影为 unit ray `r_l`；
2. 右像素 `u_r` 反投影为 unit ray `r_r`；
3. 用 `T_right_left` 把两条 ray 表达到同一坐标系；
4. 求两条空间射线的最近点；
5. 根据两射线夹角、最近距离和正深度判断有效性；
6. 输出选定的 Z-depth 或 range；
7. 输出 confidence。

远处两 ray 近似平行，三角化不稳定，不能只看像素匹配置信度。

## 3.5 需要增加的数据

建议明确区分：

- `left_fmap/right_fmap`；
- `stereo_target`；
- `stereo_weight`；
- `stereo_depth` 或 `stereo_range`；
- `stereo_valid_mask`；
- `T_right_left`；
- `camera_left/camera_right` 参数；
- 时间边和 stereo 边类型。

不要复用 `disps_prior` 同时表示 OmniData 和 stereo。OmniData 是相对先验，需要 scale grid；stereo 是米制观测，不应有每帧自由 scale。

## 3.6 当前 JDSA 如何处理

推荐分工：

```text
多视时间 BA：全图几何
stereo factor：有效区域的米制深度和尺度
OmniData + JDSA：stereo/多视不可靠区域的弱先验
```

JDSA 的 2×2 scale grid 只对 OmniData prior 生效，不能乘到原始 stereo measurement 上。

## 3.7 `normalize()` 必须条件化

当前初始化后会把单目尺度归一化。只有 stereo factor 已真正约束优化后，才可以在 stereo 模式关闭该行为。

不能只关闭 `normalize()`：如果 stereo 尚未进入 BA，平移仍没有米制尺度。

## 3.8 语法重点

### mask

```python
valid = correspondence_valid & triangulation_valid & model_valid
```

`&` 是逐元素逻辑与，左右表达式应加括号。

### `torch.where`

```python
observation = torch.where(valid, stereo_value, fallback)
```

但 loss 中仍应显式乘权重/mask，不能只用 0 代表无效。

### 固定参数与优化变量

baseline 和左右外参是固定标定量；rig pose 与 depth/range 是状态变量。第一版不要把外参加入每帧优化器。

## 3.9 测试顺序

1. 静态场景左右对应点的 ray 三角化；
2. 已知距离平面的米制误差；
3. 单时间节点 stereo depth 优化；
4. 2—3 个时间节点，时间边 + stereo 边；
5. 平移 1 米的尺度检查；
6. 弱纹理、远距离和遮挡区域；
7. 左右大 FoV 边缘；
8. mono 模式回归。

## 3.10 退出条件

- 左右图严格同步且外参方向正确；
- stereo factor 只更新共享 rig pose/depth，不破坏 baseline；
- 输出 depth/range 与 translation 同为米制；
- 原始 stereo 测量不会被 JDSA scale grid 改写；
- 无效和近似平行 ray 不造成 NaN；
- 大视场边缘也通过三角化与重投影检查。

完成后，原生双目鱼眼在线跟踪主体完成。

---

# 阶段 4：适配回环、PGBA 和全局 BA

## 4.1 阅读顺序

1. `hislam2/factor_graph.py:120-181`：inactive edge；
2. `hislam2/pgo_buffer.py:65-124`：相对 pose/covariance；
3. `hislam2/pgo_buffer.py:125-217`：回环候选；
4. `hislam2/factor_graph.py:update_pgba()`；
5. `hislam2/depth_video.py:cuda_pgba()`；
6. `hislam2/track_backend.py`；
7. `hislam2/factor_graph.py:update_lowmem()`；
8. `hislam2/hi2.py` 中 PGBA 与 full BA 调度。

## 4.2 必须替换的针孔依赖

- 回环距离 `DepthVideo.distance()`；
- `frame_distance_kernel`；
- 方向和有效视野判断；
- 回环边的重投影；
- 相对位姿估计中的 `projective_transform()`；
- PGBA CUDA 重投影；
- 离线 full BA 的所有边。

阶段 2 已修改底层共用函数时，这里应主要验证调用路径，而不是再写另一套鱼眼公式。

## 4.3 `Sim3` 先保留作为诊断

双目理论上提供绝对尺度，但第一版建议保留 `Sim3` PGBA，并记录所有 `dscale`：

- 如果长期接近 1，说明 stereo 尺度稳定；
- 如果频繁明显偏离 1，先查 stereo、同步、外参和深度定义；
- 不要立即用 `SE3` 把尺度问题隐藏起来。

米制 stereo observation 是外部测量，不应在 PGBA 后被当成单目状态一起任意缩放。应区分：

- 当前估计的 `disps/disps_up`；
- OmniData 对齐尺度；
- 原始 stereo measurement 与 confidence。

## 4.4 什么时候改 `SE3`

只有在长轨迹中：

- stereo 有效关键帧比例高；
- `dscale` 始终近似 1；
- 米制轨迹稳定；
- `Sim3` 不再改善误差反而引入局部伸缩；

才实现 stereo 模式 `SE3` PGBA，同时保留 mono `Sim3`。

## 4.5 测试

- 原地旋转闭环；
- 平移后返回原位；
- 大视场回环候选；
- 回环前后首尾误差；
- `dscale` 分布；
- 原始 stereo residual 是否保持；
- full BA 后 baseline 和 rig 刚性是否保持。

## 4.6 退出条件

- 原生鱼眼回环距离和重投影有效；
- 回环改善轨迹而不破坏米制尺度；
- 左右固定外参没有被独立漂移；
- PGBA/full BA 后所有 depth 状态同步且原始测量不被改写；
- mono 和 stereo 的 `Sim3/SE3` 策略明确。

---

# 阶段 5：实现原生鱼眼 Gaussian 初始化和 forward rasterizer

这是第二个大改造点。现有 Gaussian CUDA 与 DROID CUDA 是两套独立投影实现；跟踪改成鱼眼，不代表渲染自动变成鱼眼。

## 5.1 阅读顺序

1. `hislam2/hi2.py:call_gs()`；
2. `hislam2/gs_backend.py:72-118`；
3. `hislam2/gaussian/utils/camera_utils.py`；
4. `hislam2/gaussian/utils/graphics_utils.py`；
5. `hislam2/gaussian/scene/gaussian_model.py:32-223`；
6. `hislam2/gaussian/renderer/__init__.py`；
7. `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`；
8. 依次搜索 `preprocessCUDA`、`computeCov2D`、`renderCUDA`；
9. 最后看 tile sorting 和 `rasterizer_impl.cu`。

## 5.2 先替换 Gaussian 初始化

当前 Open3D `PinholeCameraIntrinsic` 不能用于原生鱼眼反投影。需要自定义：

```text
鱼眼 pixel
 -> CameraModel.unproject 得 unit ray
 -> 根据选定 depth/range 定义生成相机点
 -> 用 camera-to-world 变换到世界
 -> RGB 赋色
 -> mask 过滤
 -> 下采样/KNN 初始化 Gaussian scale
```

如果状态是 Z-depth：`P=(Z/rz)*r`；如果是 range：`P=range*r`。必须和阶段 2、3 完全一致。

## 5.3 原生鱼眼 2D covariance

针孔 rasterizer 用透视 Jacobian把 3D Gaussian covariance 传播到二维。原生鱼眼也可以先使用局部一阶近似：

```text
Sigma_cam = R * Sigma_world * R^T
J = d fisheye_project(P_cam) / d P_cam
Sigma_2D = J * Sigma_cam * J^T
```

这不是换投影矩阵，而是每个 Gaussian 在当前相机位置计算鱼眼投影 Jacobian。

大 Gaussian 或极端 FoV 边缘的一阶椭圆近似可能变差，需要后续以数值实验评估；第一版先保证小 Gaussian 前向正确。

## 5.4 屏幕中心与 tile

`preprocessCUDA` 需要改为：

- 使用鱼眼 `project(mean_cam)` 得二维中心；
- 使用 `Sigma_2D` 计算屏幕半径；
- 按鱼眼有效 mask 做 culling；
- 计算覆盖 tile；
- 排序深度使用统一 depth/range 定义。

鱼眼边缘的投影 footprint 可能强烈非线性，必须测试半径是否覆盖真实影响区域，宁可第一版保守扩大 bounding，也不能漏 tile。

## 5.5 原生鱼眼 depth forward

现有无偏深度计算中的 pixel ray 和 ray-plane 推导基于针孔归一化平面。原生鱼眼需要：

- 每个像素通过 `unproject` 得真实 unit ray；
- 明确输出 Z-depth 还是 range；
- 计算 ray 与 Gaussian 局部表面的交点或一致的一阶近似；
- alpha blending 时使用相同深度语义；
- 输出 mask 表示无 Gaussian 命中的像素。

不要保留 `pix -> [(x-W/2)/f,...,1]` 的针孔 ray。

## 5.6 Forward 测试必须脱离 SLAM

### 单点/单 Gaussian

- 光轴附近；
- 图像中部；
- 成像圆边缘；
- 左右相机分别测试；
- 改变深度，检查像素中心与 footprint。

### 已知平面

渲染一个已知世界平面，检查原生鱼眼 RGB/depth 与 CPU ray-plane 参考实现。

### 多 Gaussian

检查遮挡顺序、alpha blending、tile 边界和图像有效圆外的背景。

## 5.7 本阶段不要做

- 不立即实现 backward；
- 不接在线 mapping loss；
- 不同时调 densification；
- 不用透视投影矩阵伪装鱼眼；
- 不用视觉上“差不多圆”代替像素/深度数值测试。

## 5.8 退出条件

- 原生鱼眼 depth 能自定义反投影初始化 Gaussian；
- 单 Gaussian 投影中心与 CameraModel 一致；
- 2D covariance 和 tile 覆盖通过参考测试；
- RGB/depth forward 在全 FoV 有正确结果；
- 左右 camera 参数和 pose 均可独立传入 renderer；
- 整个 forward 始终使用原始鱼眼图像大小和像素坐标。

---

# 阶段 6：实现 backward、在线 mapping 和双目联合监督

## 6.1 阅读顺序

1. `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`；
2. `rasterize_points.cu` 和 Python autograd 包装；
3. `hislam2/gs_backend.py:152-273`；
4. `hislam2/gaussian/utils/slam_utils.py:93-245`；
5. `hislam2/gaussian/utils/loss_utils.py`；
6. `GaussianModel.densify_and_prune()`；
7. `hislam2/gs_backend.py:78-95`：回环地图变形。

## 6.2 backward 必须传回哪些梯度

- Gaussian mean；
- scale；
- rotation；
- color；
- opacity；
- camera/rig pose delta；
- 原生鱼眼 depth 相关中间量。

每个新解析梯度都先做有限差分或 PyTorch CPU reference 对照，再进入完整 mapping。

## 6.3 深度转点和法线

当前 `slam_utils.depths_to_points()` 写死针孔 ray 和中心主点。需要调用统一 `CameraModel.unproject()`：

```text
每像素 fish-eye ray + rendered depth/range -> camera 3D point
邻域 3D point 差 -> cross product -> normal
```

图像有效圆边缘、depth discontinuity 和无效点不能参与普通邻域叉积。

## 6.4 左右相机共享 rig pose

Gaussian mapping 中每个时间戳可以有 left/right 两个 Camera view，但它们必须由一个 rig pose 和固定外参派生。

建议概念：

```text
rig_timestamp
camera_id = left/right
view_id = (rig_timestamp, camera_id)
T_camera_world = T_camera_rig * T_rig_world
```

离线/在线 pose 优化不能给左右 view 各自一个完全独立的 6DoF delta，否则 baseline 会漂移。

## 6.5 Gaussian 初始化和监督分工

第一版：

- 左目可靠 metric depth 创建 Gaussian；
- 右目只提供 RGB/depth 渲染监督；
- 等地图稳定后，再允许右目在左目不可见区域补充 Gaussian；
- 多 view 重叠处先做深度/世界距离检查，避免双层表面。

## 6.6 loss mask

至少包含：

- 左右鱼眼有效成像圆；
- stereo correspondence confidence；
- 遮挡和左右一致性；
- rendered visibility；
- depth/range 合法范围；
- 动态物体 mask（如果有）。

当前 `get_loss_mapping_rgbd()` 仅用简单深度阈值，不能直接满足原生双目鱼眼。

## 6.7 OmniData 法线/深度先验

OmniData主要在普通透视图上训练。原始强畸变鱼眼边缘可能分布外。第一版可以：

- 中央可信区域保留先验；
- 边缘根据 stereo、多视或几何 normal 加权；
- 统计先验误差随入射角变化；
- 不让 2×2 scale grid 掩盖投影模型问题。

若后续需要，可在鱼眼数据上替换或微调 prior 模型，但这不是第一版 mapper 的前提。

## 6.8 回环后地图变形

检查每个 Gaussian 的 anchor 是 rig keyframe，而不是某个可以独立漂移的 camera view。PGBA 返回 rig pose/scale update 后，左右视图共享同一地图变形。

## 6.9 测试顺序

1. 单 Gaussian loss 的有限差分；
2. 固定相机，只优化 Gaussian；
3. 固定 Gaussian，只优化 rig pose；
4. 左目单 view 短 mapping；
5. 左右双 view 监督；
6. 回环前后地图变形；
7. densification/pruning；
8. 完整在线序列。

## 6.10 退出条件

- forward/backward 都通过梯度检查；
- 原始鱼眼 RGB/depth 可以直接监督 Gaussian；
- 左右 view 始终满足固定外参；
- depth/normal 使用真实鱼眼 ray；
- 无效圆、遮挡和低置信区域不产生漂浮 Gaussian；
- 回环变形以 rig keyframe 为锚点。

---

# 阶段 7：适配离线 refinement、轨迹和原生鱼眼 TSDF

## 7.1 阅读顺序

1. `hislam2/hi2.py:106-165`：`terminate()`；
2. `hislam2/util/trajectory_filler.py`；
3. `hislam2/track_backend.py`；
4. `hislam2/gs_backend.py:275-329`：joint refinement；
5. `hislam2/gaussian/utils/eval_utils.py`；
6. `demo.py:66-76`：轨迹保存；
7. `tsdf_integrate.py`。

## 7.2 TrajectoryFiller

普通帧只需填充 rig/左参考相机 pose。右相机 pose 由固定外参派生，不应独立运行 filler。

## 7.3 Joint refinement

当前每个 `Camera` 有独立 pose delta。原生双目 rig 应改为：

- 每个 timestamp 一个共享 rig pose delta；
- 左右外参固定；
- 左右 view 可以各自优化曝光；
- 如果未来在线微调外参，应是全序列共享的小参数并带强先验，而不是每帧独立外参。

## 7.4 输出轨迹必须明确参考系

建议分别输出：

- rig/body `T_wr`；
- left camera `T_wl`；
- right camera `T_wr_cam`，避免变量名歧义；
- 时间戳；
- 标定外参和单位说明。

内部/外部位姿方向必须在文件头或旁边 README 写清楚。

## 7.5 原生鱼眼 TSDF

当前 `tsdf_integrate.py` 使用 Open3D 针孔内参，不能直接用于原生鱼眼深度。

原生融合需要逐像素：

```text
fish-eye pixel
 -> unproject unit ray
 -> rendered range/depth 生成 3D point
 -> 根据 rig/camera pose 变到世界
 -> 沿真实 ray 更新 TSDF/occupancy
```

如果使用 Z-depth，需要先转换为沿 ray 的 range；如果 renderer 输出 range，则直接沿 unit ray 使用。所有视图可融合，但要处理同一 rig timestamp 左右重复观测权重。

## 7.6 最终测试集

至少包含：

1. 已知直线距离：检查米制尺度；
2. 明显闭环：检查 PGBA 和地图变形；
3. 大旋转：检查完整鱼眼 FoV；
4. 近距离立体：检查 baseline 和三角化；
5. 远距离场景：检查近似平行 ray；
6. 左右曝光差：检查光度优化；
7. 成像圆边缘：检查投影、Gaussian 和 TSDF 一致性。

评估：

- ATE/RPE；
- stereo 重投影误差；
- depth/range error；
- 回环 scale；
- RGB PSNR/SSIM/LPIPS；
- mesh accuracy/completeness；
- Gaussian 数量、显存、速度；
- 中心区域与边缘区域分开统计。

## 7.7 最终退出条件

- 原始左右鱼眼直接进入系统；
- 在线跟踪、stereo、回环均使用同一 CameraModel；
- Gaussian forward/backward 使用原生鱼眼投影；
- rig pose 与固定外参正确；
- 轨迹、Gaussian PLY、原生鱼眼 RGB/depth 和 mesh 均可输出；
- 跟踪、渲染和融合全部使用同一原生鱼眼 CameraModel。

---

# 8. 每阶段怎样做笔记

每个重要函数只记录五件事：

| 项目 | 记录内容 |
|---|---|
| 调用者 | 谁调用它 |
| 输入 | 类型、Tensor 形状、单位、坐标系 |
| 输出 | 返回值或修改的状态 |
| 核心动作 | 最多三句话 |
| 相机假设 | 是否仍含针孔、单目、单 K 或独立 pose 假设 |

建议一直维护四张表：

### Tensor 表

```text
变量 | shape | dtype | CPU/GPU | depth/range/disparity | 单位
```

### 坐标系表

```text
变量 | from | to | 左乘/右乘 | 内部方向 | 输出方向
```

### CameraModel 调用表

```text
文件/函数 | project | unproject | Jacobian | valid mask | Python/CUDA
```

### 阶段回归表

```text
测试 | 修改前结果 | 修改后结果 | 是否通过 | 失败原因
```

---

# 9. 新手语法速查

## 9.1 类和对象

```python
camera = CameraModel(params)
rays, valid = camera.unproject(pixels)
```

创建对象时执行 `__init__()`；方法中的 `self` 指当前对象。

## 9.2 特殊方法

- `__init__`：构造；
- `__call__`：对象可以像函数调用；
- `__getitem__`：支持 `obj[index]`；
- `__setitem__`：支持 `obj[index]=value`。

`self.frontend(...)` 实际进入 `TrackFrontend.__call__()`。

## 9.3 Tensor 维度

```python
x.shape
x.reshape(...)
x.permute(...)
x[None]
x[..., :2]
```

- `reshape` 改形状；
- `permute` 调整维度顺序；
- `None` 增加维度；
- `...` 保留前面所有维。

## 9.4 CPU/GPU

```python
x.cuda()
x.cpu()
x.detach().cpu().numpy()
```

CUDA Tensor 不能直接交给 NumPy。保存前通常先 `detach`、再 `cpu`。

## 9.5 自动求导

```python
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

依次为计算梯度、更新参数、清理梯度。

`with torch.no_grad()` 表示该区域不需要建立反向图。

## 9.6 装饰器

```python
@torch.no_grad()
@staticmethod
@property
```

- `no_grad`：关闭梯度；
- `staticmethod`：不使用 `self` 的类内函数；
- `property`：方法以属性形式读取。

## 9.7 mask 与索引

```python
valid = (depth > 0) & model_valid
selected = depth[valid]
remaining = depth[~valid]
```

`~` 对布尔 mask 取反。

## 9.8 `torch.cat` 和 `torch.stack`

- `cat`：沿已有维拼接；
- `stack`：增加一个新维度后堆叠。

因子图添加边通常使用 `cat`，左右相机形成 camera 维时可能使用 `stack`。

## 9.9 `SE3` 和 `Sim3`

- `.inv()`：变换求逆；
- `A * B`：复合变换；
- `.log()`：变换到李代数增量；
- `.exp()`：增量到变换；
- `.retr(dx)`：把局部更新应用到当前状态。

看到位姿乘法时必须结合坐标系表，不能只看变量名。

## 9.10 CUDA

```cpp
__device__
__global__
threadIdx.x
blockIdx.x
__shared__
__syncthreads()
```

分别代表 GPU 辅助函数、GPU kernel、线程索引、线程块索引、块内共享内存和线程同步。

---

# 10. 每阶段的停止规则

出现以下情况时，不继续下一个阶段：

- 相机模型或参数排列不确定；
- `project/unproject` 往返不一致；
- Jacobian 没有数值测试；
- depth/range/disparity 的定义混用；
- 左右外参方向不明确；
- stereo 测量被单目 scale grid 改写；
- 左右相机拥有可独立漂移的每帧 pose；
- Python 与 CUDA 投影结果不同；
- Gaussian forward 尚未正确就开始调 backward；
- 原生鱼眼深度仍交给针孔 TSDF。

遵循这个停止规则，代码阅读完一阶段时，这一阶段的适配和测试也同步完成；读完整个教程时，工程不会只停留在“看懂了”，而是已经逐层变成原生双目鱼眼 Gaussian SLAM。
