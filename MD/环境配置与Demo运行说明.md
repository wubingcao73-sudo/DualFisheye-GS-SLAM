# HI-SLAM2 环境配置与官方 Demo 运行说明

本文档记录本工程在当前工作站上的实际配置与验证结果。配置日期：2026-08-04。

## 1. 已完成状态

环境已经配置完成，官方 Replica `room0` Demo 已对完整的 2000 帧输入运行成功，程序正常输出 `Done`。

| 项目 | 已验证配置 |
| --- | --- |
| 操作系统 | Linux x86_64 |
| GPU | NVIDIA GeForce RTX 5060 Ti，16 GB，计算能力 12.0（`sm_120`） |
| 驱动 | 580.95.05 |
| CUDA Toolkit / NVCC | 12.8 / 12.8.61 |
| Python | 3.10.20 |
| Conda 环境 | `hislam2` |
| PyTorch | 2.7.1+cu128 |
| torchvision / torchaudio | 0.22.1+cu128 / 2.7.1+cu128 |
| NumPy | 1.26.4 |

官方提供的 `environment.yaml` 使用 PyTorch 2.1.2 + CUDA 11.8。这个组合不包含 RTX 50 系 Blackwell GPU 所需的 `sm_120` 内核，因此当前机器采用 PyTorch 2.7.1 + CUDA 12.8，并对旧 CUDA/C++ 扩展做了兼容修改。不要在这台机器上重新用官方 YAML 覆盖已经配置好的环境。

## 2. 当前机器直接运行

进入工程并激活环境：

```bash
cd /home/nonchalance/open-project/3DGS/HI-SLAM2
conda activate hislam2
```

确认 PyTorch 能识别 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

预期包含：

```text
2.7.1+cu128 12.8 NVIDIA GeForce RTX 5060 Ti (12, 0)
```

运行官方完整 Demo：

```bash
python demo.py \
  --imagedir data/Replica/room0/colors \
  --calib calib/replica.txt \
  --config config/replica_config.yaml \
  --output outputs/room0
```

若要重新跑而又希望保留现有结果，请换一个输出目录，例如 `--output outputs/room0_rerun`。

可选可视化参数：

- `--gsvis`：显示 Gaussian 地图；
- `--droidvis`：显示点云和中间结果。

这两个参数需要有效的图形桌面和 `DISPLAY`；SSH/headless 环境中建议不要加。

快速冒烟测试可以使用前 150 帧：

```bash
python demo.py \
  --imagedir data/Replica/room0/colors \
  --calib calib/replica.txt \
  --config config/replica_config.yaml \
  --output outputs/room0_smoke150 \
  --length 150 \
  --buffer 200
```

不要把冒烟测试缩短到 30 帧：该长度只产生约 5 个关键帧，结束阶段的后端 BA 需要至少 10 个关键帧，会因序列过短而退出；这不是环境安装错误。

## 3. 本次完整 Demo 验证结果

完整输入数据：`data/Replica/room0/colors`，共 2000 张 RGB 图像，约 977 MB。

结果目录：`outputs/room0`，约 151 MB。主要产物如下：

| 产物 | 结果 |
| --- | --- |
| `traj_full.txt` | 2000 行完整相机轨迹 |
| `traj_kf.txt` | 89 行关键帧轨迹 |
| `3dgs_final.ply` | 11,598,564 字节，170,561 个 Gaussian |
| `intrinsics.npy` | 相机内参 `[308.0, 303.5294, 307.7433, 171.7471]` |
| `renders/` | 946 个渲染/深度图文件 |

最终指标：

| 指标 | 全部评估帧 | 关键帧 |
| --- | ---: | ---: |
| PSNR | 35.3369 | 35.9303 |
| SSIM | 0.959438 | 0.963219 |
| LPIPS | 0.040635 | 0.039830 |

测试期间观察到的显存峰值约为 6.4 GB。指标原始记录位于：

```text
outputs/room0/psnr/after_opt/final_result.json
outputs/room0/psnr/after_opt/final_result_kf.json
```

## 4. 从零复现当前环境

这一节适用于重新创建环境。当前 `hislam2` 已经可用，不需要重复执行。

### 4.1 获取源码与子模块

全新下载时必须带递归子模块：

```bash
git clone --recursive https://github.com/Willyzw/HI-SLAM2.git
cd HI-SLAM2
```

本工作区原始源码是一个没有完整顶层 Git 元数据的快照，因此已手工补齐下列上游固定版本：

| 目录 | 固定提交 |
| --- | --- |
| `thirdparty/eigen` | `3d4ba855e014987cad86d62a8dff533492255695` |
| `thirdparty/lietorch` | `0fa9ce8ffca86d985eca9e189a99690d6f3d4df6` |
| `thirdparty/simple-knn` | `86710c2d4b46680c02301765dd79e465819c8f19` |
| `thirdparty/diff-gaussian-rasterization/third_party/glm` | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |

### 4.2 创建 Python 环境

```bash
conda create -n hislam2 python=3.10 -y
conda activate hislam2
python -m pip install -r requirements-cu128.txt
```

`requirements-cu128.txt` 是本次成功运行所用关键包的版本快照。它面向 CUDA 12.8/RTX 50 系；其他 GPU 如果仍使用官方 CUDA 11.8 配置，应使用官方 `environment.yaml`，并按其 CUDA 版本重新编译全部扩展。

### 4.3 编译 CUDA 扩展

项目根目录的 `setup.py` 已去除旧的硬编码 `sm_60`～`sm_86` 架构参数，改为让 PyTorch 读取 `TORCH_CUDA_ARCH_LIST`。`lietorch` 中弃用的 `Tensor.type()` 分派也已改为 `Tensor.scalar_type()`，以兼容当前 PyTorch。

在 RTX 5060 Ti 上执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4

python setup.py install

python -m pip install \
  --no-build-isolation \
  --force-reinstall \
  --no-deps \
  ./thirdparty/simple-knn \
  ./thirdparty/diff-gaussian-rasterization
```

如果升级或更换了 PyTorch、CUDA Toolkit 或 GPU，必须重新执行本节，不能继续使用旧 `.so` 文件。

扩展快速检查：

```bash
python -c "import droid_backends, lietorch_backends, diff_gaussian_rasterization, simple_knn, torch_scatter; print('CUDA extensions: OK')"
```

### 4.4 下载预训练模型

工程需要三个权重文件：

```text
pretrained_models/droid.pth
pretrained_models/omnidata_dpt_depth_v2.ckpt
pretrained_models/omnidata_dpt_normal_v2.ckpt
```

`droid.pth` 已随当前工程存在。两个 Omnidata 权重较大，可用本次添加的断点友好型分块脚本下载：

```bash
bash scripts/download_file_parallel.sh \
  https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt \
  pretrained_models/omnidata_dpt_depth_v2.ckpt \
  1947430832 8

bash scripts/download_file_parallel.sh \
  https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt \
  pretrained_models/omnidata_dpt_normal_v2.ckpt \
  1947430960 8
```

校验：

```bash
sha256sum pretrained_models/omnidata_dpt_{depth,normal}_v2.ckpt
```

预期值：

```text
a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e  omnidata_dpt_depth_v2.ckpt
38db3f7d952f48f900f0b32d6cce6749e41641045ebc5713cf29618f54aff3bb  omnidata_dpt_normal_v2.ckpt
```

代码已显式处理 PyTorch 2.6 以后 `torch.load` 的 `weights_only` 默认值变化。Omnidata 编码器也设置为直接加载本地完整 checkpoint，避免 `timm` 再联网下载 ImageNet 权重。

### 4.5 准备 Demo 数据

官方 Replica 压缩包约 12.4 GB，而 `demo.py` 的 room0 演示只读取 RGB 帧。当前工程已提供并下载好 2000 帧最小数据集。如需重建该数据目录，可执行：

```bash
python scripts/download_replica_room0_rgb.py --workers 8
```

检查帧数：

```bash
find data/Replica/room0/colors -maxdepth 1 -name 'frame*.jpg' | wc -l
```

应输出 `2000`。如果要运行官方全数据集评估而不只是 Demo，仍应按照 `README.md` 使用 `scripts/download_replica.sh` 下载并预处理完整 Replica 数据。

## 5. 常见问题

### `no kernel image is available` 或不支持 `sm_120`

通常是仍在使用 CUDA 11.8 的 PyTorch，或扩展是给旧显卡编译的。确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

RTX 5060 Ti 应使用本文 CUDA 12.8 配置，并重新编译三个 CUDA 扩展。

### `ModuleNotFoundError: droid_backends/lietorch_backends/simple_knn`

说明扩展尚未安装到当前 Conda 环境，或编译时激活了另一个环境。重新激活 `hislam2`，然后执行 4.3 节。

### `IndexError: index ... is out of bounds for dimension 0`

对较长自定义序列增加 `--buffer`。官方说明中 `buffer` 是预分配的最大关键帧数；它不是输入图像总帧数。

### CUDA 显存不足

先关闭其他 GPU 程序，然后适当降低 `--buffer`。本次 room0 默认设置在 16 GB GPU 上的峰值约 6.4 GB。

### OpenGL、`DISPLAY` 或可视化窗口错误

在无桌面环境中移除 `--gsvis` 和 `--droidvis`。主 Demo 的无界面运行不依赖可视化窗口。

### 出现 AMP、`meshgrid` 等弃用警告

当前 PyTorch 会对旧 API 打印 warning；本次完整 Demo 在这些 warning 存在时仍正常完成。只有 traceback 或进程非零退出才表示实际失败。

## 6. 本次新增或修改的关键文件

- `requirements-cu128.txt`：已验证的 CUDA 12.8 依赖版本；
- `setup.py`：CUDA 架构选择兼容 `sm_120`；
- `thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp` 与 `lietorch_gpu.cu`：兼容 PyTorch 2.7 的类型分派；
- `hislam2/hi2.py`、`hislam2/midas/*.py`：本地 checkpoint 与 PyTorch 2.6+ 加载兼容；
- `scripts/download_file_parallel.sh`：大权重并行分块下载；
- `scripts/download_replica_room0_rgb.py`：只提取官方 Demo 所需的 room0 RGB 数据。
