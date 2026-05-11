# DDRBandWidthStressTest
ddr带宽压测工具，支持arm、x86、Nvidia Orin and Thor
DDR Bandwidth StressTest tool, surpport arm/x86/Nvidia Orin/Thor

针对 Nvidia Thor SoC 芯片的 DDR 带宽上限测试工具，支持 CPU、GPU 及混合模式。

## 目标平台

- 架构：ARM64 (aarch64)
- 操作系统：Ubuntu 20.04
- GPU：Nvidia Thor SoC 集成 GPU (**Blackwell 架构, Compute Capability 10.1, sm_101**)
- 依赖：CUDA Toolkit 12.8+, pthread

## 文件结构

```
thor_ddr_bw_stress/
├── ddr_bw_stress.cu   # 主程序源码 (CUDA + C)
├── ddr_bw_stress      # 编译生成的 ARM64 可执行文件
├── Makefile           # 编译脚本（本机编译用）
├── run_test.sh        # 自动化测试运行脚本
└── README.md          # 本文件
```

## 测试模式

| 模式 | 参数 | 说明 |
|------|------|------|
| CPU-Only | `--cpu` | 8线程 ARM64 NEON SIMD 流式读/写/拷贝 |
| GPU-Only | `--gpu` | CUDA GPU DDR 读/写/拷贝 |
| CPU+GPU Mixed | `--mixed` | CPU 和 GPU 同时并发执行，测量混合带宽上限 |

每个模式包含三个子测试：READ、WRITE、COPY，每个子测试持续 60 秒，数据块大小 2GB。

## 实现原理

### CPU 测试

使用 `arm_neon.h` 的 128-bit SIMD 指令（`vld1q_u8` / `vst1q_u8`），每次迭代处理 64 字节，8 线程并行以最大化内存总线利用率。

### GPU 测试

Thor SoC 为统一内存架构（GPU 和 CPU 共享 DDR）。GPU 测试采用三种方式确保测量真实 DDR 带宽：

| 子测试 | 实现方式 | 说明 |
|--------|----------|------|
| READ | CUDA kernel (read + XOR transform + write) | 读取 2GB src 数据，逐元素变换后写入 dst，确保每次读取不可优化 |
| WRITE | `cudaMemset` | CUDA driver 硬件 DMA 引擎操作，不可被编译器优化 |
| COPY | `cudaMemcpy DeviceToDevice` | CUDA driver 硬件拷贝引擎操作，不可被编译器优化 |

程序包含运行时验证步骤：kernel 执行后会校验输出数据是否正确，确认 kernel 真正运行（而非被优化掉或静默失败）。

### Mixed 测试

通过 pthread 同时启动 CPU 和 GPU 测试线程，测量两者并发时的总带宽。

## 编译

### 方式一：在 Thor 平台上本机编译

```bash
make GPU_ARCH=sm_101
make CUDA_PATH=/usr/local/cuda
make clean
```

### 方式二：在 x86_64 主机交叉编译生成 ARM64 可执行文件（推荐）

#### 前提条件

1. 安装 aarch64 交叉编译工具链：

```bash
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

2. 安装 CUDA Toolkit 12.8+（x86_64 版本，需支持 sm_101），并下载 aarch64 CUDA runtime：

```bash
# 下载 aarch64 CUDA runtime 开发包
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-cudart-dev-12-8_12.8.90-1_arm64.deb" \
     -O /tmp/cuda-cudart-dev-arm64.deb
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-cudart-12-8_12.8.90-1_arm64.deb" \
     -O /tmp/cuda-cudart-arm64.deb

# 解压到临时目录
mkdir -p /tmp/cuda-aarch64
cd /tmp/cuda-aarch64
dpkg-deb -x /tmp/cuda-cudart-dev-arm64.deb .
dpkg-deb -x /tmp/cuda-cudart-arm64.deb .

# 将 aarch64 target 安装到 CUDA 目录
sudo cp -a /tmp/cuda-aarch64/usr/local/cuda-12.8/targets/sbsa-linux \
           /usr/local/cuda/targets/sbsa-linux
```

#### 执行交叉编译

```bash
/usr/local/cuda/bin/nvcc \
  -O2 \
  --std=c++11 \
  -ccbin aarch64-linux-gnu-g++ \
  -Xcompiler "-O2 -march=armv8-a+simd" \
  -target-dir sbsa-linux \
  -gencode arch=compute_101,code=sm_101 \
  -o ddr_bw_stress \
  ddr_bw_stress.cu \
  -lpthread -lrt
```

#### 验证生成文件

```bash
$ file ddr_bw_stress
ddr_bw_stress: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, for GNU/Linux 3.7.0, not stripped
```

#### 编译参数说明

| 参数 | 说明 |
|------|------|
| `-O2` | 优化级别（避免 -O3 过度优化消除内存操作） |
| `-gencode arch=compute_101,code=sm_101` | Thor SoC GPU 架构（Blackwell, Compute Capability 10.1） |
| `-ccbin aarch64-linux-gnu-g++` | 指定 aarch64 交叉编译器为主机编译器 |
| `-Xcompiler "-O2 -march=armv8-a+simd"` | 传递给主机编译器的 ARM NEON 支持标志 |
| `-target-dir sbsa-linux` | 指定 aarch64 CUDA target 目录（位于 `/usr/local/cuda/targets/` 下） |

### 关于 GPU 架构的重要说明

Thor SoC GPU 的 Compute Capability 为 **10.1**（Blackwell 架构），编译时 **必须** 使用 `sm_101`。

如果使用了错误的架构（如 sm_87/sm_89/sm_90），会出现以下症状：
- Kernel 启动静默失败（`cudaGetLastError()` 报 "no kernel image is available for execution" 或 "PTX JIT compiler library not found"）
- `cudaDeviceSynchronize()` 立即返回（无 pending work）
- GPU READ 带宽测量值异常偏高（数百万 GB/s），因为 kernel 没有执行

程序内置了 kernel 验证步骤和 `cudaGetLastError()` 检查，架构不匹配时会立即报错退出。

## 运行

将编译好的 `ddr_bw_stress` 可执行文件拷贝到 Thor 平台后：

```bash
# 运行全部测试（CPU + GPU + Mixed）
./run_test.sh all

# 单独运行某个测试
./run_test.sh cpu
./run_test.sh gpu
./run_test.sh mixed

# 直接运行二进制（支持组合参数）
./ddr_bw_stress --cpu --gpu --mixed
./ddr_bw_stress --cpu
./ddr_bw_stress --gpu
```

## 输出示例

```
================================================================
  DDR Bandwidth Stress Test - Nvidia Thor SoC
  Data block size: 2048 MB
  Test duration:   60 seconds per sub-test
  CPU threads:     8
================================================================

[TEST 1] CPU-Only DDR Bandwidth
================================================================
  [CPU] Allocating 2048 MB total (8 threads x 256 MB)
  [CPU] Running READ test for 60 seconds...
  [CPU] READ bandwidth:  153.45 GB/s
  [CPU] Running WRITE test for 60 seconds...
  [CPU] WRITE bandwidth: 240.03 GB/s
  [CPU] Running COPY test for 60 seconds...
  [CPU] COPY bandwidth:  190.51 GB/s

[TEST 2] GPU-Only DDR Bandwidth
================================================================
  [GPU] Device: Thor
  [GPU] Compute capability: 10.1
  [GPU] Allocating 2 x 2048 MB on device
  [GPU] L2 cache size: 24576 KB, SMs: 14, grid: 448 blocks x 256 threads
  [GPU] Verifying kernel execution...
  [GPU] Kernel verification PASSED.
  [GPU] Running READ test for 60 seconds (kernel: read+transform+write)...
  [GPU] READ bandwidth:  XXX.XX GB/s (XXXX iterations, X.XX ms/iter)
  [GPU] Running WRITE test for 60 seconds (cudaMemset)...
  [GPU] WRITE bandwidth: 240.83 GB/s (7225 iterations, 8.30 ms/iter)
  [GPU] Running COPY test for 60 seconds (cudaMemcpy D2D)...
  [GPU] COPY bandwidth:  239.78 GB/s (3597 iterations, 16.68 ms/iter)

[TEST 3] CPU+GPU Mixed DDR Bandwidth
================================================================
  [CPU] READ:   XXX.XX GB/s
  [GPU] READ:   XXX.XX GB/s
  [TOTAL] READ: XXX.XX GB/s (combined)
  ...
================================================================
  Test completed.
================================================================
```

测试结果日志自动保存至 `results/` 目录。

## 已确认的测试结果

基于 Thor 平台实测（使用 `cudaMemset` 和 `cudaMemcpy D2D`）：

| 测试项 | 带宽 | 每轮耗时 |
|--------|------|----------|
| GPU WRITE (cudaMemset, 2GB) | ~240 GB/s | ~8.3 ms |
| GPU COPY (cudaMemcpy D2D, 2x2GB) | ~240 GB/s | ~16.7 ms |
| CPU READ (NEON) | ~153 GB/s | - |
| CPU WRITE (NEON) | ~240 GB/s | - |
| CPU COPY (NEON) | ~190 GB/s | - |

Thor SoC DDR 带宽峰值约 **240 GB/s**。

## 预期带宽范围

Thor SoC 使用 LPDDR5x 内存，理论 DDR 带宽峰值约 200-270 GB/s。预期测试结果：

- CPU-Only：150-240 GB/s
- GPU-Only：150-240 GB/s
- Mixed 总计：受限于 DDR 总线共享，不会超过理论峰值

## 故障排查

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| "PTX JIT compiler library not found" | GPU 架构不匹配 | 用 `-gencode arch=compute_101,code=sm_101` 重新编译 |
| "no kernel image is available" | SASS 代码与 GPU 不匹配 | 确认 compute capability 后重新编译 |
| Kernel verification FAILED | Kernel 未执行 | 检查 GPU 架构是否正确 |
| GPU READ 带宽超过 1000 GB/s | Kernel 静默失败，未实际访问内存 | 检查 `cudaGetLastError()` 输出 |
| 内存分配失败 | 系统内存不足 | 需要至少 8GB 可用内存 |

## 注意事项

- Thor SoC GPU 架构为 **sm_101**（Blackwell, Compute Capability 10.1），编译必须匹配
- 测试需要足够的系统内存（建议 >= 8GB 可用内存）
- 运行全部测试预计耗时约 9 分钟（3 模式 x 3 子测试 x 60 秒）
- 建议以 root 或具有 GPU 访问权限的用户运行
- 交叉编译需要 x86_64 主机上安装 CUDA 12.8+ 及 aarch64 交叉编译工具链
- 程序运行时会打印 Compute Capability，可用于确认实际 GPU 架构
