---
title: "从一篇 NVIDIA 入门博客到 CUDA 核心概念:一次学习笔记"
date: 2026-07-21T12:00:00+08:00
draft: false
tags: ["CUDA", "GPU", "并行计算", "NVIDIA"]
---

> 以 NVIDIA 的 *An Even Easier Introduction to CUDA* 为线索,梳理 CPU/GPU 异构架构、CUDA 编程模型、线程组织(thread/block/grid/warp)、GPU 硬件规格,以及统一内存与数据预取的性能要点。本文是一次边读边问的学习总结。

## 📚 目录

- [1. CPU 与 GPU:异构的两个处理器](#1-cpu-与-gpu异构的两个处理器)
- [2. 什么是 CUDA](#2-什么是-cuda)
- [3. host 与 device 如何交互](#3-host-与-device-如何交互)
- [4. 线程的三层组织:grid / block / thread](#4-线程的三层组织grid--block--thread)
- [5. warp:硬件真正的调度单位](#5-warp硬件真正的调度单位)
- [6. GPU 硬件规格速查](#6-gpu-硬件规格速查)
- [7. 统一内存 vs 手写 cudaMemcpy](#7-统一内存-vs-手写-cudamemcpy)
- [8. 为什么统一内存需要配合 prefetch](#8-为什么统一内存需要配合-prefetch)
- [9. 完整示例:cudaMallocManaged + prefetch](#9-完整示例cudamallocmanaged--prefetch)
- [10. 关键收获](#10-关键收获)

---

## 1. CPU 与 GPU:异构的两个处理器

CPU 和 GPU **不是同一个架构**,而是两个独立的处理器,各有不同的设计取向:

- **CPU(主机 / Host)**:少量强大的核心(通常几个到几十个),擅长复杂的串行逻辑、分支判断、低延迟任务。
- **GPU(设备 / Device)**:成千上万个较简单的核心,擅长大规模并行的相同运算。

在 CUDA 术语里,CPU 侧叫 **host**,GPU 侧叫 **device**。这也是代码里 `h_`(host)、`d_`(device)变量前缀命名习惯的由来。

一个常见但需要修正的直觉:能不能把 device 理解成 host 的"外挂"?

- **对的部分**:GPU 确实从属于 CPU——程序总是从 CPU 开始,由 CPU 决定何时把哪段计算交给 GPU;GPU 是可选的加速器。
- **需要修正的部分**:它不是软件插件,而是**独立硬件 + 独立内存**;它也不是"更快的 CPU",而是"不同类型的处理器"——单个 GPU 核心其实比 CPU 核心弱,强大来自"数量取胜"。

> 一个形象的比喻:CPU 像几个博士生,聪明、能处理复杂问题但人少;GPU 像一千个小学生,每人只会简单加减法,但一千道简单题同时开做,总量上碾压。老板(你的程序)先让博士生统筹,遇到"一万道算术题"就打包丢给这一千个小学生一起算,算完再收回结果。

所以更准确的说法是:**device 是 host 指挥下的大规模并行协处理器。**

---

## 2. 什么是 CUDA

**CUDA(Compute Unified Device Architecture)** 是 NVIDIA 的并行计算平台和编程模型,让你用类似 C/C++ 的语言编写在 GPU 上运行的程序。它不只是一门语言,而是分层的一整套东西:

- **编程模型**:定义 host/device、thread/block/grid 抽象,以及 `__global__`、`<<<>>>` 等语法。
- **编译器 `nvcc`**:把代码拆两部分——CPU 部分交给普通 C++ 编译器,GPU 部分编译成 GPU 指令(PTX / SASS)。
- **运行时与驱动 API**:`cudaMalloc`、`cudaMemcpy`、`cudaDeviceSynchronize` 等,管理显存、启动核函数、做同步。
- **加速库**:cuBLAS、cuDNN、cuFFT 等。PyTorch / TensorFlow 之所以能用 GPU 训练,底层就是它们。

三种函数修饰符决定了"代码在哪运行、从哪调用":

| 修饰符 | 在哪运行 | 从哪调用 | 用途 |
|--------|---------|---------|------|
| `__global__` | GPU (device) | CPU (host) | **核函数 kernel**,CPU 进入 GPU 的入口 |
| `__device__` | GPU | GPU | GPU 内部调用的辅助函数 |
| `__host__` | CPU | CPU | 普通 CPU 函数(默认,可省略) |

把普通函数变成 GPU 核函数,只需加上 `__global__`:

```cpp
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}
```

---

## 3. host 与 device 如何交互

传统独立显卡下,CPU 与 GPU 各有独立内存,通过 **PCIe 总线**连接:

```
┌─────────────┐         PCIe 总线          ┌─────────────┐
│    CPU      │  <──────────────────────>  │    GPU      │
│  (Host)     │                            │  (Device)   │
├─────────────┤                            ├─────────────┤
│  系统内存    │                            │  显存        │
│  (RAM/DRAM) │                            │  (VRAM)     │
└─────────────┘                            └─────────────┘
```

一次典型协作的流程:

1. **分配内存**:在显存分配空间(`cudaMalloc`),或用统一内存(`cudaMallocManaged`)。
2. **拷贝数据 Host → Device**:把输入从 RAM 传到显存(`cudaMemcpy`)。
3. **启动核函数**:CPU 用 `kernel<<<numBlocks, blockSize>>>(...)` 调用 `__global__` 函数。`<<<...>>>` 是 CUDA 特有的**执行配置语法**。
4. **异步执行**:核函数启动后 CPU **不会等待**,会继续往下跑,需要 `cudaDeviceSynchronize()` 让 CPU 等 GPU 算完。
5. **拷贝结果 Device → Host**:结果从显存传回 RAM。
6. **释放内存**:`cudaFree`。

```cpp
add<<<1, 256>>>(N, x, y);   // CPU 发起,GPU 执行
cudaDeviceSynchronize();     // CPU 等 GPU 完成
```

---

## 4. 线程的三层组织:grid / block / thread

GPU 上的线程不是一盘散沙,而是被组织成三层:

```
Grid(网格)  = 一次核函数启动的全部线程
 └── Block(线程块) 若干个
      └── Thread(线程) 若干个
```

对应启动语法 `kernel<<<numBlocks, blockSize>>>`:

- `numBlocks` = grid 里有多少个 block
- `blockSize` = 每个 block 里有多少个 thread
- 总线程数 = `numBlocks × blockSize`

**为什么要分两层?** 这是硬件决定的:

- 同一个 **block** 内的线程被分配到同一个 **SM(流多处理器)**,它们能共享高速 shared memory、能互相同步(`__syncthreads()`)。
- 不同 **block** 之间基本独立、不能直接同步,这样 GPU 才能自由地把成千上万个 block 分派到几十个 SM 上,天然可扩展。

**每个线程怎么知道自己该算哪个数据?** 靠内置变量算出全局唯一索引:

| 变量 | 含义 |
|------|------|
| `threadIdx.x` | 我在自己 block 内的编号 |
| `blockIdx.x` | 我所在 block 在 grid 里的编号 |
| `blockDim.x` | 每个 block 的线程数 |
| `gridDim.x` | grid 里的 block 数 |

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;  // idiomatic CUDA
```

这就像算电影院座位号:`(前面几排 × 每排座位数) + 本排第几个`。

博客里最终的 **grid-stride loop** 版本,让代码不管启动多少线程都正确:

```cpp
__global__
void add(int n, float *x, float *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   // 总线程数
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
```

块数用向上取整计算,并在 kernel 里用 `if (index < n)`(或 stride 循环)挡住多余线程:

```cpp
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;   // 向上取整
add<<<numBlocks, blockSize>>>(N, x, y);
```

博客里三次演进的性能对比(N = 1M,T4 GPU,数据已就位):

| 版本 | 执行配置 | 线程数 | 相对加速 |
|------|---------|--------|---------|
| 单线程 | `<<<1, 1>>>` | 1 | 1x |
| 单块多线程 | `<<<1, 256>>>` | 256 | ~45x |
| 多块 | `<<<numBlocks, 256>>>` | ~100 万 | ~1932x |

---

## 5. warp:硬件真正的调度单位

**warp(线程束)是 GPU 调度和执行线程的最小单位,固定为 32 个线程。**

- **thread** 是你写代码时的逻辑单位。
- **warp(32 线程)** 才是硬件真正调度和执行的物理单位。

### SIMT(单指令多线程)

一个 warp 里的 32 个线程,在同一时刻执行**同一条指令**,只是各自作用在不同数据上。以向量加法为例:32 个线程同时执行 `y[i] = x[i] + y[i]`,只是 i 各不相同。指令只取一次、译码一次,却驱动 32 份数据运算——这是 GPU 高效的根源。

### 为什么你需要关心 warp

1. **block 大小要选 32 的倍数**。设 `blockSize = 100` 会被凑成 128 线程(4 个 warp),多出的 28 个线程空转浪费。博客选 256 = 8 个整齐的 warp。
2. **warp divergence(束内分支发散)是性能杀手**。若同一 warp 内线程走了不同的 `if/else` 分支,硬件只能让两个分支**串行执行**,损失性能。
3. **访存合并(coalescing)**。一个 warp 的 32 个线程访问连续地址时,硬件能合并成一次大内存事务,带宽利用率高。

### warp 与延迟隐藏

SM 里驻留大量 warp,调度器在它们之间**近乎零开销地快速切换**:某个 warp 卡在等内存时,立刻切到另一个就绪的 warp 执行。GPU 就是靠"海量 warp 轮换"把内存延迟藏起来的。

> 32 线程 = 1 warp。所以 2048 驻留线程 = 64 个 warp。

---

## 6. GPU 硬件规格速查

**一个 CPU 通常连几个 GPU?** 没有定值:

- 个人 / 工作站:1~2 个 GPU。
- 标准服务器节点:**4 或 8 个 GPU** 最主流(NVIDIA DGX / HGX 就是经典的 8 GPU/节点)。
- 上限主要受 **PCIe 通道数**限制,想接更多要靠 PCIe Switch。8-GPU 服务器里,GPU 之间还常用 **NVLink / NVSwitch** 直连,带宽远高于 PCIe。

**一个 GPU 最多有多少 SM?** 随架构逐代增长:

| GPU | 架构 | SM 数 |
|-----|------|-------|
| T4 | Turing | 40 |
| A100 | Ampere | 108(完整芯片 128) |
| H100 | Hopper | 132(完整 GH100 144) |

目前单颗 GPU 的 SM 数大致在几十到 150 的量级(厂商常屏蔽少量缺陷 SM 以提高良率)。

**每个 SM 最多多少线程?** 要区分两个上限:

- **每个 SM 最多驻留线程数**:Turing(如 T4)是 **1024**,Volta / Ampere / Hopper 是 **2048**。
- **每个 block 最多线程数**:**1024**(几乎所有现代架构的硬上限)。

关键澄清:"驻留 2048 线程" ≠ "2048 线程同时在算"。SM 里的执行单元远少于 2048,它靠**超额登记(oversubscription)**——让大量线程驻留,一批等内存时立刻切到另一批,用这种方式掩盖访存延迟。

---

## 7. 统一内存 vs 手写 cudaMemcpy

两者是管理 host/device 内存的两种方式,解决同一个问题:数据怎么在 RAM 和 VRAM 之间到位。

### 手写 cudaMemcpy(显式)

```cpp
float *h_x = new float[N];              // host 内存
float *d_x;
cudaMalloc(&d_x, N*sizeof(float));      // device 内存,单独分配

cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);  // 手动搬过去
add<<<blocks, threads>>>(N, d_x, ...);
cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);  // 手动搬回来

cudaFree(d_x);
delete[] h_x;
```

**两个指针**、传输时机与方向全程手控:啰嗦易错,但性能可控可预测。

### 统一内存 cudaMallocManaged(自动)

```cpp
float *x;
cudaMallocManaged(&x, N*sizeof(float));  // 一个指针,CPU/GPU 通用

add<<<blocks, threads>>>(N, x, ...);     // GPU 直接用,不用 memcpy
cudaDeviceSynchronize();

cudaFree(x);
```

**一个指针**、数据按需自动迁移:代码简洁、心智负担低。

### 对比

| 维度 | 手写 cudaMemcpy | 统一内存 cudaMallocManaged |
|------|----------------|--------------------------|
| 指针数量 | 两个(h_ / d_) | 一个 |
| 数据搬运 | 手动、显式 | 自动、按需迁移 |
| 代码复杂度 | 高 | 低 |
| 性能可预测性 | 高 | 需配合 prefetch 才达同等性能 |
| 适合场景 | 追求极致性能、老代码 | 快速开发、原型、复杂数据结构 |

**关键点:统一内存不是"更快",而是"更省心"。** 底层数据仍要在 RAM↔VRAM 移动,只是从"你手写 memcpy"变成"运行时替你搬"。

---

## 8. 为什么统一内存需要配合 prefetch

统一内存是**按需迁移的虚拟内存**:同一份数据的物理页在某一时刻只驻留在一个设备的物理内存里,谁要访问又不在本地,就触发**缺页(page fault)**,硬件才把那一页搬过来。

**不做 prefetch 时会发生什么(博客的场景):**

1. CPU 用 `for` 循环初始化数组 → 所有页都在 RAM。
2. kernel 启动,GPU 上成千上万线程访问数据 → 数据全不在显存。
3. 爆发**大量缺页**,硬件一页一页地从 RAM 迁到 VRAM。

问题就在"一页一页、按需触发":迁移零散、每次缺页都有固定开销(通信 + 更新页表 + 启动 DMA)、GPU 线程被迫 stall。这正是博客里"多块反而没加速"的原因——瓶颈从计算变成了缺页驱动的零散内存迁移。

**prefetch 做了什么:** 在 kernel 启动**之前**,主动把整块数据一次性搬到 GPU。

| | 无 prefetch(缺页驱动) | 有 prefetch(批量迁移) |
|---|---|---|
| 迁移时机 | kernel 运行中,被动触发 | kernel 启动前,主动完成 |
| 迁移粒度 | 一次一页,上千次 | 一次一大块 |
| 固定开销 | 付上千次 | 付一次 |
| GPU 是否 stall | 频繁等待 | 数据已就位,不等 |

博客里 kernel 因此从 4.5ms 掉到 **47 微秒**。

**为什么说"才能达到同等性能"?** 手写 `cudaMemcpy` 本来就是一次性批量搬运——本质是"手动的 prefetch"。所以:

- 手写 cudaMemcpy = 天生批量,快但啰嗦。
- 统一内存 + 无 prefetch = 省心,但零散缺页导致慢。
- 统一内存 + prefetch = 既省心又批量 → 追平手写 memcpy。

> 比喻:cudaMemcpy 是你自己开车送货,累但路线你定;cudaMallocManaged 是叫代驾,省事但默认可能绕远;prefetch 就是出发前把全程路线给代驾,一口气开到。

---

## 9. 完整示例:cudaMallocManaged + prefetch

基于博客的向量加法,加上双向 prefetch 和精确的 event 计时:

```cpp
#include <iostream>
#include <math.h>

// 核函数:grid-stride loop 版本
__global__
void add(int n, float *x, float *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;   // 1M 个元素

    // 1. 拿到当前 GPU 的编号(prefetch 的目标设备)
    int device = -1;
    cudaGetDevice(&device);

    // 2. 用统一内存分配:一个指针,CPU/GPU 通用
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 3. 在 CPU 上初始化 —— 此刻数据页都驻留在 CPU 内存
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 4. 关键:kernel 启动前,把 x、y 一次性预取到 GPU
    cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL);

    // 5. 用 CUDA event 给 kernel 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(stop);

    // 6. 等 GPU 算完
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;

    // 7. 结果要回 CPU 检查,顺手把 y 预取回 CPU
    //    cudaCpuDeviceId 是表示 CPU 的特殊设备号
    cudaMemPrefetchAsync(y, N * sizeof(float), cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize();

    // 8. 检查结果:全部应为 3.0
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // 9. 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);

    return 0;
}
```

编译运行:

```bash
nvcc add_prefetch.cu -o add_prefetch
./add_prefetch
```

几个要点:

- **`cudaGetDevice`**:prefetch 需要知道搬到哪个设备。GPU 用其编号,CPU 用特殊常量 `cudaCpuDeviceId`。
- **双向 prefetch**:算前 H2D 把输入搬到 GPU,算后 D2H 把结果搬回 CPU,避免回读时又触发一堆缺页。很多入门例子只做前半段。
- **用 event 计时**:kernel 启动是异步的,用 CPU 计时器只会测到"启动指令"的时间;`cudaEvent` 记录的是 GPU 时间线上的时刻,才是准确的 kernel 耗时。

**想直观看到 prefetch 的威力**:把第 4 步两行 prefetch 注释掉再跑,kernel 时间会涨到几毫秒。用 `nsys profile -t cuda --stats=true ./add_prefetch` 对比 `memcpy Unified H2D` 行,能看到"零散多次"变成"批量几次"。

> 提示:`cudaMemPrefetchAsync` 依赖统一内存的按需迁移能力,在 Linux + 支持的 GPU 上工作良好;Windows(WDDM 驱动)对统一内存迁移的支持有限制,行为可能不同。

---

## 10. 关键收获

把这次学习串成一条线:

1. **CPU / GPU 是异构、各有独立内存** → 所以有 host / device 之分,数据要搬。
2. **CUDA 是指挥 GPU 干活的整套平台** → `__global__`、`<<<>>>`、`cuda*` API 都是它的一部分。
3. **device 是 host 指挥下的大规模并行协处理器** → thread / block / grid / warp 就是组织并行工的方式,warp(32 线程)是硬件真正的调度单位。
4. **真正的性能瓶颈常在内存搬运** → 向量加法这类是**带宽受限(bandwidth-bound)**的,统一内存 + 预取就是应对手段。

最深的一课:

> **GPU 编程里,"算得快"往往不是难点,"喂数据喂得够快"才是。**

这也解释了为什么真实的深度学习框架里,数据布局、内存拷贝、H2D/D2H 的重叠(overlap)是核心优化点。

---

## 参考资料

- [An Even Easier Introduction to CUDA (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [Unified Memory in CUDA 6](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---

*最后更新时间:2026 年 7 月 21 日*


> 💡 本文整理自一次以 NVIDIA 入门博客为起点的边读边问式学习,如有理解偏差欢迎指正。
