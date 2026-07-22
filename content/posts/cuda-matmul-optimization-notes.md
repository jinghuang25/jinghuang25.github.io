---
title: "CUDA Matmul 优化学习笔记"
date: 2026-07-22T15:42:00+08:00
draft: false
tags: ["CUDA", "GPU", "Matmul", "Performance"]
---

> 基于 Simon Boehm 的《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》整理的中文学习笔记。
> 本文是个人理解性总结与讲解，非原文翻译。

## 目录

- [0. 要解决的问题](#0-要解决的问题)
- [1. 术语表（含完整英文名）](#1-术语表含完整英文名)
- [2. GPU 内存层级](#2-gpu-内存层级)
- [3. Host / Device 与 PTX](#3-host--device-与-ptx)
- [4. 分块参数 BM/BN/BK/TM/TN](#4-分块参数-bmbnbktmtn)
- [5. 十个 Kernel 逐步优化](#5-十个-kernel-逐步优化)
- [6. 为什么要自己写而不直接用 cuBLAS](#6-为什么要自己写而不直接用-cublas)
- [7. 一句话总结](#7-一句话总结)
- [8. 附：Shared Memory Bank Conflict 原理](#8-附shared-memory-bank-conflict-原理)

---

## 0. 要解决的问题

实现 **SGEMM**（单精度通用矩阵乘法）：

```
C = α·(A·B) + β·C
```

- `A` 是 M×K，`B` 是 K×N，`C` 是 M×N，全部 FP32（32 位浮点）。
- 测试用大方阵（如 4096×4096）。
- 朴素实现只有 cuBLAS 的约 1.3% 性能；通过约 10 个逐步优化的 kernel，爬到约 93.7%。

**核心矛盾**：矩阵乘理论上是"算力受限（compute-bound）"，但朴素写法实际是"访存受限（memory-bound）"——瓶颈在数据搬运，不在计算。整套优化就是想办法**喂饱计算单元**。

---

## 1. 术语表（含完整英文名）

### 硬件与内存

| 缩写/术语 | 完整英文 | 中文解释 |
|---|---|---|
| GPU | Graphics Processing Unit | 图形处理器 |
| SM | Streaming Multiprocessor | 流式多处理器，GPU 里执行计算的核心单元 |
| HBM | High Bandwidth Memory | 高带宽显存 |
| DRAM | Dynamic Random-Access Memory | 动态随机存储器（显存物理载体） |
| GMEM | Global Memory | 全局内存，所有线程可访问，容量大但慢 |
| L2 Cache | Level 2 Cache | 二级缓存，多个 SM 共享 |
| SMEM | Shared Memory | 共享内存，片上，block 内共享，快但小 |
| RMEM | Registers | 寄存器，线程私有，最快 |
| Latency | Latency | 延迟（等待周期数） |
| Bandwidth | Bandwidth | 带宽（单位时间搬运量） |

### 执行模型

| 术语 | 完整英文 | 中文解释 |
|---|---|---|
| Thread | Thread | 线程，最小执行单位 |
| Warp | Warp | 32 个线程组成的硬件调度单位 |
| Block | Thread Block | 线程块，映射到一个 SM 执行 |
| Grid | Grid | 所有 block 的集合 |
| Occupancy | Occupancy | 占用率 = 活跃 warp 数 / 最大可能 warp 数 |
| Kernel | Kernel | 在 GPU 上运行的函数 |
| Barrier | `__syncthreads()` | 线程块同步屏障 |

### 优化技术

| 术语 | 完整英文 | 中文解释 |
|---|---|---|
| Coalescing | Global Memory Coalescing | 全局内存合并访问 |
| Tiling | Cache Blocking / Tiling | 分块，把数据搬进快速内存复用 |
| Blocktiling | 1D / 2D Blocktiling | 块分块，一个线程算多个输出 |
| Warptiling | Warptiling | warp 级分块 |
| Arithmetic Intensity | Arithmetic Intensity | 算术强度 = FLOP / Byte |
| Vectorized Access | Vectorized Memory Access (float4) | 向量化访存，一条指令搬 128 位 |
| Autotuning | Autotuning | 自动调参 |
| Bank Conflict | Shared Memory Bank Conflict | 共享内存 bank 冲突 |
| Register Spilling | Register Spilling | 寄存器溢出到慢速 local memory |
| Outer Product | Outer Product | 外积（列向量 × 行向量） |

### 性能与库

| 术语 | 完整英文 | 中文解释 |
|---|---|---|
| FLOP | Floating-Point Operation | 浮点运算次数（工作量） |
| FLOPS | Floating-Point Operations per Second | 每秒浮点运算次数（速度） |
| GEMM | General Matrix Multiply | 通用矩阵乘法 C = α·A·B + β·C |
| SGEMM | Single-precision GEMM | 单精度矩阵乘（S=Single，还有 D/C/Z/H） |
| BLAS | Basic Linear Algebra Subprograms | 基础线性代数子程序库 |
| cuBLAS | CUDA Basic Linear Algebra Subprograms | NVIDIA 官方 GPU 线性代数库（性能标杆） |
| PTX | Parallel Thread Execution | NVIDIA 的虚拟指令集 / 中间表示 |
| SASS | Streaming ASSembler | 绑定具体架构的真实机器码 |
| Compute-bound | Compute-bound | 算力受限 |
| Memory-bound | Memory-bound | 访存受限 |
| Roofline | Roofline Model | 用算术强度判断瓶颈的模型 |

> BLAS 精度前缀：**S**=单精度FP32，**D**=双精度FP64，**C**=单精度复数，**Z**=双精度复数，**H**=半精度FP16。

---

## 2. GPU 内存层级

从大到小、从慢到快，这是整套优化的舞台：

| 层级 | 容量 | 速度 | 可见范围 |
|---|---|---|---|
| Global Memory (GMEM) | 几~几十 GB | 最慢（几百周期延迟） | 所有线程 |
| L2 Cache | MB 级 | 中 | 所有 SM 共享 |
| Shared Memory (SMEM) | 几十 KB / block | 快（个位数周期） | block 内共享 |
| Registers | 极小 / 线程 | 最快 | 单线程私有 |

**优化的本质**：把数据一层层往快的地方搬，然后反复复用。
形成一座"**复用金字塔**"：

```
GMEM（大而慢）
  → SMEM（block tile 复用）
    → 寄存器（warp tile → thread tile 复用）
```

---

## 3. Host / Device 与 PTX

### Host vs Device

- **Host（主机）** = CPU + 系统 RAM，跑普通 C++ 代码。
- **Device（设备）** = GPU + 显存，kernel 在这里运行，Global Memory 属于 device。
- kernel 只能直接访问 device memory，**不能直接读 host 的 RAM**。

数据流动：

```
Host RAM ──cudaMemcpy(HtoD)──► Device GMEM ──► kernel 计算 ──► Device GMEM ──cudaMemcpy(DtoH)──► Host RAM
```

传输走 PCIe（或 NVLink），带宽远低于显存内部，所以尽量少来回传。

CUDA 函数修饰符：
- `__host__`：CPU 上运行（默认）
- `__global__`：host 调用、device 运行的 kernel 入口
- `__device__`：device 上运行、被 device 代码调用

### PTX = Parallel Thread Execution

NVIDIA 的**虚拟指令集 / 中间表示**，编译链路：

```
CUDA C++ ──(nvcc 前端)──► PTX（虚拟汇编，跨架构） ──(ptxas/驱动JIT)──► SASS（真实机器码，绑定架构）
```

- **PTX** 面向抽象 GPU，向前兼容（新卡可 JIT 编译旧 PTX）。
- **SASS** 才是硬件真正执行的机器码，绑定架构（sm_80=Ampere、sm_90=Hopper）。
- 高手看 PTX/SASS 来确认编译器是否生成了向量化加载（如 `ld.global.v4.f32`）、有没有寄存器溢出。
- 类比：PTX 之于 GPU ≈ Java 字节码之于 JVM。

---

## 4. 分块参数 BM/BN/BK/TM/TN

命名规律：**B = Block（块）**，**T = Thread（线程）**；**M/N/K = 矩阵维度**（A 是 M×K，B 是 K×N，C 是 M×N）。

| 参数 | 全称 | 通俗意思 |
|---|---|---|
| BM | Block tile 在 M 方向 | 一个 block 负责 C 里多少**行** |
| BN | Block tile 在 N 方向 | 一个 block 负责 C 里多少**列** |
| BK | Block tile 在 K 方向 | 每次搬进 SMEM 的"切片厚度"（K 维一次处理多长） |
| TM | Thread tile 在 M 方向 | 一个线程负责多少**行**输出 |
| TN | Thread tile 在 N 方向 | 一个线程负责多少**列**输出 |

**贴瓷砖比喻**：

```
整面墙 = 矩阵 C (M×N)
  大瓷砖 = 每个 block 负责 (BM × BN)
    小瓷砖 = 每个线程负责 (TM × TN)
  BK = 深度方向的切片厚度（点积沿 K 累加，切段处理）
```

**互相约束**（所以要 autotuning）：
- 一个 block 的线程数 = `(BM×BN) / (TM×TN)`
- BM/BN 大 → 复用多，但 SMEM/线程数吃紧
- TM/TN 大 → 算术强度高，但寄存器用量涨 → 占用率降、可能溢出
- BK 大 → 一次算的深度多，但 SMEM tile 更占地方

---

## 5. 十个 Kernel 逐步优化

### 性能对比表（RTX 3090 级别，FP32 SGEMM）

| Kernel | GFLOPs/s | 相对 cuBLAS | 主要手段 |
|---|---|---|---|
| 1: Naive | 309.0 | 1.3% | 一线程一输出 |
| 2: GMEM Coalescing | 1986.5 | 8.5% | 改索引映射，warp 读连续地址 |
| 3: SMEM Caching | 2980.3 | 12.8% | tile 搬进共享内存复用 |
| 4: 1D Blocktiling | 8474.7 | 36.5% | 一线程算一列，寄存器复用 |
| 5: 2D Blocktiling | 15971.7 | 68.7% | 一线程算方块，外积复用 |
| 6: Vectorized Mem Access | 18237.3 | 78.4% | float4，128 位加载 |
| 9: Autotuning | 19721.0 | 84.8% | 暴力搜索最佳参数 |
| 10: Warptiling | 21779.3 | 93.7% | 加 warp 级分块层 |
| 0: cuBLAS | 23249.6 | 100.0% | 官方库（天花板） |

> 编号跳过 7、8：它们是通向自动调参的过渡版本（结构重构、修 bank conflict 等），性能提升不明显，未列入对比表。

### Kernel 1：Naive（朴素）

每个线程算 C 里一个元素 `C[x][y]`：

```cpp
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {                    // 边界检查：M/N 不是 32 倍数时必需
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * N + y];  // 点积，一维索引表示二维矩阵
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

- `blockIdx*blockDim+threadIdx`：把"第几块×每块大小+块内第几个"拼成全局唯一坐标。
- 慢的原因：**零复用**（相邻线程重复读相同数据）+ **访存不合并**。

### Kernel 2：GMEM Coalescing（合并访问）

只改 x/y 的映射，让 warp 内连续线程对应 C 同一行的相邻列：

```cpp
const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // 行：warp 内相同
const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // 列：warp 内连续
```

- warp 内 32 线程读 B 的 32 个**连续地址** → 硬件合并成 1 次宽事务；读 A 是同一地址 → 广播。
- **计算量没变**，纯靠访问模式贴合硬件搬运粒度，提速 6 倍。ROI 最高的一步。

### Kernel 3：SMEM Caching（共享内存分块）

一个 block 协作把 A、B 的 tile 搬进 SMEM，反复复用：

```cpp
__shared__ float As[BLOCKSIZE*BLOCKSIZE], Bs[BLOCKSIZE*BLOCKSIZE];
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
  As[...] = A[...]; Bs[...] = B[...];   // ① 协作加载（每线程搬一个，合并访问）
  __syncthreads();                       // ② 屏障：确保 tile 全部就位
  for (int d = 0; d < BLOCKSIZE; ++d)    // ③ 从 SMEM 做部分点积
    tmp += As[...] * Bs[...];
  __syncthreads();                       // ④ 屏障：算完再进下一段
}
```

- `__syncthreads()` 两道屏障：加载后（防读到未就位数据）、计算后（防提前覆盖）。
- 瓶颈：每线程仍只算 1 个输出，**算术强度低**。

### Kernel 4：1D Blocktiling（一维分块）

一个线程算连续一列 TM 个输出，靠寄存器复用 B 值：

```cpp
float threadResults[TM] = {0.0};
for (...bkIdx...) {
  ...load As, Bs into SMEM...
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    float tmpB = Bs[dotIdx*BN + threadCol];    // 读一次 B 值进寄存器
    for (uint r = 0; r < TM; ++r)              // 复用 TM 次
      threadResults[r] += As[...] * tmpB;
  }
}
```

- **读一次 B，用 TM 次** → 算术强度 ×TM。提速近 3 倍。
- 代价：寄存器用量上升 → 占用率下降。

### Kernel 5：2D Blocktiling（二维分块）

一个线程算 TM×TN 的方块，用**外积**双向复用：

```cpp
float threadResults[TM*TN] = {0.0};
float regM[TM], regN[TN];
for (...bkIdx...) {
  ...load As, Bs into SMEM...
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (i<TM) regM[i] = As[...];      // 一列 A 进寄存器
    for (i<TN) regN[i] = Bs[...];      // 一行 B 进寄存器
    for (m<TM) for (n<TN)              // 外积：读 TM+TN 个，算 TM×TN 次
      threadResults[m*TN+n] += regM[m] * regN[n];
  }
}
```

- 读 16 个（8+8）值，做 64 次（8×8）乘加 → 计算/访存比从 ~0.9 飙到 4.0。
- 复用金字塔三层成型：GMEM → SMEM → 寄存器。到 68.7%。

### Kernel 6：Vectorized Memory Access（向量化访存）

用 `float4`（128 位）一次搬 4 个 float：

```cpp
float4 tmp = reinterpret_cast<const float4*>(&A[...])[0];  // 一条指令读 4 个
```

- 访存指令数减少 4 倍，带宽利用更充分。
- 附带：加载 A 时**转置**存进 SMEM，保持向量化 + 避免 bank conflict。
- 要求地址 16 字节对齐（所以分块尺寸取 4 的倍数）。计算逻辑不变。

### Kernel 7、8：过渡版本（未列入对比表）

为自动调参做结构准备：把加载逻辑重构成可参数化形式、处理 SMEM bank conflict 等。性能提升不明显，作者未单列。

### Kernel 9：Autotuning（自动调参）

把 BM/BN/BK/TM/TN 写成编译期模板参数，生成大量合法组合，在目标 GPU 上逐个实测计时，选最快的：

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm(...);
```

- 不改算法，纯靠"为这块 GPU 挑对参数"拿到 6 个百分点 → 84.8%。
- CUTLASS、Triton 等库大量依赖 autotiling，因为没有一组参数对所有 GPU/shape 都最优。

### Kernel 10：Warptiling（warp 级分块）

在 block 和 thread 之间插入一层 warp tile，匹配硬件真实调度单位：

```
Block Tile (BM×BN, SMEM)
  └─ Warp Tile (WM×WN, 映射到硬件 warp)
       └─ Thread Tile (TM×TN, 寄存器)
```

- warp 是指令发射、访存、Tensor Core 运算的实际粒度。
- 好处：匹配硬件调度、进一步减少 bank conflict、为 Tensor Core（其 MMA 指令本身是 warp 级）铺路。
- 到 93.7%，逼近 cuBLAS。剩余差距来自 cuBLAS 的手写 SASS、Tensor Core、极致 shape 调优。

---

## 6. 为什么要自己写而不直接用 cuBLAS

对绝大多数应用开发者：**直接调 cuBLAS 就对了**，别重复造轮子。但学写这个 kernel 的意义在别处：

1. **cuBLAS 是黑盒，很多场景不够用**
   - 算子融合（Operator Fusion）：`矩阵乘 + bias + 激活` 融成一个 kernel，中间结果留在寄存器/SMEM，省掉大量 GMEM 往返。FlashAttention 就是这个思路。
   - 非标准数据类型/量化（INT8、FP8、稀疏、自定义 layout）。
   - 特殊 shape（很扁的 GEMM 等），cuBLAS 未必最优。
2. **学到 GPU 性能优化的通用方法论**：内存层级复用、合并访存、向量化、占用率/寄存器权衡、算术强度、roofline —— 适用于任何 kernel。
3. **造轮子是为了看懂别人的轮子**：会写 → 会读 → 会调 → 会判断该不该自己写。
4. **cuBLAS 也不是永远最优**：新硬件支持滞后、边缘 shape/精度上自研 kernel 有时能赢。框架团队（PyTorch/TVM/Triton）需要这种能力。

> 定位：cuBLAS 是**起点不是终点**。做 AI 基础设施 / 算子开发 / 推理优化的人，需要在它不适用处写出自己的高性能 kernel。

---

## 7. 一句话总结

整个系列的主线是——**每一步都在解决上一步暴露的瓶颈**：

```
朴素(访存受限) → 合并访存 → SMEM 复用 → 1D 分块(提算术强度)
→ 2D 分块(外积复用) → 向量化 → 自动调参 → warptiling(对齐硬件) → 逼近 cuBLAS
```

最终价值不是那个矩阵乘本身，而是这套"**如何系统性地把 GPU 榨到极限**"的思维框架：
**把数据往快的内存搬 + 让每次访存喂饱更多计算 + 让分块结构匹配硬件调度 + 用调参找到最优平衡点。**

---

## 8. 附：Shared Memory Bank Conflict 原理

### SMEM 被分成 32 个 bank

共享内存为了快速服务一个 warp（32 线程），物理上切成 **32 个独立存储体（bank）**。连续的 4 字节（一个 float）依次落到不同 bank，循环编号：

```
地址(按 float): 0   1   2   3  ... 31  32  33 ...
bank 编号:      0   1   2   3  ... 31   0   1 ...   ← 每 32 个 float 一轮回
公式：bank = (float 地址) % 32
```

### 冲突规则

一个 warp 的 32 线程访问 SMEM 时：
- 32 线程各落在**不同 bank** → 一个周期全部完成（满速）。
- **≥2 线程访问同一 bank 的不同地址** → **bank conflict**，硬件被迫**串行化**。
- n 个线程撞同一 bank = **n-way conflict**，慢 n 倍。
- **例外（不算冲突）**：多线程读**同一 bank 的同一地址** → 硬件 **broadcast（广播）** 一次满足所有线程。

### 典型场景：按列访问（stride = 32）

SMEM 存 32 列的行主序数组 `As[row][col]`（一行 32 个 float）：

```cpp
// 情况 A：按【行】读 —— 无冲突
As[0][threadIdx.x]   // 线程 0..31 → col 0..31 → bank 0..31 全不同 ✅

// 情况 B：按【列】读 —— 32-way 冲突
As[threadIdx.x][0]   // 线程0→地址0→bank0，线程1→地址32→bank0，线程2→地址64→bank0 ...
                     // 每跳一行 +32，32%32=0，整列全撞 bank 0 → 串行 32 次 ❌
```

矩阵乘里最常见的坑：**按列取数天然容易撞 bank**。

### 解决办法

1. **转置存储**：Kernel 6 转置 A 存进 SMEM 的另一动机——把"按列取"变成"按行连续存"（回到情况 A）。
2. **Padding（填充）**：每行多留一个占位元素，行宽 32→33。跳一行 +33，`33%32=1`，整列错开落到不同 bank。代价是浪费一点 SMEM。
   ```cpp
   __shared__ float As[32][33];  // 多填 1 列，破坏 stride=32 对齐
   ```
3. **调整访问模式 / 向量化**：让 warp 内线程地址映射到不同 bank。

### 和 Coalescing 的区别（易混）

| | Coalescing | Bank Conflict |
|---|---|---|
| 发生位置 | Global Memory（全局内存） | Shared Memory（共享内存） |
| 关心什么 | 地址是否**连续** → 能否合并成宽事务 | 地址是否撞**同一 bank** → 是否被迫串行 |
| 想要的模式 | warp 内读**连续**地址 | warp 内读**不同 bank** |

一句话：**coalescing 是全局内存层的"合并"，bank conflict 是共享内存层的"错开"**，属于不同内存层的优化，别搞混。
