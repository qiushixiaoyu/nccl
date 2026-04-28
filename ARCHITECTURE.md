# NCCL 代码仓整体结构分析

> NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 开发的高性能 GPU 间通信库，当前版本：**2.30.3**。

---

## 目录

1. [项目概述](#1-项目概述)
2. [顶层目录结构](#2-顶层目录结构)
3. [核心源码结构（`src/`）](#3-核心源码结构src)
4. [关键架构概念](#4-关键架构概念)
5. [构建系统](#5-构建系统)
6. [插件生态系统（`plugins/`）](#6-插件生态系统plugins)
7. [语言绑定（`bindings/`）](#7-语言绑定bindings)
8. [贡献代码（`contrib/`）](#8-贡献代码contrib)
9. [打包系统（`pkg/`）](#9-打包系统pkg)
10. [数据流与调用链](#10-数据流与调用链)

---

## 1. 项目概述

NCCL 提供了一套针对 GPU 优化的标准集合通信原语，包括：

| 原语 | 说明 |
|------|------|
| `ncclAllReduce` | 所有 rank 参与规约，结果广播至所有 rank |
| `ncclAllGather` | 每个 rank 收集所有其他 rank 的数据 |
| `ncclReduceScatter` | 规约后将结果分散到各 rank |
| `ncclBroadcast` | 一个 rank 向所有其他 rank 广播数据 |
| `ncclReduce` | 所有 rank 规约到单个 root rank |
| `ncclSend` / `ncclRecv` | 点对点通信 |

NCCL 自动感知底层硬件拓扑（PCIe、NVLink、NVSwitch、InfiniBand、TCP/IP），并选择最优的算法与传输路径。

---

## 2. 顶层目录结构

```
nccl/
├── src/            # 核心 C/C++/CUDA 源码
├── bindings/       # 语言绑定（Python、IR）
├── cmake/          # CMake 配置文件
├── contrib/        # 社区贡献代码（External Proxy 等）
├── docs/           # 文档与示例
├── makefiles/      # Make 构建辅助文件
├── pkg/            # 打包脚本（Debian、RedHat、tarball）
├── plugins/        # 插件开发文档与示例
├── .github/        # GitHub Issue / PR 模板
├── CMakeLists.txt  # 顶层 CMake 入口
├── Makefile        # 顶层 Make 入口
├── README.md       # 快速入门
├── CONTRIBUTING.md # 贡献指南
└── LICENSE.txt     # Apache-2.0 许可证
```

---

## 3. 核心源码结构（`src/`）

```
src/
├── include/         # 公共头文件
├── device/          # GPU 设备端集合通信实现（CUDA kernel）
├── nccl_device/     # 设备端底层原语（新架构）
├── graph/           # 拓扑检测与图搜索
├── transport/       # 传输层实现
├── plugin/          # 插件加载与版本管理
├── ras/             # 可靠性/可用性/可服务性（RAS）
├── rma/             # 远程内存访问（RMA）
├── register/        # 缓冲区注册
├── scheduler/       # 调度器
├── devcomm/         # 设备通信器（versioned ABI）
├── gin/             # GIN（GPU Interconnect Network）主机侧
├── os/              # 操作系统抽象层（Linux / Windows）
├── param/           # 环境参数管理
│
│   # 顶层 .cc 文件（主要功能模块）
├── init.cc          # ncclCommInit* 入口，通信器初始化
├── bootstrap.cc     # 初始化阶段的 out-of-band 通信
├── channel.cc       # 通信通道管理
├── collectives.cc   # 集合通信调度入口
├── enqueue.cc       # 任务入队与 CUDA kernel 启动
├── group.cc         # ncclGroupStart/End 支持
├── proxy.cc         # 代理线程（负责网络传输的 CPU 侧）
├── transport.cc     # 传输层初始化与连接建立
├── debug.cc         # 调试输出
├── allocator.cc     # 内存分配器
├── ce_coll.cc       # Copy Engine 集合通信
├── sym_kernels.cc   # 对称 kernel 支持
├── dev_runtime.cc   # 设备运行时
└── enhcompat.cc     # 增强兼容性
```

### 3.1 头文件目录（`src/include/`）

包含所有子系统的接口声明。关键头文件：

| 头文件 | 作用 |
|--------|------|
| `comm.h` | `ncclComm` 结构体定义，通信器核心数据结构 |
| `transport.h` | 传输层接口（P2P/SHM/NET/CollNet） |
| `device.h` | 设备端公共定义（`ncclDevWorkColl` 等） |
| `proxy.h` | 代理线程接口与数据结构 |
| `graph.h` | 拓扑图查询接口 |
| `net.h` | 网络传输 API |
| `channel.h` | 通信通道结构 |
| `collectives.h` | 集合通信参数与结果结构 |
| `enqueue.h` | 任务入队接口 |
| `profiler.h` | 性能分析插件接口 |
| `tuner.h` | 调优插件接口 |
| `param.h` | `NCCL_PARAM` 宏，环境变量参数 |
| `debug.h` | 调试日志宏 |

### 3.2 设备端集合通信（`src/device/`）

包含在 GPU 上运行的集合通信 CUDA kernel：

```
src/device/
├── all_reduce.h      # AllReduce（Ring/Tree/NVLS/PAT 算法）
├── all_gather.h      # AllGather
├── all_gather_v.h    # AllGather 向量版本
├── reduce_scatter.h  # ReduceScatter
├── reduce.h          # Reduce
├── broadcast.h       # Broadcast
├── sendrecv.h        # Send/Recv 点对点
├── primitives.h      # 传输原语类（Primitives<T,RedOp,Fan,Direct,Proto>）
├── prims_simple.h    # Simple 协议原语
├── prims_ll.h        # LL（Low Latency）协议原语
├── prims_ll128.h     # LL128 协议原语
├── common.h          # 设备端公共定义
├── common.cu         # 公共 CUDA 实现
├── common_kernel.h   # 通用 CUDA 内核工具
├── reduce_kernel.h   # 规约操作（Sum/Prod/Max/Min/PreMulSum）
├── op128.h           # 128 位操作
├── onerank.cu        # 单 rank 特殊处理
└── generate.py       # 代码生成脚本（生成各算法×类型×协议组合）
```

**设计模式**：NCCL 设备端采用模板元编程，将数据类型（T）、规约操作（RedOp）、通信模式（Fan）和传输协议（Proto）作为模板参数，在编译时生成高度优化的 kernel。

### 3.3 拓扑检测与图搜索（`src/graph/`）

```
src/graph/
├── topo.cc/h    # 系统拓扑构建（GPU/PCI/NVLink/CPU/NIC 节点图）
├── search.cc    # 最优通信路径图搜索算法
├── connect.cc   # 根据图为每个 rank 建立连接
├── paths.cc     # 计算 rank 间路径与带宽
├── rings.cc/h   # Ring 拓扑生成
├── trees.cc     # Tree 拓扑生成
├── tuning.cc    # 基于拓扑的性能调优（算法/协议选择）
└── xml.cc/h     # 拓扑 XML 序列化/反序列化
```

节点类型：`GPU=0, PCI=1, NVS=2, CPU=3, NIC=4, NET=5, GIN=6, DEV=7`

### 3.4 传输层（`src/transport/`）

```
src/transport/
├── p2p.cc           # P2P 传输（NVLink / PCIe 直连）
├── shm.cc           # 共享内存传输（同节点进程间）
├── net.cc           # 网络传输（调用 NET 插件）
├── coll_net.cc      # CollNet 传输（网络侧集合卸载）
├── nvls.cc          # NVLS（NVLink SHARP）传输
├── generic.cc       # 通用传输辅助函数
├── profiler.cc      # 传输层性能分析
├── net_ib/          # InfiniBand 传输实现（内置）
│   ├── init.cc      # IB 设备初始化
│   ├── common.cc/h  # 公共工具
│   ├── connect.cc/h # 连接建立（RC QP）
│   ├── p2p.cc/h     # P2P 数据传输
│   ├── gdr.cc       # GPUDirect RDMA
│   ├── gin.cc/h     # GIN 互联支持
│   ├── reg.cc       # 内存注册
│   └── p2p_resiliency*.cc # IB 链路故障恢复
└── net_socket.cc    # TCP/IP Socket 网络传输（内置）
```

传输层接口定义（`struct ncclTransport`）包括：`canConnect`、`setup`、`connect`、`sendProxy`/`recvProxy`。

### 3.5 插件系统（`src/plugin/`）

```
src/plugin/
├── plugin_open.cc    # 动态库加载（dlopen）
├── net.cc            # NET 插件加载与多版本适配
├── net/              # 各版本 NET API 适配层（v6 ~ v12）
├── tuner.cc          # Tuner 插件加载
├── tuner/            # 各版本 Tuner API 适配层（v2 ~ v6）
├── profiler.cc       # Profiler 插件加载
├── profiler/         # 各版本 Profiler API 适配层（v1 ~ v6）
├── env.cc            # Env 插件加载
├── env/              # Env 插件 v1 适配
├── gin.cc            # GIN 插件加载
└── gin/              # 各版本 GIN API 适配层（v11 ~ v13）
```

所有插件均采用**版本化结构体**（versioned struct）模式，确保二进制兼容性。

### 3.6 代理线程（`src/proxy.cc` + `src/include/proxy.h`）

代理（Proxy）线程是 NCCL 中专门负责网络 I/O 的 CPU 线程。每个进程启动一个代理线程用于：
- 发起并跟踪网络发送/接收操作
- 驱动 InfiniBand/TCP 传输的进度
- 与设备端 kernel 通过共享内存中的 FIFO 队列交互

支持的通信模式（`ncclPattern_t`）：Ring、Tree、Pipeline、CollNet、NVLS、PAT、Send/Recv。

### 3.7 RAS 子系统（`src/ras/`）

可靠性、可用性、可服务性（Reliability, Availability, Serviceability）模块：

```
src/ras/
├── ras.cc            # RAS 线程入口，监听节点健康状态
├── rasnet.cc         # RAS 专用网络通信
├── peers.cc          # Peer 节点管理
├── collectives.cc    # RAS 专用集合通信（用于故障信息聚合）
├── client.cc         # 外部客户端接口
└── client_support.cc # 客户端辅助功能
```

### 3.8 RMA 子系统（`src/rma/`）

远程内存访问（Remote Memory Access）模块，支持 LSA（Local Symmetry Aware）模式：

```
src/rma/
├── rma.cc              # RMA 入口，wait/signal 机制
├── rma_ce.cc           # Copy Engine 方式 RMA
├── rma_proxy.cc        # 代理方式 RMA
├── rma_proxy_launch.cc # 代理 RMA 启动逻辑
└── rma_proxy_progress.cc # 代理 RMA 进度驱动
```

### 3.9 OS 抽象层（`src/os/`）

提供跨平台（Linux / Windows）抽象：

```
src/os/
├── linux.cc           # Linux 平台实现
├── linux_ipcsocket.cc # Linux IPC Socket
├── windows.cc         # Windows 平台实现
├── windows_ipcsocket.cc
└── windows_stubs.cc   # Windows 缺失功能桩
```

### 3.10 参数管理（`src/param/`）

```
src/param/
├── param.cc           # NCCL_PARAM 宏实现（环境变量 → 参数）
├── ncclparam.cc       # ncclGetParam/ncclSetParam 公共 API
├── c_api.cc           # C API 封装
└── param_registry.cc  # 参数注册表
```

---

## 4. 关键架构概念

### 4.1 算法（Algorithm）

| 算法 | 常量 | 适用场景 |
|------|------|----------|
| Tree | `NCCL_ALGO_TREE` | 延迟敏感型，小消息 |
| Ring | `NCCL_ALGO_RING` | 带宽密集型，大消息 |
| CollNetDirect | `NCCL_ALGO_COLLNET_DIRECT` | 网络侧 SHARP 直连 |
| CollNetChain | `NCCL_ALGO_COLLNET_CHAIN` | 网络侧 SHARP 链式 |
| NVLS | `NCCL_ALGO_NVLS` | NVLink SHARP（单节点多 GPU） |
| NVLSTree | `NCCL_ALGO_NVLS_TREE` | NVLink SHARP + Tree（跨节点） |
| PAT | `NCCL_ALGO_PAT` | Pipeline Adaptive Tree |

### 4.2 传输协议（Protocol）

| 协议 | 说明 | 适用场景 |
|------|------|----------|
| `LL`（Low Latency）| 通过标志位实现低延迟同步 | 小消息 |
| `LL128` | LL 的 128 字节对齐优化版本 | 中等消息 |
| `Simple` | 简单 DMA 传输，无标志位同步 | 大消息 |

### 4.3 传输类型（Transport）

| 类型 | 常量 | 说明 |
|------|------|------|
| P2P | `TRANSPORT_P2P=0` | NVLink / PCIe 直连（同节点 GPU 间） |
| SHM | `TRANSPORT_SHM=1` | 共享内存（同节点多进程） |
| NET | `TRANSPORT_NET=2` | 网络（IB / TCP，跨节点） |
| CollNet | `TRANSPORT_COLLNET=3` | 网络侧集合卸载（SHARP） |

### 4.4 通信器（`ncclComm`）

`ncclComm` 是 NCCL 的核心数据结构，保存了：
- 所有 rank 的 peer 信息（`ncclPeerInfo`）
- 通信通道列表（`ncclChannel[]`）
- 传输层连接（`ncclConnector`）
- 代理线程状态（`ncclProxyState`）
- 拓扑图（`ncclTopoSystem`）
- 设备端共享状态（`ncclDevComm`）
- 插件句柄（tuner/profiler/net）

---

## 5. 构建系统

NCCL 同时支持 **Make** 和 **CMake** 两套构建系统。

### Make 构建

```bash
# 编译库
make -j src.build

# 指定 CUDA 路径
make -j src.build CUDA_HOME=/usr/local/cuda

# 仅针对特定 GPU 架构（加速编译）
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# 打 Debian 包
make pkg.debian.build

# 打 RPM 包
make pkg.redhat.build

# 打 tarball
make pkg.txz.build
```

### `makefiles/` 辅助文件

| 文件 | 作用 |
|------|------|
| `version.mk` | 版本号（MAJOR/MINOR/PATCH） |
| `common.mk` | 编译器标志、NVCC 参数、GPU 架构列表 |
| `formatting.mk` | 代码格式化规则 |
| `examples.mk` | 示例编译规则 |

---

## 6. 插件生态系统（`plugins/`）

`plugins/` 目录提供各类插件的**开发文档**：

| 目录 | 插件类型 | 说明 |
|------|----------|------|
| `plugins/net/` | 网络插件（`libnccl-net.so`） | 自定义网络传输（替代内置 IB/Socket） |
| `plugins/tuner/` | 调优插件（`libnccl-tuner.so`） | 自定义算法/协议选择策略 |
| `plugins/profiler/` | 性能分析插件（`libnccl-profiler.so`） | 对接外部性能分析工具 |
| `plugins/env/` | 环境插件 | 自定义参数读取来源 |
| `plugins/mixed/` | 混合插件 | 多功能组合插件示例 |

所有插件均通过 `dlopen` 动态加载，采用**版本化结构体**保证向前/向后兼容。环境变量 `NCCL_NET_PLUGIN`、`NCCL_TUNER_PLUGIN` 等控制插件选择。

---

## 7. 语言绑定（`bindings/`）

### Python 绑定（`bindings/nccl4py/`）

提供 Python 封装，使 Python 程序可直接调用 NCCL API：
- `setup.py` / `pyproject.toml`：标准 Python 打包配置
- 使用 `make nccl4py.build` 构建

### IR 绑定（`bindings/ir/`）

生成 NCCL 的中间表示（Intermediate Representation）头文件：
- `nccl_device_wrapper.h`：设备端接口包装
- 依赖先构建主库（`ir.build` → `src.build`）

---

## 8. 贡献代码（`contrib/`）

### `contrib/nccl_ep/`（External Proxy）

External Proxy（EP）是一种将 NCCL 代理线程外置到独立进程的架构，用于：
- 在 CPU 资源受限的环境下提升扩展性
- 支持 IB 网络下的高效多 rail 通信

包含：
- `nccl_ep.cc`：EP 实现
- `ep_test.cu` / `ep_bench.cu`：测试与基准
- `ep_test.py`：Python 测试脚本

---

## 9. 打包系统（`pkg/`）

```
pkg/
├── debian/     # Debian/Ubuntu DEB 包配置
├── redhat/     # RedHat/CentOS RPM 包配置（nccl.spec.in）
├── txz/        # OS 无关 tarball
└── srctxz/     # 源码 tarball
```

---

## 10. 数据流与调用链

```
用户代码
   │
   ▼
ncclAllReduce() / ncclSend() 等公共 API
   │  (src/collectives.cc, src/enqueue.cc)
   ▼
ncclGroupEnd() → 任务入队 → 选择算法/协议
   │  (src/graph/tuning.cc → Tuner 插件)
   ▼
enqueueCollective() → 生成 ncclKernelPlan
   │  (src/enqueue.cc)
   ▼
启动 CUDA Kernel（设备端）          代理线程（CPU 侧）
   │  (src/device/*.h)                │  (src/proxy.cc)
   │                                  │
   ▼                                  ▼
传输原语 Primitives<T,RedOp,Fan,Proto>  ncclProxyArgs 进度驱动
   │  (src/device/primitives.h)        │
   │                                  ▼
   │                         NET 插件 / IB / Socket
   │                         (src/transport/net_ib/, net_socket.cc)
   │
   ▼
传输层连接（P2P / SHM / NET / CollNet）
   (src/transport/*.cc)
```

### 通信器初始化流程

```
ncclCommInitRank()
   │
   ├── ncclBootstrapInit()       # 建立 out-of-band 通信环
   ├── ncclTopoGetSystem()       # 检测硬件拓扑（GPU/NVLink/PCIe/NIC）
   ├── ncclTopoComputePaths()    # 计算 rank 间路径与带宽
   ├── ncclTopoSearchAndConnect() # 图搜索，选择最优 Ring/Tree 路径
   ├── ncclTransportP2pSetup()   # 建立传输层连接
   └── ncclProxyCreate()         # 启动代理线程
```

---

## 附：关键环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_DEBUG` | 调试级别（`VERSION/WARN/INFO/TRACE`） |
| `NCCL_NET_PLUGIN` | 指定 NET 插件名（加载 `libnccl-net-<name>.so`） |
| `NCCL_TUNER_PLUGIN` | 指定 Tuner 插件名 |
| `NCCL_PROFILER_PLUGIN` | 指定 Profiler 插件名 |
| `NCCL_ALGO` | 强制指定算法（`Ring/Tree/NVLS` 等） |
| `NCCL_PROTO` | 强制指定协议（`LL/LL128/Simple`） |
| `NCCL_NTHREADS` | 指定每 kernel 的线程数 |
| `NCCL_NCHANNELS_PER_NET_PEER` | 每对 NET peer 的通道数 |
| `NCCL_IB_HCA` | 指定 IB 网卡 |
| `NCCL_SOCKET_IFNAME` | 指定 TCP Socket 网络接口 |
| `NCCL_TOPO_FILE` | 指定自定义拓扑 XML 文件 |
| `NCCL_P2P_DISABLE` | 禁用 P2P 传输 |
| `NCCL_SHM_DISABLE` | 禁用共享内存传输 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 级别 |
