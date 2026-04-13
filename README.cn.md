<img src="asset/sii-infrawaves-buaa.svg" alt="VCCL Logo" style="width: 100%; max-width: 100%;">
<br>
<br>
<h1 align="center">VCCL: Venus Collective Communication Library</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2510.00991">📄 论文</a> | 
  <a href="https://vccl-doc.readthedocs.io/">📚 技术文档</a> | 
  <a href="https://discord.gg/VBwk5PuY"><img src="asset/discord.svg" width="16" height="16" style="vertical-align: middle;"> Discord</a> | 
  <a href="asset/wechat_fig.png"><img src="asset/wechat.svg" width="16" height="16" style="vertical-align: middle;"> 微信交流群</a> | 
  <a href="README.md">🇺🇸 English</a>
</p>
VCCL 是一款面向 GPU 的集合通信库，提供 all-reduce、all-gather、reduce、broadcast、reduce-scatter 以及通用 send/recv 通信原语。兼容 PCIe、NVLink、NVSwitch，并支持通过 InfiniBand Verbs 或 TCP/IP 套接字进行跨节点通信；适配单机/多机、多进程（如 MPI）或单进程应用。

## 🅾 简介

VCCL 以**高效率、高可用、高可视**三大核心能力，重塑 GPU 通信体验：

- **高效率（High Efficiency）**  
  借鉴 DPDK 的设计理念，VCCL 引入 “DPDK-Like P2P” 高性能调度，让 GPU 尽可能保持满载：  
  早期在 CPU 侧实现万兆网络的用户态高速数据面，需要通过**大页内存 + 零拷贝**并将数据通路从**内核态迁移到用户态**，以绕开协议栈多次拷贝和中断开销。  
  类比之下，当前 CUDA 在通信/计算调度与 API 颗粒度方面仍有限制（公开资料亦提到 H800 的 132 个 SM 中会预留约 20 个用于通信）。VCCL 的做法与 DPDK 优化内核路径类似：**通过智能调度将通信任务从 GPU CUDA 栈卸载到 CPU 侧**，联合**零拷贝**与 **PP 工作流的全局负载均衡**，显著缩短 GPU 空闲时间。  
  在**千亿参数 Dense 模型训练**中，我们的内部对比结果显示：**整网训练算力利用率在 SOTA 之上再提升约 2%**（环境可复现，见[示例说明](https://vccl-doc.readthedocs.io/en/latest/features/sm-free-overlap/)）。
  Ps:无核现在还不支持容错和telemetry，后续工作之一。

- **高可用（High Availability）**  
  提供**轻量级本地恢复**容错([Fault Tolerance](https://vccl-doc.readthedocs.io/en/latest/features/fault-tolerance/))机制，在不显著增加系统负担的前提下，有效应对**网口 Down / 交换机故障**等引发的训练中断。具体而言，VCCL通过创建备份QP的方式来实现链路故障后的流量切换，同时，VCCL支持当链路状态恢复后无感的流量切回。整体故障率可**降低超过 50%**。

- **高可视（High Visibility）**  
  提供**微秒级滑动窗口**的通信流量可观测([Flow Telemetry](https://vccl-doc.readthedocs.io/en/latest/features/flow-telemetry/))，助力研发调优与**拥塞检测**。

更多 VCCL 介绍和使用指南, 请参考[VCCL 文档](https://vccl-doc.readthedocs.io/)。

## 获取与构建

> **说明**：当前提供源码构建方式。

```shell
$ git clone https://github.com/sii-research/VCCL.git
$ cd VCCL
$ make -j src.build
```

如 CUDA 不在 /usr/local/cuda,

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

VCCL 编译产物位于 `build/`（可通过 BUILDDIR 修改）。

默认情况下，VCCL 会针对所有支持的架构进行编译。为了加速编译并减小二进制文件大小，可以考虑重新定义 `NVCC_GENCODE`（在 `makefiles/common.mk` 中定义），使其仅包含目标平台的架构：
```shell
# 仅编译目标架构以加速构建并减小体积（示例：Hopper 架构 H100/H200）
$ make -j80 src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## 📦 打包安装

要在系统上安装 VCCL，请创建一个包，然后以 root 身份安装它。

```shell
# Debian/Ubuntu
sudo apt install -y build-essential devscripts debhelper fakeroot
make pkg.debian.build
ls build/pkg/deb/

# RedHat/CentOS
sudo yum install -y rpm-build rpmdevtools
make pkg.redhat.build
ls build/pkg/rpm/

# OS-agnostic tarball
make pkg.txz.build
ls build/pkg/txz/
```

## 🍽️ 测试

VCCL的测试，请参考 https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## 🙇‍♂️ 许可证与致谢

* 本项目基于 `nccl_2.26.6-1` 开发，并在相关文件中保留上游版权与许可信息。

* 详细许可条款请见 LICENSE。

* 感谢开源社区（包括但不限于 NCCL 与 nccl-tests 项目）的卓越工作。
