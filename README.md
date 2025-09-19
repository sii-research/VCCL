<img src="asset/LOGO.svg" alt="VCCL Logo" style="width: 100%; max-width: 100%;">
<br>
<br>
<h1 align="center">VCCL: Venus Collective Communication Library</h1>

<p align="center">
  <a href="https://example.com/paper">📄 Paper</a> | 
  <a href="https://vccl-doc.readthedocs.io/">📚 Documentation</a> | 
  <a href="https://discord.gg/VBwk5PuY"><img src="asset/discord.svg" width="16" height="16" style="vertical-align: middle;"> Discord</a> | 
  <a href="asset/QRcode.jpeg"><img src="asset/wechat.svg" width="16" height="16" style="vertical-align: middle;"> WeChat Group</a> | 
  <a href="README.cn.md">🇨🇳 中文</a>
</p>

VCCL is a collective communication library for GPUs. It provides communication primitives such as **all-reduce, all-gather, reduce, broadcast, reduce-scatter**, and general **send/recv**. It is compatible with **PCIe, NVLink, and NVSwitch**, and supports **cross-node communication via InfiniBand Verbs or TCP/IP sockets**. It can be used in single-node/multi-node, multi-process (e.g., MPI), or single-process applications.

---

## 🅾 Introduction

VCCL redefines the GPU communication experience with three core capabilities: **High Efficiency, High Availability, and High Visibility**.

- **High Efficiency**  
  Inspired by the **DPDK** design philosophy, VCCL introduces a **“DPDK-Like P2P” high-performance scheduling mechanism**, ensuring that GPUs remain fully utilized.  
  In the early days of high-speed networking on CPUs, achieving 10Gbps network performance was nearly impossible due to kernel stack overhead (multiple memory copies, interrupt handling inefficiencies). DPDK solved this by leveraging **hugepage memory + zero-copy** and moving the data path from kernel space to user space.  
  Similarly, current CUDA still faces limitations in communication/computation scheduling and API granularity (public sources note that ~20 out of 132 SMs on the H800 GPU are reserved for communication). VCCL adopts an analogous optimization strategy: **offloading communication tasks from GPU CUDA stack to the CPU side**, combined with **zero-copy** and **global load balancing across pipeline parallel workflows (PP)**.  
  In **training dense models with hundreds of billions of parameters**, our internal benchmarks show that **cluster-wide training compute efficiency improves by ~2% compared to state-of-the-art baselines** ([More about use zerocopy for training](https://vccl-doc.readthedocs.io/en/latest/features/sm-free-overlap/)).
  
  Note: The SM-Free mode currently does not support fault tolerance or telemetry; this is planned as future work.

- **High Availability**  
  Provides a **lightweight local recovery fault-tolerance mechanism** that effectively handles **NIC failures and switch faults** without significantly increasing system overhead. Concretly, when link fail occurs, VCCL can migrades the traffic within one iteration by **creating a backup QP**. Simultaneously, VCCL supports **seamless traffic recovery to the primary QP** once link integrity is re-established. In practice, this reduces overall training interruption rates by **over 50%** ([More about fault tolerance](https://vccl-doc.readthedocs.io/en/latest/features/fault-tolerance/)).

- **High Visibility**  
  Offers **microsecond-level sliding-window flow telemetry**, enabling efficient bottleneck localization and **congestion detection** for performance tuning([More about flow telemetry](https://vccl-doc.readthedocs.io/en/latest/features/flow-telemetry/)).

For more information about VCCL and how to use it, please refer to the [VCCL documentation](https://vccl-doc.readthedocs.io/).

---
## Build from Source

> **Note**: Currently, only source builds are supported.

```bash
$ git clone https://github.com/sii-research/VCCL.git
$ cd VCCL
$ make -j src.build
```

If CUDA is not installed under /usr/local/cuda:

```bash
$ make src.build CUDA_HOME=<path to cuda install>
```

Build artifacts are placed in the build/ directory (can be customized via BUILDDIR).

By default, VCCL compiles for all supported architectures. To speed up builds and reduce binary size, redefine NVCC_GENCODE (in makefiles/common.mk) to include only your target architecture(s):

```sh
# Example: build only for Hopper architecture (H100/H200, sm_90)
$ make -j80 src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## 📦 Packaging & Installation
To install VCCL on your system, build a package and install it as root:

```sh
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

## 🍽️ Testing
Tests for VCCL are maintained separately at NVIDIA NCCL Tests.

```sh
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## 🙇‍♂️ License & Acknowledgements

* This project is developed based on nccl_2.26.6-1, and retains upstream copyright and license information in the relevant files.

* See the LICENSE file for detailed terms.

* Thanks to the open-source community (including but not limited to NCCL and nccl-tests) for their outstanding work.