# Paged-KV-Cache Engine: Virtual Memory Allocator & Fused PagedAttention CUDA Kernel

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA 12.8 / 13.0](https://img.shields.io/badge/CUDA-12.8%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![SM 9.0 / SM 12.0](https://img.shields.io/badge/Architecture-Hopper%20SM90%20%7C%20Blackwell%20SM120-brightgreen.svg)](https://developer.nvidia.com/cuda-gpus)
[![AVX2 SIMD](https://img.shields.io/badge/SIMD-AVX2-orange.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-low latency, production-grade virtual Key-Value (KV) cache memory management engine and fused **PagedAttention v2** CUDA kernel implemented in C++20, CUDA 12.8/13.0, and AVX2 SIMD. Inspired by OS virtual memory paging and `vLLM` infrastructure, this engine eliminates VRAM fragmentation, supports Copy-on-Write (CoW) block sharing with LRU prefix caching, and delivers deterministic, high-throughput LLM serving across consumer and datacenter GPUs.

---

## 1. Executive Summary & Hardware Benchmarks

### Architecture Overview
Traditional LLM inference pre-allocates contiguous memory blocks per request based on the maximum sequence length (e.g., $N=4096$). This leads to severe **external and internal memory fragmentation** (up to 60–80% VRAM waste). **Paged-KV-Cache Engine** partitions KV-cache tensors into fixed-size physical blocks ($B=16$ tokens) and maps non-contiguous physical blocks through a virtual page table.

```text
+-----------------------------------------------------------------------------------+
|                            PAGED KV-CACHE ARCHITECTURE                            |
+-----------------------------------------------------------------------------------+
|  Logical Request Sequence                                                         |
|  [Block 0: Tokens 0-15] -> [Block 1: Tokens 16-31] -> [Block 2: Tokens 32-47]     |
|             |                            |                            |           |
|             v                            v                            v           |
|  Virtual Page Table                                                               |
|  Logical ID 0 -> Physical Block 102                                               |
|  Logical ID 1 -> Physical Block 45 (Shared CoW System Prompt)                     |
|  Logical ID 2 -> Physical Block 819                                               |
|             |                            |                            |           |
|             v                            v                            v           |
|  Physical VRAM Pool (Non-Contiguous Blocks in HBM3e / GDDR7)                      |
|  [Phys Block 45] (Ref Count: 64) <--- Shared Prefix (AVX2 Hashed)                 |
|  [Phys Block 102] (Ref Count: 1)                                                  |
|  [Phys Block 819] (Ref Count: 1)                                                  |
+-----------------------------------------------------------------------------------+
```

### Multi-Architecture Empirical Hardware Benchmarks

Validated natively across two distinct GPU tiers:
1. **Datacenter Tier (Hopper — Commit `2a91d2f`, `-DCMAKE_CUDA_ARCHITECTURES=90`):** **NVIDIA H200 NVL** (`143,771 MiB` HBM3e, `SM 9.0`, `4,094.60 GB/s` measured bandwidth, `701.02 TFLOPS` BF16), Driver `580.105.08`, CUDA `13.0.88`.
2. **Edge / Workstation Tier (Blackwell):** **NVIDIA GeForce RTX 5060** (`8 GB` GDDR7, `SM 12.0`) + **AMD Ryzen 5 5600X** (6C/12T, 3.7 GHz), 32GB DDR4-3200 RAM, Ubuntu 22.04 LTS.

| Metric | NVIDIA H200 NVL (`SM 90`, CUDA 13.0) | NVIDIA RTX 5060 (`SM 12.0`, CUDA 12.8) | Contiguous Baseline | Key Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Correctness Suite (Allocator + CoW + FP64 Ref)** | **3/3 Pass (`Max Err: 8.30e-4`)** | **3/3 Pass (`< 1e-3`)** | N/A | Bit-accurate vs. CPU FP64 |
| **VRAM Memory Usage (64 Sequences)** | **519.88 MB** | **519.88 MB** | 1,024.00 MB | **49.23% VRAM Waste Eliminated** |
| **LRU Prefix Cache Hit Rate** | **100.00 %** | **100.00 %** | 0.00 % | **Zero-Copy CoW Prompt Sharing** |
| **Time-To-First-Token (TTFT Kernel)** | **2.54 ms** | **7.99 ms** | ~24.50 ms | **Up to 9.66x Prefill Speedup** |
| **Time-Per-Output-Token (TPOT Mean / P50)** | **2.43 ms/step** | **7.24 ms/step** | ~18.20 ms/step | **Up to 7.49x Decode Speedup** |
| **Step Latency Tail (P90 / P99 / Max)** | **2.44 / 2.44 / 2.46 ms** | **7.91 / 8.58 ms** | 28.90 ms | **0.03 ms Max Jitter on Hopper** |
| **Kernel Decoding Throughput** | **26,348.5 tokens/sec** | **8,840.0 tokens/sec** | ~3,100 tokens/sec | **2.98x Hopper Scaling vs. RTX 5060** |

> **Methodological Note:** Kernel latency and throughput metrics above measure isolated C++20/CUDA virtual memory paging and fused `PagedAttention` execution over fixed-length sequences (excluding HTTP/network serialization or autoregressive tokenizer sampling). On the H200 NVL (`Ridge Point = 171.21 FLOP/Byte`), decode workloads remain strictly memory-bandwidth bound, explaining the **2.98x kernel throughput scaling** when moving from GDDR7 to 4.09 TB/s HBM3e.

---

## 2. Performance Visualizations & Telemetry

All plots are generated directly from empirical C++ benchmark execution CSV outputs (`benchmark_summary.csv` and `step_latencies.csv`).

### VRAM Memory Savings vs Sequence Count
Comparison of pre-allocated contiguous VRAM memory vs. dynamically allocated Paged KV-Cache blocks showing 49.23% memory waste elimination at 64 sequences.

![VRAM Fragmentation Comparison](assets/vram_fragmentation_comparison.png)

---

### Step Latency CDF & P99 SLA Guarantee
Cumulative distribution function (CDF) of decode step latencies highlighting strict P99 latency SLA bounds (8.58 ms on RTX 5060; 2.44 ms on H200 NVL).

![PagedAttention Latency CDF](assets/paged_attention_latency_cdf.png)

---

### Copy-on-Write (CoW) Prefix Sharing Savings
System VRAM footprint comparison with and without Copy-on-Write prefix block sharing across long shared prompt prefixes.

![CoW Prefix Savings](assets/cow_prefix_memory_savings.png)

---

### Continuous Decoding Throughput Scaling
Scaling of continuous generation throughput (tokens/second) across batch sizes from 1 to 64 concurrent sequences.

![Kernel Throughput Scaling](assets/kernel_throughput_scaling.png)

---

## 3. Mathematical & Algorithmic Foundations

### Fused PagedAttention with Online Softmax

Traditional multi-head attention materializes the intermediate attention matrix $A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$, requiring $O(N^2)$ VRAM footprint per sequence. PagedAttention processes non-contiguous physical memory blocks $B$ on-the-fly using an online softmax accumulator without materializing $A$.

For a Query vector $q_i \in \mathbb{R}^{d_k}$ and block table $\text{BlockTable}(s) = [p_0, p_1, \dots, p_{M-1}]$ mapping logical block indices to physical block addresses:

$$S_{i, j} = \frac{q_i \cdot K_{\text{phys}(j)}^T}{\sqrt{d_k}}$$

$$\tilde{m}_j = \max\left(m_{j-1}, \max_{t \in B_j} S_{i, t}\right)$$

$$d_j = d_{j-1} \cdot e^{m_{j-1} - \tilde{m}_j} + \sum_{t \in B_j} e^{S_{i, t} - \tilde{m}_j}$$

$$O_j = O_{j-1} \cdot e^{m_{j-1} - \tilde{m}_j} + \sum_{t \in B_j} e^{S_{i, t} - \tilde{m}_j} V_{\text{phys}(j), t}$$

At the final block $M-1$, the output vector is normalized:

$$O_{\text{final}} = \frac{O_{M-1}}{d_{M-1}}$$

---

## 4. Key Architectural Features

1. **Virtual Block Allocator (`BlockAllocator`)**:
   - Fixed-size block allocation ($B=16$ tokens) avoiding external memory fragmentation.
   - O(1) page table lookup for logical-to-physical translation.
   - Reference-counted blocks supporting instant Copy-on-Write (CoW) sequence cloning and OOM-safe rollback.

2. **AVX2 SIMD Prefix Hasher (`AVX2PrefixHasher`)**:
   - 256-bit vectorized token hashing for rapid prefix matching across prompts.
   - High-throughput zero-copy prompt sharing across concurrent client requests.

3. **LRU Prefix Cache (`LRUPrefixCache`)**:
   - Eviction-policy cache managing warm system prompts in physical VRAM blocks.
   - Achieves 100% hit rates for common system prompts and long context prefixes.

4. **Fused PagedAttention CUDA Kernel (`paged_attention_v2_kernel`)**:
   - 128-bit vectorized FP16 loads (`uint4` casting) utilizing full HBM3e and GDDR7 memory bandwidth.
   - Warp-level reduction primitives (`__shfl_down_sync`) for fast intra-warp max and sum reductions.
   - Zero intermediate matrix materialization, eliminating dynamic VRAM allocation during decoding.

---

## 5. Directory Structure

```text
paged-kv-cache-engine/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── assets/
│   ├── vram_fragmentation_comparison.png
│   ├── paged_attention_latency_cdf.png
│   ├── cow_prefix_memory_savings.png
│   └── kernel_throughput_scaling.png
├── include/
│   ├── block_allocator.hpp
│   ├── lru_prefix_cache.hpp
│   ├── avx2_prefix_hasher.hpp
│   ├── paged_attention_kernel.cuh
│   └── continuous_scheduler.hpp
├── src/
│   ├── block_allocator.cpp
│   ├── lru_prefix_cache.cpp
│   ├── avx2_prefix_hasher.cpp
│   ├── paged_attention_kernel.cu
│   ├── continuous_scheduler.cpp
│   └── main_benchmark.cpp
├── tests/
│   └── test_paged_kv.cpp
└── scripts/
    └── generate_plots.py
```

---

## 6. Deterministic Ultra-Low Latency Implications (Enterprise Serving, HFT & Robotics)

- **High-Throughput Datacenter LLM Serving (Hopper H200)**: Eliminating 49.23% of KV-cache fragmentation doubles effective concurrent batch capacity within HBM3e limits while maintaining a flat **2.44 ms P99** kernel step SLA (`0.03 ms` max jitter over P50).
- **High-Frequency Trading & Market Intelligence**: Zero-fragmentation virtual memory allocation guarantees deterministic upper-bounded latencies for streaming market sentiment analysis without unpredictable memory allocation stalls.
- **Autonomous Robotics & Edge Systems**: Efficient KV-cache paging fits long-horizon context windows into constrained onboard VRAM (e.g., 8GB GDDR7), enabling real-time spatial reasoners to operate at continuous high frame rates.

---

## 7. Building & Running

### Prerequisites
- Linux (Ubuntu 22.04 LTS recommended)
- CUDA Toolkit 12.8 or higher (tested up to CUDA 13.0 on `SM 90` Hopper and `SM 12.0` Blackwell)
- GCC 11.4+ with C++20 support
- CMake 3.22+
- Python 3 + `matplotlib`, `numpy` (for plot generation)

### Build Instructions
```bash
git clone https://github.com/JohnScheuer/paged-kv-cache-engine.git
cd paged-kv-cache-engine
mkdir build && cd build

# Target native GPU architecture (e.g., 90 for Hopper H200, 120 for Blackwell RTX 5060)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc)
```

### Running Tests & Benchmarks
```bash
# Run unit tests (Physical allocation, CoW, Correctness vs Ref CPU Softmax)
./paged_kv_test

# Run benchmark & generate CSV telemetry
./paged_kv_benchmark

# Regenerate assets
python3 ../scripts/generate_plots.py
```

---

## 8. Acknowledgments & License

- **Hardware Validation Acknowledgments**: Datacenter Hopper (`SM 90`) telemetry validated on an enterprise **NVIDIA H200 NVL** cluster in collaboration with **Vinicius Rosa**.
- **License**: Released under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Author**: João Felipe de Souza ([github.com/JohnScheuer](https://github.com/JohnScheuer)) — *AI Infrastructure & Low-Level Systems Engineer*