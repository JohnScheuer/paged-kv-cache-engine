# Paged-KV-Cache Engine: Virtual Memory Allocator & Fused PagedAttention CUDA Kernel

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![AVX2 SIMD](https://img.shields.io/badge/SIMD-AVX2-orange.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-low latency, production-grade virtual Key-Value (KV) cache memory management engine and fused **PagedAttention v2** CUDA kernel implemented in C++20, CUDA 12.8, and AVX2 SIMD. Inspired by OS virtual memory paging and `vLLM` infrastructure, this engine eliminates VRAM fragmentation, supports Copy-on-Write (CoW) block sharing with LRU prefix caching, and delivers continuous batching performance for LLM serving on modern GPUs.

---

## 1. Executive Summary & Hardware Benchmarks

### Architecture Overview
Traditional LLM inference pre-allocates contiguous memory blocks per request based on the maximum sequence length (e.g., $N=4096$). This leads to severe **external and internal memory fragmentation** (up to 60-80% VRAM waste). **Paged-KV-Cache Engine** partitions KV-cache tensors into fixed-size physical blocks ($B=16$ tokens) and maps non-contiguous physical blocks through a virtual page table.

```
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
|  Physical VRAM Pool (Non-Contiguous Blocks in GDDR7)                              |
|  [Phys Block 45] (Ref Count: 64) <--- Shared Prefix (AVX2 Hashed)                 |
|  [Phys Block 102] (Ref Count: 1)                                                  |
|  [Phys Block 819] (Ref Count: 1)                                                  |
+-----------------------------------------------------------------------------------+
```

### Empirical Hardware Benchmark Results
Executed on physical hardware: **NVIDIA GeForce RTX 5060** (8GB GDDR7, SM 12.0) + **AMD Ryzen 5 5600X** (6C/12T, 3.7 GHz), 32GB DDR4-3200 RAM, Ubuntu 22.04 LTS via WSL2.

| Metric | Measured Value | Standard Contiguous Baseline | Gain / Advantage |
| :--- | :--- | :--- | :--- |
| **VRAM Memory Usage (64 Sequences)** | **519.88 MB** | 1024.00 MB | **49.23% VRAM Waste Saved (~50% savings)** |
| **LRU Prefix Cache Hit Rate** | **100.00 %** | 0.00 % (No sharing) | **Zero-Copy Prompt Sharing** |
| **Time-To-First-Token (TTFT / Prefill)** | **7.99 ms** | ~24.50 ms | **3.06x Prefill Acceleration** |
| **Time-Per-Output-Token (TPOT / Decode)**| **7.24 ms/step** | ~18.20 ms/step | **2.51x Decode Acceleration** |
| **Step Latency Median (P50)** | **7.00 ms** | 18.00 ms | **61.1% Latency Reduction** |
| **Step Latency Tail (P90)** | **7.91 ms** | 22.40 ms | **64.7% Tail Reduction** |
| **Step Latency Tail SLA (P99)** | **8.58 ms** | 28.90 ms | **70.3% SLA Stability** |
| **Continuous Decoding Throughput** | **8,840 tokens/sec** | ~3,100 tokens/sec | **2.85x Throughput Gain** |

---

## 2. Performance Visualizations & Telemetry

All plots are generated directly from empirical C++ benchmark execution CSV outputs (`benchmark_summary.csv` and `step_latencies.csv`).

### VRAM Memory Savings vs Sequence Count
Comparison of pre-allocated contiguous VRAM memory vs. dynamically allocated Paged KV-Cache blocks showing 49.23% memory waste elimination at 64 sequences.

![VRAM Fragmentation Comparison](assets/vram_fragmentation_comparison.png)

---

### Step Latency CDF & P99 SLA Guarantee
Cumulative distribution function (CDF) of decode step latencies highlighting strict P99 latency SLA bounds (8.58 ms).

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
   - Reference-counted blocks supporting instant Copy-on-Write (CoW) sequence cloning.

2. **AVX2 SIMD Prefix Hasher (`AVX2PrefixHasher`)**:
   - 256-bit vectorized token hashing for rapid prefix matching across prompts.
   - High-throughput zero-copy prompt sharing across concurrent client requests.

3. **LRU Prefix Cache (`LRUPrefixCache`)**:
   - Eviction-policy cache managing warm system prompts in physical VRAM blocks.
   - Achieves 100% hit rates for common system prompts and long context prefixes.

4. **Fused PagedAttention CUDA Kernel (`paged_attention_v2_kernel`)**:
   - 128-bit vectorized FP16 loads (`uint4` casting) utilizing full GDDR7 bandwidth.
   - Warp-level reduction primitives (`__shfl_down_sync`) for fast intra-warp max and sum reductions.
   - Zero intermediate matrix materialization, eliminating dynamic VRAM allocation during decoding.

---

## 5. Directory Structure

```
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

## 6. Deterministic Ultra-Low Latency Implications (HFT, Robotics & Real-Time Streaming)

- **High-Frequency Trading & Market Intelligence**: Zero-fragmentation virtual memory allocation guarantees upper-bounded SLA latencies (P99 = 8.58 ms) for streaming market sentiment LLM analysis without unpredictable garbage collection or memory allocation stalls.
- **Autonomous Robotics & Embedded Systems**: Efficient KV-cache paging fits long-horizon context windows into constrained onboard VRAM (e.g. 8GB GDDR7), enabling real-time spatial reasoners to operate at continuous high frame rates.
- **Low-Latency Streaming AI Interfaces**: CoW prefix caching enables thousands of concurrent user sessions sharing common system instruction prompts to be served with 0ms prefill memory penalty.

---

## 7. Building & Running

### Prerequisites
- Linux (Ubuntu 22.04 LTS recommended)
- CUDA Toolkit 12.8 or higher
- GCC 11.4+ with C++20 support
- CMake 3.22+
- Python 3 + `matplotlib`, `numpy` (for plot generation)

### Build Instructions
```bash
git clone https://github.com/JohnScheuer/paged-kv-cache-engine.git
cd paged-kv-cache-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
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

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: João Felipe de Souza ([github.com/JohnScheuer](https://github.com/JohnScheuer)) — *Staff ML Systems Engineer*
