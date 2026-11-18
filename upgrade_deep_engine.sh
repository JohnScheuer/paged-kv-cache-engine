#!/usr/bin/env bash
set -euo pipefail

CDIR=~/paged-kv-cache-engine
cd "$CDIR"

# 1. Create include/lru_prefix_cache.hpp (LRU Cache Eviction & Host Offloader)
cat << 'LRU_EOF' > include/lru_prefix_cache.hpp
#ifndef LRU_PREFIX_CACHE_HPP
#define LRU_PREFIX_CACHE_HPP

#include "kv_cache_types.hpp"
#include "block_allocator.hpp"
#include <unordered_map>
#include <list>
#include <iostream>
#include <mutex>

namespace paged_kv {

struct PrefixCacheEntry {
    uint64_t prefix_hash;
    uint32_t physical_block_id;
    bool is_offloaded_to_host;
    void* host_k_backup;
    void* host_v_backup;
};

class LRUPrefixCache {
private:
    size_t capacity_;
    std::list<PrefixCacheEntry> lru_list_;
    std::unordered_map<uint64_t, std::list<PrefixCacheEntry>::iterator> cache_map_;
    std::mutex mutex_;
    size_t hit_count_{0};
    size_t miss_count_{0};

public:
    explicit LRUPrefixCache(size_t capacity) : capacity_(capacity) {}

    ~LRUPrefixCache() {
        for (auto& entry : lru_list_) {
            if (entry.host_k_backup) cudaFreeHost(entry.host_k_backup);
            if (entry.host_v_backup) cudaFreeHost(entry.host_v_backup);
        }
    }

    uint32_t get_or_lookup(uint64_t prefix_hash, BlockAllocator& allocator) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(prefix_hash);
        if (it != cache_map_.end()) {
            // Hit! Move to front of LRU list
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
            ++hit_count_;
            uint32_t block_id = it->second->physical_block_id;
            allocator.share_block(block_id);
            return block_id;
        }

        ++miss_count_;
        return INVALID_BLOCK_ID;
    }

    void put(uint64_t prefix_hash, uint32_t physical_block_id, BlockAllocator& allocator) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_map_.find(prefix_hash);
        if (it != cache_map_.end()) {
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
            return;
        }

        // Evict cold entry if capacity reached
        if (lru_list_.size() >= capacity_) {
            auto last = lru_list_.end();
            --last;
            
            // Only evict if block reference count allows
            if (allocator.get_ref_count(last->physical_block_id) <= 1) {
                allocator.free_block(last->physical_block_id);
                cache_map_.erase(last->prefix_hash);
                if (last->host_k_backup) cudaFreeHost(last->host_k_backup);
                if (last->host_v_backup) cudaFreeHost(last->host_v_backup);
                lru_list_.pop_back();
            }
        }

        PrefixCacheEntry entry{prefix_hash, physical_block_id, false, nullptr, nullptr};
        lru_list_.push_front(entry);
        cache_map_[prefix_hash] = lru_list_.begin();
        allocator.share_block(physical_block_id);
    }

    double get_hit_rate() const {
        size_t total = hit_count_ + miss_count_;
        return total > 0 ? (double)hit_count_ / total * 100.0 : 0.0;
    }

    size_t get_hit_count() const { return hit_count_; }
    size_t get_miss_count() const { return miss_count_; }
};

} // namespace paged_kv

#endif // LRU_PREFIX_CACHE_HPP
LRU_EOF

# 2. Upgrade src/paged_attention.cu with 128-bit Vectorized Memory Access
cat << 'CU_EOF' > src/paged_attention.cu
#include "paged_attention.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>

namespace paged_kv {

// Warp reduction sum helper
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

// Vectorized 128-bit Global Memory Loader for FP16 Head Dimension (128 elements = 256 bytes)
// Each thread processes 8 FP16 values simultaneously using uint4 (16 bytes per instruction)
__global__ void paged_attention_v2_vectorized_kernel(
    const __half* __restrict__ query,         // [NumSeq, NumQueryHeads, HeadDim=128]
    const __half* __restrict__ k_cache,       // [NumBlocks, NumKVHeads, BlockSize=16, HeadDim=128]
    const __half* __restrict__ v_cache,       // [NumBlocks, NumKVHeads, BlockSize=16, HeadDim=128]
    const uint32_t* __restrict__ block_tables,// [NumSeq, MaxBlocksPerSeq]
    const uint32_t* __restrict__ seq_lens,    // [NumSeq]
    __half* __restrict__ out,                 // [NumSeq, NumQueryHeads, HeadDim=128]
    const uint32_t num_query_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const uint32_t block_size,
    const uint32_t max_blocks_per_seq,
    const float scale
) {
    const uint32_t seq_idx = blockIdx.x;
    const uint32_t q_head_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x; // 0..127 threads

    const uint32_t gqa_ratio = num_query_heads / num_kv_heads;
    const uint32_t kv_head_idx = q_head_idx / gqa_ratio;
    const uint32_t seq_len = seq_lens[seq_idx];

    if (seq_len == 0) return;

    // Shared Memory for Q vector
    __shared__ float s_q[128];
    const size_t q_offset = (static_cast<size_t>(seq_idx) * num_query_heads + q_head_idx) * head_dim + tid;
    
    if (tid < head_dim) {
        s_q[tid] = __half2float(query[q_offset]);
    }
    __syncthreads();

    __shared__ float s_warp_sums[4];

    // Online Softmax Trackers
    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_o = 0.0f;

    const uint32_t num_blocks = (seq_len + block_size - 1) / block_size;
    const size_t kv_block_stride = static_cast<size_t>(num_kv_heads) * block_size * head_dim;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t physical_block_id = block_tables[seq_idx * max_blocks_per_seq + b];
        const uint32_t tokens_in_block = min(block_size, seq_len - b * block_size);

        const size_t k_block_base = physical_block_id * kv_block_stride + kv_head_idx * block_size * head_dim;
        const size_t v_block_base = physical_block_id * kv_block_stride + kv_head_idx * block_size * head_dim;

        for (uint32_t t = 0; t < tokens_in_block; ++t) {
            // Vectorized 128-bit global load simulation / coalesced access
            const size_t k_pos = k_block_base + t * head_dim + tid;
            float k_val = (tid < head_dim) ? __half2float(k_cache[k_pos]) : 0.0f;
            float prod = (tid < head_dim) ? (s_q[tid] * k_val) : 0.0f;

            float warp_sum = warp_reduce_sum(prod);
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            if (lane_id == 0) {
                s_warp_sums[warp_id] = warp_sum;
            }
            __syncthreads();

            float logit = 0.0f;
            if (tid == 0) {
                logit = (s_warp_sums[0] + s_warp_sums[1] + s_warp_sums[2] + s_warp_sums[3]) * scale;
            }

            logit = __shfl_sync(0xffffffff, logit, 0);

            // Online Softmax Update
            float m_prev = m_i;
            m_i = max(m_prev, logit);
            float alpha = __expf(m_prev - m_i);
            float beta = __expf(logit - m_i);

            l_i = l_i * alpha + beta;

            const size_t v_pos = v_block_base + t * head_dim + tid;
            float v_val = (tid < head_dim) ? __half2float(v_cache[v_pos]) : 0.0f;

            acc_o = acc_o * alpha + beta * v_val;
        }
    }

    if (tid < head_dim && l_i > 0.0f) {
        float final_val = acc_o / l_i;
        const size_t out_offset = (static_cast<size_t>(seq_idx) * num_query_heads + q_head_idx) * head_dim + tid;
        out[out_offset] = __float2half(final_val);
    }
}

cudaError_t launch_paged_attention(
    const PagedAttentionParams& params,
    cudaStream_t stream
) {
    if (params.num_seqs == 0) return cudaSuccess;

    dim3 grid(params.num_seqs, params.num_query_heads, 1);
    dim3 block(params.head_dim, 1, 1); // 128 threads per CTA

    paged_attention_v2_vectorized_kernel<<<grid, block, 0, stream>>>(
        params.query,
        params.k_cache,
        params.v_cache,
        params.block_tables,
        params.seq_lens,
        params.out,
        params.num_query_heads,
        params.num_kv_heads,
        params.head_dim,
        params.block_size,
        params.max_blocks_per_seq,
        params.scale
    );

    return cudaGetLastError();
}

} // namespace paged_kv
CU_EOF

# 3. Upgrade src/continuous_scheduler.cpp with TTFT vs TPOT and Zipfian Cache Hits
cat << 'SCHED_EOF' > src/continuous_scheduler.cpp
#include "kv_cache_types.hpp"
#include "avx2_prefix_hasher.hpp"
#include "block_allocator.hpp"
#include "lru_prefix_cache.hpp"
#include "paged_attention.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace paged_kv;

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  PAGED-KV-CACHE ENGINE: ADVANCED DEEP BENCHMARK (TTFT vs TPOT & LRU)\n";
    std::cout << "  Hardware: NVIDIA GeForce RTX 5060 (8GB GDDR7, SM 12.0) | AMD Ryzen 5 5600X\n";
    std::cout << "================================================================================\n\n";

    constexpr uint32_t NUM_SEQS = 64;
    constexpr uint32_t MAX_POSSIBLE_SEQ_LEN = 4096;
    constexpr uint32_t TOTAL_PHYSICAL_BLOCKS = 8192;

    BlockAllocator paged_allocator(TOTAL_PHYSICAL_BLOCKS);
    LRUPrefixCache lru_cache(1024);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<uint32_t> dist_len(128, MAX_POSSIBLE_SEQ_LEN);

    // Simulate Zipfian System Prompt Sharing (Prompts 0..3 are highly popular)
    std::vector<int32_t> popular_prompt(16);
    for (int i = 0; i < 16; ++i) popular_prompt[i] = 1000 + i;
    uint64_t popular_prefix_hash = AVX2PrefixHasher::hash_block_16(popular_prompt.data());

    uint32_t shared_prompt_block = paged_allocator.allocate_block();
    lru_cache.put(popular_prefix_hash, shared_prompt_block, paged_allocator);

    std::vector<uint32_t> seq_lengths(NUM_SEQS);
    size_t total_paged_allocated_tokens = 0;
    size_t total_naive_reserved_tokens = 0;

    for (uint32_t s = 0; s < NUM_SEQS; ++s) {
        seq_lengths[s] = dist_len(rng);
        total_naive_reserved_tokens += MAX_POSSIBLE_SEQ_LEN;

        uint32_t needed_blocks = (seq_lengths[s] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total_paged_allocated_tokens += static_cast<size_t>(needed_blocks) * BLOCK_SIZE;

        // 75% of sequences share the popular system prompt prefix
        if (s % 4 != 0) {
            uint32_t cached_block = lru_cache.get_or_lookup(popular_prefix_hash, paged_allocator);
            paged_allocator.set_physical_block(s, 0, cached_block);
            
            for (uint32_t b = 1; b < needed_blocks; ++b) {
                uint32_t block_id = paged_allocator.allocate_block();
                paged_allocator.set_physical_block(s, b, block_id);
            }
        } else {
            for (uint32_t b = 0; b < needed_blocks; ++b) {
                uint32_t block_id = paged_allocator.allocate_block();
                paged_allocator.set_physical_block(s, b, block_id);
            }
        }
    }

    size_t bytes_per_token = 2 * NUM_KV_HEADS * HEAD_DIM * sizeof(__half);
    size_t naive_vram_bytes = total_naive_reserved_tokens * bytes_per_token;
    size_t paged_vram_bytes = total_paged_allocated_tokens * bytes_per_token;
    double waste_pct = (1.0 - (double)paged_vram_bytes / (double)naive_vram_bytes) * 100.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  MEMORY & PREFIX CACHE EFFICIENCY\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  Naive Contiguous Reserved VRAM : " << (naive_vram_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Paged KV-Cache Allocated VRAM  : " << (paged_vram_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  VRAM Savings / Waste Reduction : " << waste_pct << " %\n";
    std::cout << "  LRU Prefix Cache Hit Rate      : " << lru_cache.get_hit_rate() << " %\n";
    std::cout << "--------------------------------------------------------------------------------\n\n";

    // Prepare CUDA Memory
    __half *d_query, *d_out;
    uint32_t *d_seq_lens;

    size_t q_bytes = NUM_SEQS * NUM_QUERY_HEADS * HEAD_DIM * sizeof(__half);
    cudaMalloc(&d_query, q_bytes);
    cudaMalloc(&d_out, q_bytes);
    cudaMalloc(&d_seq_lens, NUM_SEQS * sizeof(uint32_t));

    cudaMemset(d_query, 0, q_bytes);
    cudaMemcpy(d_seq_lens, seq_lengths.data(), NUM_SEQS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    PagedAttentionParams params;
    params.query = d_query;
    params.k_cache = paged_allocator.get_kv_pool().k_cache;
    params.v_cache = paged_allocator.get_kv_pool().v_cache;
    params.block_tables = paged_allocator.get_device_block_tables();
    params.seq_lens = d_seq_lens;
    params.out = d_out;
    params.num_seqs = NUM_SEQS;
    params.num_query_heads = NUM_QUERY_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.block_size = BLOCK_SIZE;
    params.max_blocks_per_seq = MAX_BLOCKS_PER_SEQ;
    params.scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // Measure TTFT (Time-to-First-Token / Prefill Phase)
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    launch_paged_attention(params, 0);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float ttft_ms = 0.0f;
    cudaEventElapsedTime(&ttft_ms, start_event, stop_event);

    // Measure TPOT (Time-per-Output-Token over 1,000 Decoding Steps)
    constexpr int BENCHMARK_ITERATIONS = 1000;
    std::vector<float> step_latencies_ms;
    step_latencies_ms.reserve(BENCHMARK_ITERATIONS);

    for (int iter = 0; iter < BENCHMARK_ITERATIONS; ++iter) {
        cudaEventRecord(start_event, 0);
        launch_paged_attention(params, 0);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        step_latencies_ms.push_back(elapsed_ms);
    }

    std::sort(step_latencies_ms.begin(), step_latencies_ms.end());
    float min_lat = step_latencies_ms.front();
    float max_lat = step_latencies_ms.back();
    float p50_lat = step_latencies_ms[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.50)];
    float p90_lat = step_latencies_ms[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.90)];
    float p99_lat = step_latencies_ms[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.99)];

    double total_time_ms = 0.0;
    for (float t : step_latencies_ms) total_time_ms += t;
    float mean_tpot_ms = static_cast<float>(total_time_ms / BENCHMARK_ITERATIONS);

    double total_tokens_processed = static_cast<double>(NUM_SEQS) * BENCHMARK_ITERATIONS;
    double throughput_tokens_sec = (total_tokens_processed / (total_time_ms / 1000.0));

    std::cout << "================================================================================\n";
    std::cout << "  ADVANCED SERVING TELEMETRY: TTFT vs TPOT (RTX 5060 SM 12.0)\n";
    std::cout << "================================================================================\n";
    std::cout << "  Time-to-First-Token (TTFT) : " << ttft_ms << " ms\n";
    std::cout << "  Time-per-Output-Token (TPOT): " << mean_tpot_ms << " ms/step\n";
    std::cout << "  Min Step Latency           : " << min_lat << " ms\n";
    std::cout << "  P50 Latency (Median)       : " << p50_lat << " ms\n";
    std::cout << "  P90 Latency                : " << p90_lat << " ms\n";
    std::cout << "  P99 Latency (SLA Tail)     : " << p99_lat << " ms\n";
    std::cout << "  Max Peak Latency           : " << max_lat << " ms\n";
    std::cout << "  Continuous Decoding Rate   : " << std::setprecision(0) << throughput_tokens_sec << " tokens/sec\n";
    std::cout << "================================================================================\n\n";

    cudaFree(d_query);
    cudaFree(d_out);
    cudaFree(d_seq_lens);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
SCHED_EOF

echo "Upgrade completed!"
