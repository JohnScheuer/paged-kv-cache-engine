#include "kv_cache_types.hpp"
#include "avx2_prefix_hasher.hpp"
#include "block_allocator.hpp"
#include "lru_prefix_cache.hpp"
#include "paged_attention.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <string>

using namespace paged_kv;

int main() {
    int device_id = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);

    double total_vram_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    char hw_header[256];
    std::snprintf(
        hw_header,
        sizeof(hw_header),
        "%s (%.2f GB VRAM, SM %d.%d)",
        prop.name,
        total_vram_gb,
        prop.major,
        prop.minor
    );

    char sm_badge[64];
    std::snprintf(sm_badge, sizeof(sm_badge), "%s SM %d.%d", prop.name, prop.major, prop.minor);

    std::cout << "================================================================================\n";
    std::cout << "  PAGED-KV-CACHE ENGINE: ADVANCED DEEP BENCHMARK (TTFT vs TPOT & LRU)\n";
    std::cout << "  Hardware: " << hw_header << "\n";
    std::cout << "================================================================================\n\n";

    constexpr uint32_t NUM_SEQS = 64;
    constexpr uint32_t MAX_POSSIBLE_SEQ_LEN = 4096;
    constexpr uint32_t TOTAL_PHYSICAL_BLOCKS = 16384;

    BlockAllocator paged_allocator(TOTAL_PHYSICAL_BLOCKS);
    LRUPrefixCache lru_cache(1024);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<uint32_t> dist_len(128, MAX_POSSIBLE_SEQ_LEN);

    // Popular system prompt
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

    // Measure TTFT
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    launch_paged_attention(params, 0);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float ttft_ms = 0.0f;
    cudaEventElapsedTime(&ttft_ms, start_event, stop_event);

    // Measure TPOT over 1,000 Steps
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

    std::vector<float> sorted_latencies = step_latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    
    float min_lat = sorted_latencies.front();
    float max_lat = sorted_latencies.back();
    float p50_lat = sorted_latencies[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.50)];
    float p90_lat = sorted_latencies[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.90)];
    float p99_lat = sorted_latencies[static_cast<size_t>(BENCHMARK_ITERATIONS * 0.99)];

    double total_time_ms = 0.0;
    for (float t : step_latencies_ms) total_time_ms += t;
    float mean_tpot_ms = static_cast<float>(total_time_ms / BENCHMARK_ITERATIONS);

    double total_tokens_processed = static_cast<double>(NUM_SEQS) * BENCHMARK_ITERATIONS;
    double throughput_tokens_sec = (total_tokens_processed / (total_time_ms / 1000.0));

    std::cout << "================================================================================\n";
    std::cout << "  ADVANCED SERVING TELEMETRY: TTFT vs TPOT (" << sm_badge << ")\n";
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

    // Export Step Latencies to CSV
    std::ofstream lat_file("step_latencies.csv");
    lat_file << "step_idx,latency_ms\n";
    for (size_t i = 0; i < step_latencies_ms.size(); ++i) {
        lat_file << i << "," << step_latencies_ms[i] << "\n";
    }
    lat_file.close();

    // Export Summary Benchmark Metrics to CSV
    std::ofstream sum_file("benchmark_summary.csv");
    sum_file << "metric,value\n";
    sum_file << "naive_vram_mb," << (naive_vram_bytes / (1024.0 * 1024.0)) << "\n";
    sum_file << "paged_vram_mb," << (paged_vram_bytes / (1024.0 * 1024.0)) << "\n";
    sum_file << "vram_savings_pct," << waste_pct << "\n";
    sum_file << "lru_hit_rate_pct," << lru_cache.get_hit_rate() << "\n";
    sum_file << "ttft_ms," << ttft_ms << "\n";
    sum_file << "tpot_ms," << mean_tpot_ms << "\n";
    sum_file << "min_latency_ms," << min_lat << "\n";
    sum_file << "p50_latency_ms," << p50_lat << "\n";
    sum_file << "p90_latency_ms," << p90_lat << "\n";
    sum_file << "p99_latency_ms," << p99_lat << "\n";
    sum_file << "max_latency_ms," << max_lat << "\n";
    sum_file << "throughput_tokens_sec," << throughput_tokens_sec << "\n";
    sum_file.close();

    std::cout << "[INFO] Empirical benchmark metrics exported to step_latencies.csv & benchmark_summary.csv\n";

    cudaFree(d_query);
    cudaFree(d_out);
    cudaFree(d_seq_lens);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}