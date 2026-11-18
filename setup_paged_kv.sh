#!/usr/bin/env bash
set -euo pipefail

# 1. Create project directories
mkdir -p ~/paged-kv-cache-engine/{include,src,scripts,assets,build}

cd ~/paged-kv-cache-engine

# 2. Generate CMakeLists.txt
cat << 'CMAKE_EOF' > CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(paged_kv_cache_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization Flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -Wall -Wextra -pthread")

# CUDA SM 12.0 (RTX 5060 Blackwell) & SM 8.9/9.0 Backward Compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89;90;120")
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo -Xcompiler=-O3,-mavx2,-mfma")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA PagedAttention Kernel & Core Library
add_library(paged_attention_kernel STATIC
  src/paged_attention.cu
)

set_target_properties(paged_attention_kernel PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Automated Test Suite Target
add_executable(paged_kv_test
  src/paged_kv_test.cpp
)
target_link_libraries(paged_kv_test PRIVATE paged_attention_kernel ${CUDA_LIBRARIES})

# Continuous Scheduler & Benchmark Target
add_executable(paged_kv_benchmark
  src/continuous_scheduler.cpp
)
target_link_libraries(paged_kv_benchmark PRIVATE paged_attention_kernel ${CUDA_LIBRARIES})

CMAKE_EOF

# 3. Generate include/kv_cache_types.hpp
cat << 'TYPES_EOF' > include/kv_cache_types.hpp
#ifndef KV_CACHE_TYPES_HPP
#define KV_CACHE_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <vector>

namespace paged_kv {

// Constants for PagedAttention Architecture
constexpr uint32_t BLOCK_SIZE = 16;          // Tokens per physical memory block
constexpr uint32_t HEAD_DIM = 128;           // Head dimension
constexpr uint32_t NUM_QUERY_HEADS = 32;     // Query attention heads
constexpr uint32_t NUM_KV_HEADS = 8;         // Key/Value attention heads (GQA 4:1 ratio)
constexpr uint32_t GQA_RATIO = NUM_QUERY_HEADS / NUM_KV_HEADS; // 4
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 256; // Max 4096 tokens per sequence
constexpr uint32_t MAX_SEQUENCES = 64;       // Max concurrent batch sequences
constexpr uint32_t INVALID_BLOCK_ID = static_cast<uint32_t>(-1);

// Aligned Block Table metadata for host/device zero-copy sharing
struct alignas(64) BlockTable {
    uint32_t physical_blocks[MAX_BLOCKS_PER_SEQ];
    uint32_t num_blocks{0};

    BlockTable() {
        for (size_t i = 0; i < MAX_BLOCKS_PER_SEQ; ++i) {
            physical_blocks[i] = INVALID_BLOCK_ID;
        }
    }
};

// Sequence Metadata structure
struct alignas(64) SequenceMetadata {
    uint32_t seq_id{0};
    uint32_t prompt_len{0};
    uint32_t cur_seq_len{0};
    uint32_t block_table_idx{0};
};

// Physical KV-Cache Tensor Layout Descriptor
// Storage layout per block: [2 (K=0, V=1), NumKVHeads, BlockSize, HeadDim] in FP16
struct KVCachePoolParams {
    __half* k_cache{nullptr}; // Pointer to contiguous physical K cache pool
    __half* v_cache{nullptr}; // Pointer to contiguous physical V cache pool
    uint32_t num_total_blocks{0};
    uint32_t block_size{BLOCK_SIZE};
    uint32_t head_dim{HEAD_DIM};
    uint32_t num_kv_heads{NUM_KV_HEADS};

    size_t get_block_num_elements() const {
        return static_cast<size_t>(num_kv_heads) * block_size * head_dim;
    }

    size_t get_block_bytes() const {
        return get_block_num_elements() * sizeof(__half);
    }
};

// PagedAttention Launch Parameters
struct PagedAttentionParams {
    const __half* query{nullptr};       // [NumSeq, NumQueryHeads, HeadDim]
    const __half* k_cache{nullptr};     // Physical K Pool
    const __half* v_cache{nullptr};     // Physical V Pool
    const uint32_t* block_tables{nullptr}; // Flat array [NumSeq, MAX_BLOCKS_PER_SEQ]
    const uint32_t* seq_lens{nullptr};  // Array [NumSeq] of sequence lengths
    __half* out{nullptr};               // Output [NumSeq, NumQueryHeads, HeadDim]
    
    uint32_t num_seqs{0};
    uint32_t num_query_heads{NUM_QUERY_HEADS};
    uint32_t num_kv_heads{NUM_KV_HEADS};
    uint32_t head_dim{HEAD_DIM};
    uint32_t block_size{BLOCK_SIZE};
    uint32_t max_blocks_per_seq{MAX_BLOCKS_PER_SEQ};
    float scale{1.0f / 11.3137085f};    // 1 / sqrt(128)
};

} // namespace paged_kv

#endif // KV_CACHE_TYPES_HPP
TYPES_EOF

# 4. Generate include/avx2_prefix_hasher.hpp
cat << 'HASHER_EOF' > include/avx2_prefix_hasher.hpp
#ifndef AVX2_PREFIX_HASHER_HPP
#define AVX2_PREFIX_HASHER_HPP

#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <vector>
#include <span>

namespace paged_kv {

class AVX2PrefixHasher {
public:
    // Computes a 64-bit hash over 16 token IDs (64 bytes of 32-bit int tokens) using AVX2 SIMD
    static uint64_t hash_block_16(const int32_t* token_ids) {
        // Load 16x 32-bit integers into 2x 256-bit AVX2 registers
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(token_ids));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(token_ids + 8));

        // Constants for Murmur/FNV-inspired 32-bit integer mixing
        const __m256i prime_vec = _mm256_set1_epi32(0x01000193);
        const __m256i seed_vec  = _mm256_set1_epi32(0x811C9DC5);

        // Vectorized XOR and 32-bit multiplication
        __m256i m0 = _mm256_xor_si256(v0, seed_vec);
        m0 = _mm256_mullo_epi32(m0, prime_vec);

        __m256i m1 = _mm256_xor_si256(v1, seed_vec);
        m1 = _mm256_mullo_epi32(m1, prime_vec);

        // Fold two 256-bit vectors into one
        __m256i combined = _mm256_xor_si256(m0, m1);

        // Extract 4x 64-bit lanes onto host
        alignas(32) uint64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), combined);

        // Final scalar mix across 4 lanes
        uint64_t hash = 14695981039346656037ULL; // FNV-1a 64-bit offset basis
        for (int i = 0; i < 4; ++i) {
            hash ^= lanes[i];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    // Sequence cumulative prefix block hashing
    static uint64_t compute_prefix_hash(uint64_t prev_prefix_hash, uint64_t current_block_hash) {
        uint64_t combined = prev_prefix_hash ^ (current_block_hash + 0x9e3779b97f4a7c15ULL + (prev_prefix_hash << 6) + (prev_prefix_hash >> 2));
        return combined;
    }
};

} // namespace paged_kv

#endif // AVX2_PREFIX_HASHER_HPP
HASHER_EOF

# 5. Generate include/block_allocator.hpp
cat << 'ALLOC_EOF' > include/block_allocator.hpp
#ifndef BLOCK_ALLOCATOR_HPP
#define BLOCK_ALLOCATOR_HPP

#include "kv_cache_types.hpp"
#include "avx2_prefix_hasher.hpp"
#include <vector>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <cuda_runtime.h>

namespace paged_kv {

class BlockAllocator {
private:
    uint32_t total_blocks_;
    uint32_t free_blocks_count_;
    
    // Physical Memory Pool parameters (VRAM)
    KVCachePoolParams kv_pool_;
    
    // Free list of physical block IDs
    std::vector<uint32_t> free_list_;
    
    // Reference count for each physical block ID
    std::vector<uint32_t> ref_counts_;

    // Prefix Cache: Hash -> Physical Block ID
    std::unordered_map<uint64_t, uint32_t> prefix_cache_;

    // Host pinned mapped memory for block tables
    uint32_t* host_block_tables_{nullptr};
    uint32_t* device_block_tables_{nullptr};

public:
    BlockAllocator(uint32_t total_blocks) 
        : total_blocks_(total_blocks), free_blocks_count_(total_blocks) {
        
        ref_counts_.resize(total_blocks_, 0);
        free_list_.reserve(total_blocks_);
        for (int32_t i = static_cast<int32_t>(total_blocks_) - 1; i >= 0; --i) {
            free_list_.push_back(static_cast<uint32_t>(i));
        }

        // Initialize Physical VRAM Pool
        kv_pool_.num_total_blocks = total_blocks_;
        size_t block_bytes = kv_pool_.get_block_bytes();
        size_t total_vram_bytes = block_bytes * total_blocks_;

        cudaError_t err_k = cudaMalloc(&kv_pool_.k_cache, total_vram_bytes);
        cudaError_t err_v = cudaMalloc(&kv_pool_.v_cache, total_vram_bytes);

        if (err_k != cudaSuccess || err_v != cudaSuccess) {
            throw std::runtime_error("Failed to allocate KV-Cache VRAM Pool! Total requested: " + 
                                     std::to_string((total_vram_bytes * 2) / (1024 * 1024)) + " MB");
        }

        cudaMemset(kv_pool_.k_cache, 0, total_vram_bytes);
        cudaMemset(kv_pool_.v_cache, 0, total_vram_bytes);

        // Allocate Zero-Copy Pinned Mapped Memory for Block Tables
        size_t table_bytes = MAX_SEQUENCES * MAX_BLOCKS_PER_SEQ * sizeof(uint32_t);
        cudaHostAlloc(&host_block_tables_, table_bytes, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&device_block_tables_, host_block_tables_, 0);

        // Initialize with INVALID_BLOCK_ID
        for (size_t i = 0; i < MAX_SEQUENCES * MAX_BLOCKS_PER_SEQ; ++i) {
            host_block_tables_[i] = INVALID_BLOCK_ID;
        }
    }

    ~BlockAllocator() {
        if (kv_pool_.k_cache) cudaFree(kv_pool_.k_cache);
        if (kv_pool_.v_cache) cudaFree(kv_pool_.v_cache);
        if (host_block_tables_) cudaFreeHost(host_block_tables_);
    }

    // Disable copy
    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    // Allocate a free physical block
    uint32_t allocate_block() {
        if (free_list_.empty()) {
            throw std::runtime_error("Physical KV-Cache VRAM Out of Memory (OOM)! No free blocks available.");
        }
        uint32_t block_id = free_list_.back();
        free_list_.pop_back();
        --free_blocks_count_;
        ref_counts_[block_id] = 1;
        return block_id;
    }

    // Free / Decrement Reference Count
    void free_block(uint32_t block_id) {
        if (block_id >= total_blocks_) return;
        if (ref_counts_[block_id] == 0) return;

        --ref_counts_[block_id];
        if (ref_counts_[block_id] == 0) {
            free_list_.push_back(block_id);
            ++free_blocks_count_;
        }
    }

    // Increment Ref Count for Copy-on-Write / Shared Prefix
    void share_block(uint32_t block_id) {
        if (block_id < total_blocks_) {
            ++ref_counts_[block_id];
        }
    }

    // Copy-on-Write (CoW) Trigger: If block ref_count > 1, duplicate and detach
    uint32_t trigger_cow_if_needed(uint32_t seq_idx, uint32_t logical_block_idx, cudaStream_t stream = 0) {
        uint32_t old_block_id = get_physical_block(seq_idx, logical_block_idx);
        if (old_block_id == INVALID_BLOCK_ID) {
            uint32_t new_block_id = allocate_block();
            set_physical_block(seq_idx, logical_block_idx, new_block_id);
            return new_block_id;
        }

        if (ref_counts_[old_block_id] == 1) {
            // Unique owner, no CoW needed
            return old_block_id;
        }

        // Ref Count > 1: Duplicate block into new physical allocation
        uint32_t new_block_id = allocate_block();
        size_t block_bytes = kv_pool_.get_block_bytes();

        // Copy FP16 K and V tensors device-to-device asynchronously
        cudaMemcpyAsync(kv_pool_.k_cache + (new_block_id * kv_pool_.get_block_num_elements()),
                        kv_pool_.k_cache + (old_block_id * kv_pool_.get_block_num_elements()),
                        block_bytes, cudaMemcpyDeviceToDevice, stream);

        cudaMemcpyAsync(kv_pool_.v_cache + (new_block_id * kv_pool_.get_block_num_elements()),
                        kv_pool_.v_cache + (old_block_id * kv_pool_.get_block_num_elements()),
                        block_bytes, cudaMemcpyDeviceToDevice, stream);

        // Update block table and decrement old block reference
        set_physical_block(seq_idx, logical_block_idx, new_block_id);
        free_block(old_block_id);

        return new_block_id;
    }

    // Prefix Caching Lookup
    uint32_t lookup_or_register_prefix(uint64_t prefix_hash) {
        auto it = prefix_cache_.find(prefix_hash);
        if (it != prefix_cache_.end()) {
            uint32_t physical_block_id = it->second;
            share_block(physical_block_id);
            return physical_block_id;
        }
        return INVALID_BLOCK_ID;
    }

    void register_prefix(uint64_t prefix_hash, uint32_t physical_block_id) {
        prefix_cache_[prefix_hash] = physical_block_id;
    }

    // Block Table setters/getters
    void set_physical_block(uint32_t seq_idx, uint32_t logical_block_idx, uint32_t physical_block_id) {
        host_block_tables_[seq_idx * MAX_BLOCKS_PER_SEQ + logical_block_idx] = physical_block_id;
    }

    uint32_t get_physical_block(uint32_t seq_idx, uint32_t logical_block_idx) const {
        return host_block_tables_[seq_idx * MAX_BLOCKS_PER_SEQ + logical_block_idx];
    }

    // Accessors
    uint32_t get_total_blocks() const { return total_blocks_; }
    uint32_t get_free_blocks() const { return free_blocks_count_; }
    uint32_t get_used_blocks() const { return total_blocks_ - free_blocks_count_; }
    uint32_t get_ref_count(uint32_t block_id) const { return ref_counts_[block_id]; }
    const KVCachePoolParams& get_kv_pool() const { return kv_pool_; }
    const uint32_t* get_device_block_tables() const { return device_block_tables_; }
    uint32_t* get_host_block_tables() { return host_block_tables_; }
};

} // namespace paged_kv

#endif // BLOCK_ALLOCATOR_HPP
ALLOC_EOF

chmod +x setup_paged_kv.sh
./setup_paged_kv.sh
