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
