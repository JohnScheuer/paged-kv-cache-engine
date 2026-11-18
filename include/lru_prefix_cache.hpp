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
