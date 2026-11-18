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
