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
