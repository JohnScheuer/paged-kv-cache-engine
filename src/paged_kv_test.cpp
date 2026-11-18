#include "kv_cache_types.hpp"
#include "avx2_prefix_hasher.hpp"
#include "block_allocator.hpp"
#include "paged_attention.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <chrono>

using namespace paged_kv;

// High-precision FP64 CPU Reference Attention Implementation for Accuracy Verification
void cpu_reference_attention(
    const std::vector<float>& h_query,    // [NumSeq, NumQHeads, HeadDim]
    const std::vector<float>& h_k_all,    // [NumSeq, SeqLen, NumKVHeads, HeadDim]
    const std::vector<float>& h_v_all,    // [NumSeq, SeqLen, NumKVHeads, HeadDim]
    std::vector<float>& h_out,            // [NumSeq, NumQHeads, HeadDim]
    uint32_t num_seqs, uint32_t seq_len, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, float scale
) {
    uint32_t gqa_ratio = num_q_heads / num_kv_heads;

    for (uint32_t s = 0; s < num_seqs; ++s) {
        for (uint32_t qh = 0; qh < num_q_heads; ++qh) {
            uint32_t kvh = qh / gqa_ratio;

            std::vector<double> logits(seq_len, 0.0);
            double max_logit = -1e30;

            // Step 1: Compute Q . K^T
            for (uint32_t t = 0; t < seq_len; ++t) {
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    size_t q_idx = (s * num_q_heads + qh) * head_dim + d;
                    size_t k_idx = ((s * seq_len + t) * num_kv_heads + kvh) * head_dim + d;
                    dot += static_cast<double>(h_query[q_idx]) * static_cast<double>(h_k_all[k_idx]);
                }
                logits[t] = dot * static_cast<double>(scale);
                if (logits[t] > max_logit) max_logit = logits[t];
            }

            // Step 2: Softmax
            double sum_exp = 0.0;
            std::vector<double> probs(seq_len, 0.0);
            for (uint32_t t = 0; t < seq_len; ++t) {
                probs[t] = std::exp(logits[t] - max_logit);
                sum_exp += probs[t];
            }

            for (uint32_t t = 0; t < seq_len; ++t) {
                probs[t] /= sum_exp;
            }

            // Step 3: Weighted Sum of V
            for (uint32_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (uint32_t t = 0; t < seq_len; ++t) {
                    size_t v_idx = ((s * seq_len + t) * num_kv_heads + kvh) * head_dim + d;
                    acc += probs[t] * static_cast<double>(h_v_all[v_idx]);
                }
                size_t out_idx = (s * num_q_heads + qh) * head_dim + d;
                h_out[out_idx] = static_cast<float>(acc);
            }
        }
    }
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  PAGED-KV-CACHE ENGINE: AUTOMATED DIAGNOSTIC & VERIFICATION SUITE\n";
    std::cout << "  Author: Joao Felipe de Souza (github.com/JohnScheuer)\n";
    std::cout << "================================================================================\n\n";

    // -------------------------------------------------------------------------
    // TEST 1: Physical Block Allocator & OOM Safety
    // -------------------------------------------------------------------------
    std::cout << "[RUNNING TEST 1/3] Physical Block Allocator & Ref-Counting Integrity...\n";
    {
        constexpr uint32_t TOTAL_BLOCKS = 32;
        BlockAllocator allocator(TOTAL_BLOCKS);

        assert(allocator.get_free_blocks() == TOTAL_BLOCKS);
        
        uint32_t b0 = allocator.allocate_block();
        uint32_t b1 = allocator.allocate_block();
        
        assert(allocator.get_ref_count(b0) == 1);
        assert(allocator.get_free_blocks() == TOTAL_BLOCKS - 2);

        allocator.share_block(b0);
        assert(allocator.get_ref_count(b0) == 2);

        allocator.free_block(b0);
        assert(allocator.get_ref_count(b0) == 1);
        assert(allocator.get_free_blocks() == TOTAL_BLOCKS - 2);

        allocator.free_block(b0);
        assert(allocator.get_ref_count(b0) == 0);
        assert(allocator.get_free_blocks() == TOTAL_BLOCKS - 1);

        allocator.free_block(b1);
        assert(allocator.get_free_blocks() == TOTAL_BLOCKS);
        std::cout << "  -> Physical Allocation & Free List Logic: PASSED.\n";
    }

    // -------------------------------------------------------------------------
    // TEST 2: AVX2 Prefix Hasher & Copy-on-Write (CoW) Detach Verification
    // -------------------------------------------------------------------------
    std::cout << "[RUNNING TEST 2/3] AVX2 Prefix Hasher & Copy-on-Write (CoW) Verification...\n";
    {
        alignas(32) int32_t prompt_tokens_a[16] = {101, 2054, 2003, 1037, 3899, 1200, 4500, 99, 12, 45, 88, 123, 900, 1, 2, 3};
        alignas(32) int32_t prompt_tokens_b[16] = {101, 2054, 2003, 1037, 3899, 1200, 4500, 99, 12, 45, 88, 123, 900, 1, 2, 3};

        uint64_t hash_a = AVX2PrefixHasher::hash_block_16(prompt_tokens_a);
        uint64_t hash_b = AVX2PrefixHasher::hash_block_16(prompt_tokens_b);

        assert(hash_a == hash_b);

        BlockAllocator allocator(64);
        uint32_t shared_block = allocator.allocate_block();
        allocator.register_prefix(hash_a, shared_block);

        // Sequence 0 and Sequence 1 share the prompt prefix
        allocator.set_physical_block(0, 0, shared_block);
        uint32_t look_block = allocator.lookup_or_register_prefix(hash_b);
        allocator.set_physical_block(1, 0, look_block);

        assert(allocator.get_ref_count(shared_block) == 2);

        // Sequence 1 diverges -> Trigger Copy-on-Write
        uint32_t cow_block = allocator.trigger_cow_if_needed(1, 0);
        assert(cow_block != shared_block);
        assert(allocator.get_ref_count(shared_block) == 1);
        assert(allocator.get_ref_count(cow_block) == 1);

        std::cout << "  -> AVX2 SIMD Hash: 0x" << std::hex << hash_a << std::dec << " | CoW Detach: PASSED.\n";
    }

    // -------------------------------------------------------------------------
    // TEST 3: CUDA PagedAttention Numerical Accuracy vs FP64 CPU Reference
    // -------------------------------------------------------------------------
    std::cout << "[RUNNING TEST 3/3] CUDA PagedAttention Numerical Accuracy vs FP64 CPU Reference...\n";
    {
        constexpr uint32_t NUM_SEQS = 2;
        constexpr uint32_t SEQ_LEN = 32; // 2 blocks of 16 tokens
        constexpr uint32_t NUM_Q_HEADS = 8;
        constexpr uint32_t NUM_KV_HEADS = 2;
        constexpr uint32_t HEAD_DIM = 128;
        constexpr uint32_t BLOCK_SIZE = 16;
        constexpr float SCALE = 1.0f / 11.3137085f;

        BlockAllocator allocator(128);

        // Host buffers
        std::vector<float> h_q(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM);
        std::vector<float> h_k(NUM_SEQS * SEQ_LEN * NUM_KV_HEADS * HEAD_DIM);
        std::vector<float> h_v(NUM_SEQS * SEQ_LEN * NUM_KV_HEADS * HEAD_DIM);
        std::vector<float> h_out_cpu(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM);

        // Fill deterministic mock tensors
        for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.5f;
        for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = std::cos(static_cast<float>(i) * 0.05f) * 0.5f;
        for (size_t i = 0; i < h_v.size(); ++i) h_v[i] = std::sin(static_cast<float>(i) * 0.02f) * 0.5f;

        // Compute FP64 CPU Reference Attention
        cpu_reference_attention(h_q, h_k, h_v, h_out_cpu, NUM_SEQS, SEQ_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALE);

        // Populate Paged KV-Cache in VRAM
        std::vector<uint32_t> h_seq_lens = {SEQ_LEN, SEQ_LEN};

        for (uint32_t s = 0; s < NUM_SEQS; ++s) {
            uint32_t num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (uint32_t b = 0; b < num_blocks; ++b) {
                uint32_t p_block = allocator.allocate_block();
                allocator.set_physical_block(s, b, p_block);

                // Copy K & V data into physical block VRAM
                std::vector<__half> block_k(NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM);
                std::vector<__half> block_v(NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM);

                for (uint32_t t = 0; t < BLOCK_SIZE; ++t) {
                    uint32_t global_t = b * BLOCK_SIZE + t;
                    for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
                        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                            size_t src_k_idx = ((s * SEQ_LEN + global_t) * NUM_KV_HEADS + h) * HEAD_DIM + d;
                            size_t dst_k_idx = (h * BLOCK_SIZE + t) * HEAD_DIM + d;
                            block_k[dst_k_idx] = __float2half(h_k[src_k_idx]);
                            block_v[dst_k_idx] = __float2half(h_v[src_k_idx]);
                        }
                    }
                }

                size_t block_num_elem = NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM;
                cudaMemcpy(allocator.get_kv_pool().k_cache + p_block * block_num_elem, block_k.data(), block_num_elem * sizeof(__half), cudaMemcpyHostToDevice);
                cudaMemcpy(allocator.get_kv_pool().v_cache + p_block * block_num_elem, block_v.data(), block_num_elem * sizeof(__half), cudaMemcpyHostToDevice);
            }
        }

        // Allocate Device Query and Output tensors
        __half *d_q, *d_out;
        uint32_t *d_seq_lens;
        size_t q_bytes = NUM_SEQS * NUM_Q_HEADS * HEAD_DIM * sizeof(__half);
        
        cudaMalloc(&d_q, q_bytes);
        cudaMalloc(&d_out, q_bytes);
        cudaMalloc(&d_seq_lens, NUM_SEQS * sizeof(uint32_t));

        std::vector<__half> h_q_fp16(h_q.size());
        for (size_t i = 0; i < h_q.size(); ++i) h_q_fp16[i] = __float2half(h_q[i]);

        cudaMemcpy(d_q, h_q_fp16.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seq_lens, h_seq_lens.data(), NUM_SEQS * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Setup Launch Parameters
        PagedAttentionParams params;
        params.query = d_q;
        params.k_cache = allocator.get_kv_pool().k_cache;
        params.v_cache = allocator.get_kv_pool().v_cache;
        params.block_tables = allocator.get_device_block_tables();
        params.seq_lens = d_seq_lens;
        params.out = d_out;
        params.num_seqs = NUM_SEQS;
        params.num_query_heads = NUM_Q_HEADS;
        params.num_kv_heads = NUM_KV_HEADS;
        params.head_dim = HEAD_DIM;
        params.block_size = BLOCK_SIZE;
        params.max_blocks_per_seq = MAX_BLOCKS_PER_SEQ;
        params.scale = SCALE;

        // Launch Kernel
        launch_paged_attention(params, 0);
        cudaDeviceSynchronize();

        // Download output and compare against CPU reference
        std::vector<__half> h_out_gpu_fp16(h_q.size());
        cudaMemcpy(h_out_gpu_fp16.data(), d_out, q_bytes, cudaMemcpyDeviceToHost);

        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < h_out_gpu_fp16.size(); ++i) {
            float gpu_val = __half2float(h_out_gpu_fp16[i]);
            float cpu_val = h_out_cpu[i];
            float diff = std::abs(gpu_val - cpu_val);
            if (diff > max_abs_diff) max_abs_diff = diff;
        }

        std::cout << "  -> Max Absolute Error FP16 GPU vs FP64 CPU Ref: " << max_abs_diff << "\n";
        assert(max_abs_diff < 1e-3f);
        std::cout << "  -> CUDA PagedAttention Numerical Check: PASSED.\n";

        cudaFree(d_q);
        cudaFree(d_out);
        cudaFree(d_seq_lens);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "  DIAGNOSTIC TEST SUMMARY: ALL 3 TESTS PASSED (100% SUCCESS RATE)\n";
    std::cout << "================================================================================\n";

    return 0;
}
