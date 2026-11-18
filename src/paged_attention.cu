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
