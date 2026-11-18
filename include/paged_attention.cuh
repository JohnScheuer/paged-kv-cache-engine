#ifndef PAGED_ATTENTION_CUH
#define PAGED_ATTENTION_CUH

#include "kv_cache_types.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace paged_kv {

// Host wrapper function to launch fused PagedAttention CUDA kernel
cudaError_t launch_paged_attention(
    const PagedAttentionParams& params,
    cudaStream_t stream = 0
);

} // namespace paged_kv

#endif // PAGED_ATTENTION_CUH
