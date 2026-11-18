cmake_minimum_required(VERSION 3.22)
project(paged_kv_cache_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
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

