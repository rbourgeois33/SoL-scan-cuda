cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(tp_scan LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main)

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_scan)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
# Set the C++ standard
set(CXX_STANDARD 17)
# Add fconcept for lambdas
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
# Generate cubin for this specific GPU
set(CMAKE_CUDA_ARCHITECTURES "native")
# Set fpic for Google Benchmark
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(to_bench STATIC
  src/to_bench.cu)
target_link_libraries(to_bench PUBLIC rmm::rmm raft::raft benchmark::benchmark)

add_library(test_helpers STATIC
  bench/test_helpers.cu)
target_link_libraries(test_helpers PUBLIC rmm::rmm raft::raft benchmark::benchmark)

include_directories(src)

add_executable(bench bench/main.cc)
target_link_libraries(bench to_bench test_helpers benchmark::benchmark)