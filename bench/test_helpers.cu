#include "test_helpers.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

#include <benchmark/benchmark.h>

#include <iostream>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

template <typename T>
static auto host_copy(const T* device_ptr, size_t size, rmm::cuda_stream_view stream_view)
{
  if (!device_ptr) return std::vector<T>{};
  std::vector<T> host_vec(size);
  raft::copy(host_vec.data(), device_ptr, size, stream_view);
  stream_view.synchronize();
  return host_vec;
}

template <typename T>
static auto host_copy(const rmm::device_uvector<T>& device_vec)
{
  return host_copy(device_vec.data(), device_vec.size(), device_vec.stream());
}

template <typename T>
void check_buffer(const rmm::device_uvector<T>& buffer,
                  const std::vector<T>& expected,
                  benchmark::State& st)
{
    const auto& host_buffer = host_copy(buffer);

    if (!std::equal(host_buffer.cbegin(),
                    host_buffer.cend(),
                    expected.cbegin()))
    {
        auto [first, second] = std::mismatch(host_buffer.cbegin(),
                                             host_buffer.cend(),
                                             expected.cbegin());
        std::cout << "Error at " << first - host_buffer.cbegin() << ": "
                  << *first << " " << *second << std::endl;
        st.SkipWithError("Failed test");
    }
}

template <typename T>
void fill_buffer(const raft::handle_t& handle,
                 rmm::device_uvector<T>& buffer,
                 T val)
{
    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               buffer.begin(),
                               buffer.end(),
                               val);
}

template void check_buffer(const rmm::device_uvector<int>& scalar,
                           const std::vector<int>& expected,
                           benchmark::State& st);

template void fill_buffer(const raft::handle_t& handle,
                          rmm::device_uvector<int>& buffer,
                          int val);