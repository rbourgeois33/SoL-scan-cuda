#pragma once

#include <cmath>
#include <tuple>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

template <typename Tuple>
constexpr auto tuple_length(Tuple)
{
    return std::tuple_size_v<Tuple>;
}

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }
inline auto make_pool()
{
  // 128MB of initial pool size
  const size_t initial_pool_size = 128 * 1024 * 1024;
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_async(),
                                                                     initial_pool_size);
}

bool parse_arguments(int argc, char* argv[])
{
    bool bench_nsight = false;
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
        // Set iteration number to 1 not to mess with nsight
        if (argv[i] == std::string_view("--bench-nsight"))
        {
            bench_nsight = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    return bench_nsight;
}