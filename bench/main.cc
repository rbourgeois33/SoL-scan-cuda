#include "benchmark_registerer.hh"
#include "fixture.hh"
#include "main_helper.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

#include <rmm/mr/device/pool_memory_resource.hpp>

int main(int argc, char** argv)
{
    // Google bench setup
    using benchmark_t = benchmark::internal::Benchmark;
    ::benchmark::Initialize(&argc, argv);

    // RMM Setup
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    bool bench_nsight = parse_arguments(argc, argv);

    // Benchmarks registration
    Fixture fx;
    {
        constexpr std::array sizes = {
            // 4,
            // 16,
            // 64,
            // 63,
            // 65,
            // 128,
            // 2048,
            // 524288,
            1024*1024*1024
        };

        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple scan_to_bench{
            "DLB",
            &DLB,
        };

        //  / 2 because we store name + function pointer
        benchmark_t* b[tuple_length(scan_to_bench) / 2];
        int function_index = 0;

        // Call to registerer
        registerer_scan(&fx,
                        b,
                        function_index,
                        sizes,
                        bench_nsight,
                        scan_to_bench);
    }
    ::benchmark::RunSpecifiedBenchmarks();
}
