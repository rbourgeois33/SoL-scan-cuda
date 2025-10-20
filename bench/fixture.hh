#pragma once

#include <benchmark/benchmark.h>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "test_helpers.cuh"

class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench_scan(benchmark::State& st, FUNC callback, int size, Args&&... args)
    {
        constexpr int val = 1;
        const raft::handle_t handle{};
        rmm::device_uvector<int> buffer(size, handle.get_stream());
        fill_buffer(handle, buffer, val);

        for (auto _ : st)
        {
            st.PauseTiming();
            fill_buffer(handle, buffer, val);
            st.ResumeTiming();
            callback(buffer);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        std::vector<int> expected(size);
        std::iota(expected.begin(), expected.end(), 1);
        if (!no_check)
            check_buffer(buffer, expected, st);
    }

    template <typename FUNC>
    void register_scan(benchmark::State& st, FUNC func)
    {
        int size = st.range(0);
        this->bench_scan(st, func, size);
    }
};

bool Fixture::no_check = false;