#pragma once

#include <benchmark/benchmark.h>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

template <typename T>
void check_buffer(const rmm::device_uvector<T>& buffer,
                  const std::vector<T>& expected,
                  benchmark::State& st);

template <typename T>
void fill_buffer(const raft::handle_t& handle,
                 rmm::device_uvector<T>& buffer,
                 T value);