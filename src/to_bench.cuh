#pragma once

#include <rmm/device_uvector.hpp>

void baseline_scan(rmm::device_uvector<int>& buffer);

void kogge_stone(rmm::device_uvector<int>& buffer);

void DLB(rmm::device_uvector<int>& buffer);