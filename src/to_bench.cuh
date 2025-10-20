#pragma once

#include <rmm/device_uvector.hpp>

void baseline_scan(rmm::device_uvector<int>& buffer);

void your_scan(rmm::device_uvector<int>& buffer);