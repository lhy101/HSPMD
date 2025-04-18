#pragma once

#include "hspmd/common/macros.h"
#include <random>

namespace hspmd {
namespace impl {

void SetCPURandomSeed(uint64_t seed);
uint64_t GenNextRandomSeed();

} // namespace impl
} // namespace hspmd
