#pragma once

#include <chrono>

namespace captains::log {

using Duration = std::chrono::nanoseconds;

Duration stamp(void);

}
