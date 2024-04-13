#pragma once

#include <chrono>

namespace captains::llog {

using Duration = std::chrono::nanoseconds;

Duration stamp(void);

}
