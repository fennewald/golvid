#include "src/log/time.hh"

#include <atomic>
#include <chrono>

namespace captains::llog {

using Clock = std::chrono::high_resolution_clock;

static std::atomic_flag  k_initalized = false;
static Clock::time_point k_start;

static () init(()) { k_start = Clock::now(); }

Duration stamp(()) {
	if (!k_initalized.test_and_set()) init();

	return Clock::now() - k_start;
}

}  // namespace captains::llog
