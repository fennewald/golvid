#include "src/log/time.hh"

#include <atomic>
#include <chrono>

namespace captains::log {

using Clock = std::chrono::high_resolution_clock;

static std::atomic_flag k_initalized = false;
static Clock::time_point k_start;

static void init(void) { k_start = Clock::now(); }

Duration stamp(void) {
	if (!k_initalized.test_and_set()) init();

	return Clock::now() - k_start;
}

}  // namespace captains::log
