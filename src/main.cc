#include "fmt/core.h"
#include "src/agent_vec.cuh"
#include "src/log.hh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/sink.hh"

#include <cuda_runtime.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#include <cstdlib>
#include <utility>

#define CU_CALL(_expr)                         \
	do {                                       \
		cudaError_t _err = _expr;              \
		if (_err != cudaSuccess) {             \
			llog::error("got error {}", _err); \
			exit(EXIT_FAILURE);                \
		}                                      \
	} while (0)

int main() {
	llog::info("start");
	using namespace std::chrono_literals;

	auto sink = Sink(Sink::Params{
	    .width = params::width,
	    .height = params::height,
	    .fps = {30, 1},
	});
	auto duration = 30s;
	auto n_frames = duration / sink.fps().frame_dur();

	// Cuda types
	float       time;
	cudaEvent_t start, stop;
	CU_CALL(cudaEventCreate(&start));
	CU_CALL(cudaEventCreate(&stop));

	Pixel * pixels;
	size_t  pix_pitch;
	CU_CALL(cudaMallocPitch(
	    &pixels, &pix_pitch, params::width * sizeof(Pixel), params::height));
	llog::trace(
	    "allocated {}x{} frame (pitch: {}, ptr: {})",
	    params::width,
	    params::height,
	    pix_pitch,
	    fmt::ptr(pixels));

	auto prev_m = Medium::alloc();
	auto next_m = Medium::alloc();

	auto agents = AgentVec::alloc();

	// Initalize data layers
	CU_CALL(cudaEventRecord(start));
	prev_m.init();
	next_m.init();
	CU_CALL(cudaEventRecord(stop));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));
	llog::info(
	    "initalizing data layer done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));

	// Initalize agent arrays
	CU_CALL(cudaEventRecord(start));
	agents.init();
	CU_CALL(cudaEventRecord(stop));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));
	llog::info(
	    "initalizing agents done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));


	for (long i = 0; i < n_frames; ++i) {
		fmt::print("{} / {}\r", i, n_frames);
		agents.step(prev_m);
		agents.deposit(prev_m);
		prev_m.render(pixels, pix_pitch);
		prev_m.step(next_m);
		sink.submit_frame(pixels, pix_pitch);
		std::swap(prev_m, next_m);
	}
	sink.end();
}
