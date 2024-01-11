#include "src/exception.hh"
#include "src/log.hh"
#include "src/sim.cuh"
#include "src/params.hh"
#include "src/sink.hh"

#include <cstdlib>
#include <cuda_runtime.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#define CU_CALL(_expr)                             \
	do {                                           \
		cudaError_t _err = _expr;                  \
		if (_err != cudaSuccess) {                 \
			log::error("got error {}", (int)_err); \
			exit(EXIT_FAILURE);                    \
		}                                          \
	} while (0)

int main(void) {
	log::info("start");
	using namespace std::chrono_literals;

	auto sink = Sink(Sink::Params{
	    .width = params::width,
	    .height = params::height,
	    .fps = {1, 1},
	});
	auto duration = 60s;
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
	log::info(
	    "allocated {}x{} frame (pitch: {}, ptr: {})",
	    params::width,
	    params::height,
	    pix_pitch,
	    fmt::ptr(pixels));

	// Setup data layers
	Cell * cells_prev, *cells_next;
	size_t p0, p1;
	CU_CALL(cudaMallocPitch(
	    &cells_prev, &p0, params::width * sizeof(Cell), params::height));
	CU_CALL(cudaMallocPitch(
	    &cells_next, &p1, params::width * sizeof(Cell), params::height));
	if (p0 != p1) throw Exception::format("pitch mismatch: {} != {}", p0, p1);
	size_t cell_pitch = p0;
	log::info(
	    "allocated {}x{} data layer (pitch: {}, ptr: {}, {})",
	    params::width,
	    params::height,
	    cell_pitch,
	    fmt::ptr(cells_prev),
	    fmt::ptr(cells_next));

	// Setup agent layers
	float *agent_x, *agent_y, *agent_dir;
	CU_CALL(cudaMalloc(&agent_x, params::n_agents * sizeof(float)));
	log::info("allocated {} agent xs: {}", params::n_agents, fmt::ptr(agent_x));
	CU_CALL(cudaMalloc(&agent_y, params::n_agents * sizeof(float)));
	log::info("allocated {} agent ys: {}", params::n_agents, fmt::ptr(agent_y));
	CU_CALL(cudaMalloc(&agent_dir, params::n_agents * sizeof(float)));
	log::info(
	    "allocated {} agent dirs: {}", params::n_agents, fmt::ptr(agent_dir));

	// Initalize data layers
	CU_CALL(cudaEventRecord(start));
	initialize::cells(cells_prev, cell_pitch);
	initialize::cells(cells_next, cell_pitch);
	CU_CALL(cudaGetLastError());
	CU_CALL(cudaEventRecord(stop));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));
	log::info(
	    "initalizing data layer done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));

	// Initalize agent arrays
	CU_CALL(cudaEventRecord(start));
	initialize::agents(agent_x, agent_y, agent_dir);
	CU_CALL(cudaGetLastError());
	CU_CALL(cudaEventRecord(stop));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));
	log::info(
	    "initalizing agents done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));


	for (long i = 0; i < n_frames; ++i) {
		step(
		    &cells_prev,
		    &cells_next,
		    cell_pitch,
		    agent_x,
		    agent_y,
		    agent_dir,
		    pixels,
		    pix_pitch);
		sink.submit_frame(pixels, pix_pitch);
	}
	sink.end();
}
