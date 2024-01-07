#include "src/kernel.cuh"
#include "src/log.hh"
#include "src/pixel.hh"
#include "src/ratio.hh"
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
	using namespace std::chrono_literals;

	auto params = Sink::Params{
	    .width = 1080,
	    .height = 1920,
	};

	auto sink = Sink(params);
	auto res = res_t{
	    .width = (unsigned int)params.width,
	    .height = (unsigned int)params.height};
	auto duration = 30s;
	auto n_frames = duration / sink.fps().frame_dur();

	// Cuda types
	float       time;
	cudaEvent_t start, stop;
	CU_CALL(cudaEventCreate(&start));
	CU_CALL(cudaEventCreate(&stop));

	pixel_t * pixels;
	size_t    pix_pitch;
	CU_CALL(cudaMallocPitch(
	    &pixels, &pix_pitch, params.width * sizeof(pixel_t), params.height));
	log::info("created frame with pitch {}", pix_pitch);

	// Setup world states
	uint8_t *w_prev, *w_next;
	size_t   w_pitch_0, w_pitch_1, w_pitch;
	CU_CALL(cudaMallocPitch(&w_prev, &w_pitch_0, params.width, params.height));
	CU_CALL(cudaMallocPitch(&w_next, &w_pitch_1, params.width, params.height));
	if (w_pitch_0 != w_pitch_1) {
		fmt::println("mismatch");
		exit(EXIT_FAILURE);
	}
	w_pitch = w_pitch_0;
	log::info("allocated worlds");

	CU_CALL(cudaEventRecord(start, 0));
	initalize(w_prev, w_next, res, w_pitch, pixels, pix_pitch);
	CU_CALL(cudaGetLastError());
	CU_CALL(cudaEventRecord(stop, 0));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));

	log::info(
	    "initalizing done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));

	// Warm simulation
	for (long i = 0; i < 100; ++i) {
		step(&w_prev, &w_next, res, w_pitch, pixels, pix_pitch);
	}

	for (long i = 0; i < n_frames; ++i) {
		step(&w_prev, &w_next, res, w_pitch, pixels, pix_pitch);
		sink.submit_frame(pixels, pix_pitch);
	}
	sink.end();
}

