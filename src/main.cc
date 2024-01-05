#include "gst/gstmemory.h"
#include "src/gst/init.hh"
#include "src/gst/pipeline.hh"
#include "src/gst/state.hh"
#include "src/kernel.cuh"
#include "src/log.hh"
#include "src/pixel.hh"
#include "src/ratio.hh"

#include <cstdlib>
#include <cuda_runtime.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <gst/gstbuffer.h>

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

	gst::init();

	std::string_view pixfmt = "RGBA";
	std::string_view output_file = "out.mp4";
	constexpr int    width = 1080;
	constexpr int    height = 1920;
	constexpr auto   duration = 30s;
	constexpr auto   fps = Ratio{30, 1};
	constexpr auto   n_frames = duration / fps.frame_duration();
	constexpr long   frame_size = width * height * sizeof(pixel_t);

	// Cuda types
	float       time;
	cudaEvent_t start, stop;
	CU_CALL(cudaEventCreate(&start));
	CU_CALL(cudaEventCreate(&stop));

	// Setup frame
	pixel_t * pixels;
	size_t    pix_pitch;
	CU_CALL(cudaMallocPitch(&pixels, &pix_pitch, width * sizeof(pixel_t), height));
	log::info("created frame with pitch {}", pix_pitch);

	// Setup world states
	uint8_t *world_0, *world_1;
	size_t   w_pitch_0, w_pitch_1, w_pitch;
	CU_CALL(cudaMallocPitch(&world_0, &w_pitch_0, width, height));
	CU_CALL(cudaMallocPitch(&world_1, &w_pitch_1, width, height));
	if (w_pitch_0 != w_pitch_1) {
		fmt::println("mismatch");
		exit(EXIT_FAILURE);
	}
	w_pitch = w_pitch_0;
	log::info("allocated worlds");

	CU_CALL(cudaEventRecord(start, 0));
	initalize(world_0, world_1, {width, height}, w_pitch, pixels, pix_pitch);
	CU_CALL(cudaGetLastError());
	CU_CALL(cudaEventRecord(stop, 0));
	CU_CALL(cudaEventSynchronize(stop));
	CU_CALL(cudaEventElapsedTime(&time, start, stop));

	uint8_t * w_prev = world_0;
	uint8_t * w_next = world_1;

	log::info(
	    "initalizing done, took {}",
	    std::chrono::duration<float, std::ratio<1, 1000>>(time));

	auto launch_str = fmt::format(
	    "appsrc name=src caps=video/x-raw,format={},width={},height={} ! videoconvert ! x264enc bitrate=1000000 ! h264parse ! mp4mux ! filesink location={}",
	    pixfmt,
	    width,
	    height,
	    output_file);

	log::info("launch_str: {}", launch_str);

	auto pipeline = gst::Pipeline::parse_launch(launch_str);
	log::info("created pipeline");

	auto src = pipeline.get_by_name("src").value();
	pipeline.start();

	for (size_t i = 0; i < n_frames; ++i) {
		int ret;
		GstBuffer * buff = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
		if (buff == nullptr) {
			log::error("buffer_new returned nullptr");
			exit(EXIT_FAILURE);
		}
		buff->dts = i * fps.frame_duration().count();
		buff->pts = i * fps.frame_duration().count();

		GstMapInfo map_info;
		if (!gst_buffer_map(
		        buff, &map_info, (GstMapFlags)(GST_MAP_READ | GST_MAP_WRITE))) {
			log::error("couldn't map :(");
			exit(EXIT_FAILURE);
		}

		uint8_t * tmp = w_prev;
		w_prev = w_next;
		w_next = tmp;

		step(w_prev, w_next, {width, height}, w_pitch, pixels, pix_pitch);

		CU_CALL(cudaMemcpy2D(
		    map_info.data,
		    width * sizeof(pixel_t),
		    pixels,
		    pix_pitch,
		    width * sizeof(pixel_t),
		    height,
		    cudaMemcpyDeviceToHost));

		/*
		pixel_t * tmp = (pixel_t *)map_info.data;
		int       y = height - 2;
		for (int x = 0; x < width; x += 2) { tmp[(y * width) + x] = k_red; }
		int x = width / 2;
		for (y = 0; y < height; ++y) {
			tmp[(y * width) + x - 1] = k_red;
			tmp[(y * width) + x] = k_white;
		}
		tmp[((height - 1) * width) + width - 1] = k_red;
		*/

		gst_buffer_unmap(buff, &map_info);

		g_signal_emit_by_name(src, "push-buffer", buff, &ret);
		// log::debug("ret({}): {}%", ret, (double)i * 100.0 /
		// (double)n_frames);
		gst_buffer_unref(buff);
	}

	log::info("exiting");
	src.send_eos();
	pipeline.wait_eos();
	pipeline.set_state(gst::State::Null);

	// Free up cuda resources
	CU_CALL(cudaFree(pixels));
	CU_CALL(cudaFree(world_0));
	CU_CALL(cudaFree(world_1));
}
