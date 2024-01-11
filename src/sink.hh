#pragma once

#include "src/ratio.hh"
#include "src/sim.cuh"

#include <gst/gstbus.h>
#include <gst/gstelement.h>
#include <gst/gst.h>

#include <cstdint>
#include <chrono>
#include <string_view>

class Sink {
public:
	using Duration = std::chrono::nanoseconds;
	struct Params {
		uint64_t width;
		uint64_t height;
		uint64_t bitrate = 1000000;
		std::string_view pixfmt = "RGBA";
		std::string_view output = "out.mp4";
		Ratio fps = {30};
	};
	
	Sink(Params params);

	void submit_frame(const Pixel * frame, int pitch);

	void end(void);

	Ratio fps(void) const;

	~Sink(void);

private:
	uint64_t frame_size(void) const;

	Duration frame_dur(void) const;

	Duration m_next_frame;

	uint64_t m_width;
	uint64_t m_height;

	Ratio m_fps;

	GstElement * m_pipeline;
	GstBus * m_bus;
	GstElement * m_src;
};
