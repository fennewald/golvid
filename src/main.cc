#include "src/gst/init.hh"
#include "src/gst/pipeline.hh"
#include "src/gst/state.hh"
#include "src/log.hh"
#include "src/ratio.hh"
#include "src/universe.hh"

#include <fmt/format.h>

int main(void) {
	using namespace std::chrono_literals;

	gst::init();

	std::string_view pixfmt = "RGBA";
	constexpr int    width = 1000;
	constexpr int    height = 1000;
	constexpr auto   duration = 30s;
	constexpr auto   fps = Ratio{30, 1};
	constexpr auto   n_frames = duration / fps.frame_duration();

	log::info("{} frames", n_frames);

	auto universe = Universe(width, height);

	auto launch_str = fmt::format(
	    "appsrc name=src caps=video/x-raw,format={},width={},height={} ! videoconvert ! x264enc bitrate=1000000 ! h264parse ! mp4mux ! filesink location=out.mp4",
	    pixfmt,
	    width,
	    height);

	log::info("launch_str: {}", launch_str);

	auto pipeline = gst::Pipeline::parse_launch(launch_str);
	auto src = pipeline.get_by_name("src").value();
	pipeline.start();

	for (size_t i = 0; i < n_frames; ++i) {
		log::info("{}%", (double)i * 100.0 / (double)n_frames);
		int ret;
		universe.step();
		auto buff = universe.to_buffer();
		buff->dts = i * fps.frame_duration().count();
		buff->pts = i * fps.frame_duration().count();
		g_signal_emit_by_name(src, "push-buffer", buff, &ret);
		gst_buffer_unref(buff);
	}
	src.send_eos();
	pipeline.wait_eos();
	log::info("got eos");
	pipeline.set_state(gst::State::Null);
}
