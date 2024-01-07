#include "src/sink.hh"

#include "src/exception.hh"
#include "src/log.hh"
#include "src/pixel.hh"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/gstbus.h>
#include <gst/gstclock.h>
#include <gst/gstelement.h>
#include <gst/gstmemory.h>
#include <gst/gstmessage.h>
#include <gst/gstpipeline.h>

#include <atomic>
#include <vector>

static constexpr auto k_src_name = "m_src";

static std::atomic_flag k_gst_init = false;

void check_gst_init(void) {
	if (!k_gst_init.test_and_set()) {
		int     argc = 0;
		char ** argv = {nullptr};
		gst_init(&argc, &argv);
		log::debug("gstreamer initalized");
	}
}

std::string join(std::string_view sep, const std::vector<std::string_view> & items) {
	std::string out = "";

	bool flag = false;
	for (const auto & it : items) {
		if (flag) out += sep;
		flag = true;
		out += it;
	}
	return out;
}

Sink::Sink(Params params) :
    m_width{params.width}, m_height{params.height}, m_fps{params.fps} {
	GError * err = nullptr;

	check_gst_init();

	auto appsrc_str = fmt::format(
	    "appsrc name={} caps=video/x-raw,format={},width={},height={}",
	    k_src_name,
	    params.pixfmt,
	    m_width,
	    m_height);
	auto enc_str = fmt::format("x264enc bitrate={}", params.bitrate);
	auto filesink_str = fmt::format("filesink location={}", params.output);
	auto pipeline_str = join(
	    " ! ",
	    {appsrc_str, "videoconvert", enc_str, "h264parse", "mp4mux", filesink_str});
	log::info("pipeline str: {}", pipeline_str);

	m_pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
	if (err) throw Exception::format("gst_parse_launch err: {}", err->message);

	m_bus = gst_pipeline_get_bus(GST_PIPELINE_CAST(m_pipeline));
	m_src = gst_bin_get_by_name(GST_BIN_CAST(m_pipeline), k_src_name);

	// Start sink
	gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
}

void Sink::submit_frame(const pixel_t * frame, int pitch) {
	GstBuffer * buff = gst_buffer_new_allocate(nullptr, frame_size(), nullptr);
	if (buff == nullptr) throw Exception("allocation failed");

	auto ts = m_next_frame.count();
	m_next_frame += frame_dur();
	buff->pts = ts;
	buff->dts = ts;
	buff->duration = frame_dur().count();

	GstMapInfo map;
	if (!gst_buffer_map(buff, &map, GST_MAP_READWRITE))
		throw Exception("couldn't map buffer");

	cudaError_t err = cudaMemcpy2D(
	    map.data,
	    m_width * sizeof(pixel_t),
	    frame,
	    pitch,
	    m_width * sizeof(pixel_t),
	    m_height,
	    cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) throw Exception("cuda memcpy failed");

	gst_buffer_unmap(buff, &map);

	int ret;
	g_signal_emit_by_name(m_src, "push-buffer", buff, &ret);
	if (ret) log::warn("push-buffer returned {}", ret);
	gst_buffer_unref(buff);
}

void Sink::end(void) {
	log::trace("ending stream");
	gst_element_send_event(m_src, gst_event_new_eos());

	// Wait for eos signal
	GstMessage * msg =
	    gst_bus_timed_pop_filtered(m_bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);
	if (msg) {
		gst_message_unref(msg);
	} else {
		log::warn("got back nullptr for bus eos");
	}
}

uint64_t Sink::frame_size(void) const {
	return sizeof(pixel_t) * m_height * m_width;
}

Sink::Duration Sink::frame_dur(void) const { return m_fps.frame_dur(); }

Ratio Sink::fps(void) const { return m_fps; }

Sink::~Sink() {
	gst_element_set_state(m_pipeline, GST_STATE_NULL);

	gst_object_unref(m_src);
	gst_object_unref(m_bus);
	gst_object_unref(m_pipeline);
}
