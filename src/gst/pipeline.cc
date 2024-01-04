#include "src/gst/pipeline.hh"

#include <gst/gstmessage.h>
#include <gst/gstparse.h>

namespace gst {

Pipeline Pipeline::parse_launch(const std::string & launch_str) {
	GstElement * pipeline = gst_parse_launch(launch_str.c_str(), nullptr);
	return Pipeline{(GstPipeline *)pipeline};
}

Pipeline::Pipeline(GstPipeline * pipeline) : Bin{(GstBin *)pipeline} {}

Bus Pipeline::bus(void) {
	auto ref = gst_pipeline_get_bus(*this);
	return Bus{ref};
}

Pipeline::operator GstPipeline *(void) { return (GstPipeline *)m_object; }

void Pipeline::start(void) {
	auto b = bus();
	set_state(State::Playing);
	auto res = b.pop_filtered(GST_MESSAGE_STATE_CHANGED);
	log::info("got res: {}", fmt::ptr(res));
	gst_message_unref(res);
}

void Pipeline::wait_eos(void) {
	auto b = bus();
	auto res = b.pop_filtered(GST_MESSAGE_EOS);
	log::info("got res: {}", fmt::ptr(res));
	if (res) gst_message_unref(res);
}

}  // namespace gst
