#include "src/gst/element.hh"

#include "src/exception.hh"
#include "src/gst/state_change_return.hh"

#include <gst/gstclock.h>
#include <gst/gstelement.h>

namespace gst {

Element::Element(GstElement * element) : Object{(GstObject *)element} {}

Element::operator GstElement *(void) { return (GstElement *)m_object; }

bool Element::send_eos(void) {
	return gst_element_send_event(*this, gst_event_new_eos());
}

State Element::state(void) {
	GstState state, pending;
	gst_element_get_state(*this, &state, &pending, GST_CLOCK_TIME_NONE);
	return static_cast<State>(state);
}

State Element::pending_state(void) {
	GstState state, pending;
	gst_element_get_state(*this, &state, &pending, GST_CLOCK_TIME_NONE);
	return static_cast<State>(pending);
}

StateChangeReturn Element::set_state(State desired) {
	GstStateChangeReturn raw_ret =
	    gst_element_set_state(*this, static_cast<GstState>(desired));
	auto ret = static_cast<StateChangeReturn>(raw_ret);
	if (ret == StateChangeReturn::Failure)
		throw Exception("failed to change element state");
	return ret;
}

}  // namespace gst
