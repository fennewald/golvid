#include "src/gst/bus.hh"

#include <gst/gstclock.h>
#include <gst/gstbus.h>

namespace gst {

Bus::Bus(GstBus * bus) : Object{(GstObject *)bus} {}

Bus::operator GstBus *(void) { return GST_BUS_CAST(m_object); }

GstMessage * Bus::pop_filtered(GstMessageType types, std::optional<std::chrono::nanoseconds> timeout) {
	unsigned long t = GST_CLOCK_TIME_NONE;
	if (timeout.has_value()) {
		t = timeout->count();
	}
	return gst_bus_timed_pop_filtered(*this, t, types);
}

}  // namespace gst
