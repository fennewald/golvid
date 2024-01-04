#pragma once

#include "src/gst/object.hh"

#include <chrono>
#include <gst/gstbus.h>

#include <optional>

namespace gst {

class Bus : public Object {
public:
	Bus(GstBus * bus);

	operator GstBus *();

	GstMessage * pop_filtered(
	    GstMessageType                          types,
	    std::optional<std::chrono::nanoseconds> timeout = std::nullopt);
};

}  // namespace gst
