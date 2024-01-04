#pragma once

#include <fmt/format.h>
#include <gst/gstelement.h>

namespace gst {

enum class StateChangeReturn {
	Failure = GST_STATE_CHANGE_FAILURE,
	Success = GST_STATE_CHANGE_SUCCESS,
	Async = GST_STATE_CHANGE_ASYNC,
	NoPreroll = GST_STATE_CHANGE_NO_PREROLL,
};

}

template<>
struct fmt::formatter<gst::StateChangeReturn> : fmt::formatter<std::string_view> {
	auto format(gst::StateChangeReturn state, format_context & ctx) const
	    -> format_context::iterator {
		std::string_view name = "ERROR_UNKNOWN";

		switch (state) {
			using enum gst::StateChangeReturn;
		case Failure:   name = "Failure";   break;
		case Success:   name = "Success";   break;
		case Async:     name = "Async";     break;
		case NoPreroll: name = "NoPreroll"; break;
		}

		return fmt::formatter<std::string_view>::format(name, ctx);
	}
};
