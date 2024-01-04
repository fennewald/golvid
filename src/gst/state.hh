#pragma once

#include <fmt/format.h>
#include <gst/gstelement.h>

namespace gst {

enum class State {
	VoidPending = GST_STATE_VOID_PENDING,
	Null = GST_STATE_NULL,
	Ready = GST_STATE_READY,
	Paused = GST_STATE_PAUSED,
	Playing = GST_STATE_PLAYING,
};

}

template<>
struct fmt::formatter<gst::State> : fmt::formatter<std::string_view> {
	auto format(gst::State state, format_context & ctx) const
	    -> format_context::iterator {
		std::string_view name = "ERROR_UNKNOWN";

		switch (state) {
			using enum gst::State;
		case VoidPending: name = "VoidPending"; break;
		case Null:        name = "Null";        break;
		case Ready:       name = "Ready";       break;
		case Paused:      name = "Paused";      break;
		case Playing:     name = "Playing";     break;
		}

		return fmt::formatter<std::string_view>::format(name, ctx);
	}
};
