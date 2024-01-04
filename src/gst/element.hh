#pragma once

#include "src/gst/object.hh"
#include "src/gst/state.hh"
#include "src/gst/state_change_return.hh"

#include <gst/gstelement.h>

namespace gst {

class Element : public Object {
public:
	Element(GstElement * element);

	operator GstElement *(void);

	bool send_eos(void);

	State state(void);

	State pending_state(void);

	StateChangeReturn set_state(State desired);
};

}  // namespace gst
