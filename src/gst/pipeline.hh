#pragma once

#include "src/gst/bin.hh"
#include "src/gst/bus.hh"

#include "gst/gstpipeline.h"

#include <string>

namespace gst {

class Pipeline : public Bin {
public:
	static Pipeline parse_launch(const std::string & launch_str);

	Pipeline(GstPipeline * pipeline);

	operator GstPipeline *(void);

	Bus bus(void);

	void start(void);

	void wait_eos(void);
};

}  // namespace gst
