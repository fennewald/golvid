#pragma once

#include "src/gst/element.hh"

#include <gst/gstbin.h>

#include <string>
#include <optional>

namespace gst {

class Bin : public Element {
public:
	Bin(GstBin * bin);

	operator GstBin *(void);

	std::optional<Element> get_by_name(const std::string & name);
};

}
