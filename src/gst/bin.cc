#include "src/gst/bin.hh"

#include <gst/gstbin.h>

#include <optional>

namespace gst {

Bin::Bin(GstBin * bin) : Element{(GstElement *)bin} {}

Bin::operator GstBin *(void) { return GST_BIN_CAST(m_object); }

std::optional<Element> Bin::get_by_name(const std::string & name) {
	GstElement * ref = gst_bin_get_by_name(*this, name.c_str());
	if (ref) {
		return std::make_optional(ref);
	} else {
		return std::nullopt;
	}
}

}  // namespace gst
