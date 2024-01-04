#pragma once

#include "src/log.hh"

#include <gst/gstobject.h>

namespace gst {

// Base object of gstreamer stuff
class Object {
public:
	Object(GstObject * object) : m_object{object} {}
	Object(const Object & rhs) {
		log::info("reffing");
		m_object = rhs.m_object;
		gst_object_ref(m_object);
	}
	Object & operator=(const Object & rhs) = delete;
	~Object(void) { gst_object_unref(m_object); }

protected:
	GstObject * m_object;
};

}  // namespace gst
