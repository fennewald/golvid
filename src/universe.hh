#pragma once

#include "src/log.hh"
#include "src/pixel.hh"

#include <gst/gstbuffer.h>
#include <gst/gstmemory.h>

#include <cstdint>
#include <cstdlib>
#include <vector>


inline void notify_done(void * data) { log::info("freeing {}", data); }

class Universe {
public:
	constexpr Universe(uint64_t width, uint64_t height) :
	    m_width{width},
	    m_height{height},
		m_a(width * height),
		m_b(width * height),
	    m_pixels{width * height, pixel{0, 0, 0, 0xff}} {
		randomize();
	}

	void randomize(void) {
		for (long x = 0; x < (long)m_width; ++x) {
			for (long y = 0; y < (long)m_height; ++y) {
				set_current(x, y, std::rand() % 2 == 0);
			}
		}
	}

	void step(void) {
		m_current = !m_current;
		for (long x = 0; x < (long)m_width; ++x) {
			for (long y = 0; y < (long)m_height; ++y) {
				set_current(x, y, next_state(x, y));
			}
		}
		mix();
	}

	void mix(void) {
		for (uint64_t x = 0; x < m_width; ++x) {
			for (uint64_t y = 0; y < m_height; ++y) {
				auto idx = index(x, y);
				m_pixels.at(idx) = mix_pixel(m_pixels.at(idx), current().at(idx));
			}
		}
	}

	GstBuffer * to_buffer(void) {
		auto size = m_width * m_height * sizeof(pixel);
		return gst_buffer_new_wrapped_full(
		    GST_MEMORY_FLAG_READONLY,
		    m_pixels.data(),
		    size,
		    0,
		    size,
		    m_pixels.data(),
		    notify_done);
	}

private:
	bool next_state(long x, long y) {
		bool old_state = get_prev(x, y);
		auto neighbors = 0;
		// clang-format off
		if (get_prev(x-1, y-1)) neighbors++;
		if (get_prev( x,  y-1)) neighbors++;
		if (get_prev(x+1, y-1)) neighbors++;
		if (get_prev(x-1,  y )) neighbors++;
		if (get_prev(x+1,  y )) neighbors++;
		if (get_prev(x-1, y+1)) neighbors++;
		if (get_prev( x,  y+1)) neighbors++;
		if (get_prev(x+1, y+1)) neighbors++;
		// clang-format on
		if (old_state && neighbors == 2) return true;
		if (neighbors == 3) return true;
		if (neighbors == 6) return true;
		if (neighbors == 7) return true;
		return false;
	}

	static pixel mix_pixel(pixel old, bool current) {
		if (current) return k_white;
		else return old.dim(0.9);
	}

	constexpr void set_current(long x, long y, bool val) {
		auto idx = index(x, y);
		current().at(idx) = val;
	}

	constexpr bool get_prev(long x, long y) const {
		auto idx = index(x, y);
		return prev().at(idx);
	}

	constexpr uint64_t index(long x, long y) const {
		x %= m_width;
		y %= m_height;
		return x + (y * m_width);
	}

	const std::vector<bool> & prev(void) const {
		if (m_current) {
			return m_a;
		} else {
			return m_b;
		}
	}

	std::vector<bool> & current(void) {
		if (m_current) {
			return m_b;
		} else {
			return m_a;
		}
	}

	uint64_t m_width;
	uint64_t m_height;

	bool m_current = true;

	std::vector<bool>  m_a;
	std::vector<bool>  m_b;
	std::vector<pixel> m_pixels;
};
