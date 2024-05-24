#pragma once

#include "src/exception.hh"
#include "src/util.hh"

#include <cuda_runtime.h>

#include <cstdint>
#include <string_view>

class Pixel {
public:
	static constexpr __host__ Pixel from_hex(std::string_view str) {
		auto it = str.cbegin();
		auto end = str.cend();
		if (*it == '#') it++;
		uint8_t channels[3] = {0, 0, 0};

		for (int i = 0; i < 3; ++i) {
			if (it == end)
				throw Exception::format("Invalid hex string '{}'", str);
			channels[i] = util::octect_from_char(*it++) << 4;
			if (it == end)
				throw Exception::format("Invalid hex string '{}'", str);
			channels[i] |= util::octect_from_char(*it++);
		}
		if (it != end) throw Exception::format("Invalid hex string '{}'", str);

		return Pixel{channels[0], channels[1], channels[2]};
	}

	static __device__ constexpr Pixel from_rgb(unsigned int raw) {
		return Pixel{
		    static_cast<uint8_t>((raw >> 16) & 0xff),
		    static_cast<uint8_t>((raw >> 8) & 0xff),
		    static_cast<uint8_t>(raw & 0xff)};
	}

	static __device__ Pixel ramp(Pixel l, Pixel r, float t) {
		return Pixel{
		    static_cast<uint8_t>(((r.r() - l.r()) * t) + l.r()),
		    static_cast<uint8_t>(((r.g() - l.g()) * t) + l.g()),
		    static_cast<uint8_t>(((r.b() - l.b()) * t) + l.b()),
		    static_cast<uint8_t>(((r.a() - l.a()) * t) + l.a()),
		};
	}

	__device__ __host__ constexpr Pixel(void) {}
	__device__ __host__ constexpr Pixel(uint8_t r, uint8_t g, uint8_t b) :
	    m_raw{r, g, b, 0xff} {}
	__device__ __host__ constexpr Pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
	    m_raw{r, g, b, a} {}

	__device__ __host__ inline uint8_t r(void) const { return m_raw.x; }
	__device__ __host__ inline uint8_t g(void) const { return m_raw.y; }
	__device__ __host__ inline uint8_t b(void) const { return m_raw.z; }
	__device__ __host__ inline uint8_t a(void) const { return m_raw.w; }

	__device__ __host__ inline void set_r(uint8_t n) { m_raw.x = n; }
	__device__ __host__ inline void set_g(uint8_t n) { m_raw.y = n; }
	__device__ __host__ inline void set_b(uint8_t n) { m_raw.z = n; }
	__device__ __host__ inline void set_a(uint8_t n) { m_raw.w = n; }

	__device__ __host__ inline void set_rgb(uint8_t r, uint8_t g, uint8_t b) {
		set_r(r);
		set_g(g);
		set_b(b);
	}

private:
	uchar4 m_raw = {0, 0, 0, 0xff};
};

namespace color {

inline constexpr Pixel white = Pixel(0xff, 0xff, 0xff);
inline constexpr Pixel black = Pixel::from_hex("#000000");
inline constexpr Pixel red = Pixel::from_hex("#ff0000");
inline constexpr Pixel green = Pixel::from_hex("#00ff00");
inline constexpr Pixel blue = Pixel::from_hex("#0000ff");

}  // namespace color
