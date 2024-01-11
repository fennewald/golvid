#pragma once

#include "src/exception.hh"

#include <cuda_runtime.h>

#include <cstdint>
#include <string_view>

class Pixel {
private:
	static constexpr char octect_from_char(char c) {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		throw Exception::format("unexpected character: {}", c);
	}

public:
	static constexpr Pixel from_hex(std::string_view str) {
		auto it = str.cbegin();
		auto end = str.cend();
		if (*it == '#') it++;
		uint8_t channels[3] = {0, 0, 0};

		for (int i = 0; i < 3; ++i) {
			if (it == end)
				throw Exception::format("Invalid hex string '{}'", str);
			channels[i] = octect_from_char(*it++) << 4;
			if (it == end)
				throw Exception::format("Invalid hex string '{}'", str);
			channels[i] |= octect_from_char(*it++);
		}
		if (it != end) throw Exception::format("Invalid hex string '{}'", str);

		return Pixel{channels[0], channels[1], channels[2]};
	}

	__device__ __host__ Pixel(void) = default;
	__device__ __host__ Pixel(uint8_t r, uint8_t g, uint8_t b) :
	    m_raw{r, g, b, 0xff} {}
	__device__ __host__ Pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
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
