#pragma once

#include <cstdint>

struct pixel_t {
	uint8_t r, g, b, a;

	pixel_t half(void) const {
		return {
		    .r = (uint8_t)(r / 2),
		    .g = (uint8_t)(g / 2),
		    .b = (uint8_t)(b / 2),
		    .a = (uint8_t)(a / 2)};
	}

	pixel_t operator+(const pixel_t & rhs) const {
		return {
		    .r = (uint8_t)(r + rhs.r),
		    .g = (uint8_t)(g + rhs.g),
		    .b = (uint8_t)(b + rhs.b),
		    .a = (uint8_t)(a + rhs.a)};
	}

	pixel_t avg(const pixel_t & rhs) const { return half() + rhs.half(); }

	pixel_t dim(double factor) const {
		return {
		    .r = (uint8_t)((double)r * factor),
		    .g = (uint8_t)((double)g * factor),
		    .b = (uint8_t)((double)b * factor),
		    .a = a};
	}
};

static constexpr pixel_t k_white = {0xff, 0xff, 0xff, 0xff};
static constexpr pixel_t k_black = {0x00, 0x00, 0x00, 0xff};
static constexpr pixel_t k_red = {0xff, 0x00, 0x00, 0xff};
