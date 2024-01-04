#pragma once

#include <cstdint>

struct pixel {
	uint8_t r, g, b, a;

	pixel half(void) const {
		return {
		    .r = (uint8_t)(r / 2),
		    .g = (uint8_t)(g / 2),
		    .b = (uint8_t)(b / 2),
		    .a = (uint8_t)(a / 2)};
	}

	pixel operator+(const pixel & rhs) const {
		return {
		    .r = (uint8_t)(r + rhs.r),
		    .g = (uint8_t)(g + rhs.g),
		    .b = (uint8_t)(b + rhs.b),
		    .a = (uint8_t)(a + rhs.a)};
	}

	pixel avg(const pixel & rhs) const { return half() + rhs.half(); }

	pixel dim(double factor) const {
		return {
		    .r = (uint8_t)((double)r * factor),
		    .g = (uint8_t)((double)g * factor),
		    .b = (uint8_t)((double)b * factor),
		    .a = a};
	}
};

static constexpr pixel k_white = { 0xff, 0xff, 0xff, 0xff };
static constexpr pixel k_black = { 0x00, 0x00, 0x00, 0xff };
