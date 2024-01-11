#pragma once

#include "src/exception.hh"

namespace util {

inline constexpr double pi = 3.14159265358979323846;

constexpr char octect_from_char(char c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	throw Exception::format("unexpected character: {}", c);
}

template<typename N, typename D>
constexpr std::common_type_t<N, D> ceil_div(N num, D denom) {
	return (num + denom - 1) / denom;
}

template<typename T>
constexpr T deg_to_rad(T deg) {
	return deg * 3.14159265358979323846 / 180.0;
}

}  // namespace util
