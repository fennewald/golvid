#pragma once

#include <fmt/core.h>

#include <exception>
#include <string>

// Base class for exceptions that just want to pass a single string
class Exception : public std::exception {
public:
	template<typename... T>
	static Exception format(fmt::format_string<T...> format, T &&... ts) {
		// TODO: find way to avoid vformat?
		return Exception(fmt::vformat(format, fmt::make_format_args(ts...)));
	}

	static Exception todo(void) { return Exception("Unimplemented"); }

	Exception() = default;

	Exception(std::string msg) : m_msg{msg} {}

	virtual const char * what(void) const noexcept { return m_msg.c_str(); }

protected:
	std::string m_msg;
};
