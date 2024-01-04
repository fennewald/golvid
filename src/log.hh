#pragma once

#include "src/log/level.hh"

#include <fmt/format.h>

namespace captains::log {

template<typename... T>
void handle(Level lvl, fmt::format_string<T...> format, T &&... args) {
	fmt::print("[{:c}] ", lvl);
	fmt::println(format, args...);
}

#define DEF_LEVEL(name, enm)                                   \
	template<typename... T>                                    \
	void name(fmt::format_string<T...> format, T &&... args) { \
		fmt::print("[{:c}] ", enm);                            \
		fmt::vprint(format, fmt::make_format_args(args...));   \
		fmt::println("");                                      \
	}

DEF_LEVEL(trace, Level::Trace)
DEF_LEVEL(debug, Level::Debug)
DEF_LEVEL(info, Level::Info)
DEF_LEVEL(warn, Level::Warn)
DEF_LEVEL(error, Level::Error)
DEF_LEVEL(fatal, Level::Fatal)

#undef DEF_LEVEL


}  // namespace captains::log
using namespace captains;
