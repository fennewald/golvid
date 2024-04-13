#pragma once

#include <fmt/format.h>

namespace captains::llog {

enum class Level {
	Trace = 1,
	Debug = 5,
	Info = 9,
	Warn = 13,
	Error = 17,
	Fatal = 21,
};

// Converts an integer into a level, by ratcheting at each log level
constexpr Level level_from_int(int n) {
	if (n < static_cast<int>(Level::Debug)) return Level::Trace;
	if (n < static_cast<int>(Level::Info)) return Level::Debug;
	if (n < static_cast<int>(Level::Warn)) return Level::Info;
	if (n < static_cast<int>(Level::Error)) return Level::Warn;
	if (n < static_cast<int>(Level::Fatal)) return Level::Error;
	return Level::Fatal;
}

}  // namespace captains::llog

template<>
struct fmt::formatter<captains::llog::Level> {
	bool color = false;

	constexpr auto parse(format_parse_context & ctx)
	    -> format_parse_context::iterator {
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && *it == 'c') {
			it++;
			color = true;
		}
		if (it != end && *it != '}') throw format_error("invalid format");
		return it;
	}

	auto format(const captains::llog::Level & level, format_context & ctx) const
	    -> format_context::iterator {
		std::string_view name = "NONE";
		using namespace captains::llog;

		if (color) {
			// Raw ansi escape sequence
			// caveat lector
			switch (level) {
			case Level::Fatal: name = "\x1b[1;31mfatal\x1b[0m"; break;
			case Level::Error: name = "\x1b[31merror\x1b[0m"; break;
			case Level::Warn: name = "\x1b[33mwarn\x1b[0m"; break;
			case Level::Info: name = "\x1b[32minfo\x1b[0m"; break;
			case Level::Debug: name = "\x1b[34mdebug\x1b[0m"; break;
			case Level::Trace: name = "\x1b[34mtrace\x1b[0m"; break;
			}
		} else {
			switch (level) {
			case Level::Fatal: name = "fatal"; break;
			case Level::Error: name = "error"; break;
			case Level::Warn: name = "warn"; break;
			case Level::Info: name = "info"; break;
			case Level::Debug: name = "debug"; break;
			case Level::Trace: name = "trace"; break;
			}
		}

		return fmt::formatter<std::string_view>().format(name, ctx);
	}
};
