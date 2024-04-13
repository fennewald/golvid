#pragma once

#include <chrono>

class Ratio {
public:
	constexpr Ratio(long numerator, long denominator = 1) :
	    m_numerator{numerator}, m_denominator{denominator} {}

	constexpr std::chrono::nanoseconds frame_dur() const {
		using namespace std::chrono_literals;
		return std::chrono::duration_cast<std::chrono::nanoseconds>(1s) *
		       m_denominator / m_numerator;
	}

private:
	long m_numerator;
	long m_denominator;
};
