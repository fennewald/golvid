#pragma once

#include "src/params.hh"
#include "src/pixel.cuh"

#include <cuda_runtime.h>


class Cell {
public:
	static __device__ __host__ constexpr Cell empty() { return Cell(-1); }

	__device__ __host__ constexpr Cell() {}
	__device__ __host__ constexpr Cell(int a) : m_a{a} {}

	__device__ __host__ inline constexpr void clear() { m_a = 0; }

	__device__ inline Pixel color() const {
		// Color ramp
		static constexpr Pixel palette[] = {
		    Pixel::from_rgb(0x000000),
		    Pixel::from_rgb(0x471a61),
		    Pixel::from_rgb(0xbe3e4b),
		    Pixel::from_rgb(0xf36c15),
		    Pixel::from_rgb(0xffaa1f),
		    Pixel::from_rgb(0xffedaf),
		    Pixel::from_rgb(0xffffff)
		};
		static constexpr size_t ramp_len = sizeof(palette) / sizeof(palette[0]);

		float t = (float)m_a / params::max_brightness_cell;
		if (t >= 1.0) return palette[ramp_len - 1];
		t *= ramp_len - 1;
		int l = static_cast<int>(t);
		int r = l + 1;
		t -= (float)l;
		return Pixel::ramp(palette[l], palette[r], t);

		// long c = static_cast<long>(t * 255.0);
		// if (c > 0xff) c = 0xff;
		// auto p = Pixel{};
		// p.set_r(c);
		// p.set_g(c);
		// p.set_b(c);
		// return p;
	}

	__device__ __host__ inline constexpr void operator+=(Cell rhs) {
		m_a += rhs.m_a;
	}

	__device__ __host__ inline constexpr Cell operator/(int rhs) {
		return Cell{m_a / rhs};
	}

	__device__ __host__ inline constexpr Cell operator-(int rhs) {
		return Cell{m_a - rhs};
	}

	__device__ inline constexpr bool operator>(Cell rhs) const {
		return m_a > rhs.m_a;
	}

	// Atomically deposit into this cell
	__device__ inline void deposit() {
		// atomicAdd(&m_a, params::deposit_amount);
		m_a += params::deposit_amount;
	}

	__device__ inline void decay() { m_a *= params::decay_factor; }

private:
	int m_a = 0;
};
