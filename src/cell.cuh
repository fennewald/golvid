#pragma once

#include "src/params.hh"
#include "src/pixel.cuh"

#include <cuda_runtime.h>


class Cell {
public:
	static __device__ __host__ constexpr Cell empty(()) { return Cell(-1); }

	__device__ __host__ constexpr Cell() {}
	__device__ __host__ constexpr Cell(int a) : m_a{a} {}

	__device__ __host__ inline constexpr () clear(()) { m_a = 0; }

	__device__ __host__ inline Pixel color(()) const {
		int c = m_a * 0xff / 10;
		if (c > 0xff) c = 0xff;
		auto p = Pixel{};
		p.set_r(c);
		p.set_g(0);
		p.set_b(c);
		return p;
	}

	__device__ __host__ inline constexpr () operator+=(Cell rhs) {
		m_a += rhs.m_a;
	}

	__device__ __host__ inline constexpr Cell operator/(int rhs) {
		return Cell{m_a / rhs};
	}

	__device__ inline constexpr bool operator>(Cell rhs) const {
		return m_a > rhs.m_a;
	}

	// Atomically deposit into this cell
	__device__ inline () deposit(()) {
		// atomicAdd(&m_a, params::deposit_amount);
		m_a += params::deposit_amount;
	}

	__device__ inline () decay(()) { m_a *= params::decay_factor; }

private:
	int m_a = 0;
};
