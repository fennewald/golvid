#pragma once

#include "src/pixel.cuh"

#include <cuda_runtime.h>


class Cell {
public:
	static __device__ __host__ constexpr Cell empty(void) { return Cell(); }

	__device__ __host__ constexpr Cell(void) = default;
	__device__ __host__ constexpr Cell(int a) : m_a{a} {}

	__device__ __host__ inline constexpr void clear(void) { m_a = 0; }

	__device__ __host__ inline Pixel color(void) const {
		if (m_a > 0xff * 3) return color::white;
		Pixel p;
		p.set_r(m_a % 0xff);
		p.set_g((m_a - 0xff) % 0xff);
		p.set_g((m_a - (2 * 0xff)) % 0xff);
		return p;
	}

	__device__ __host__ inline constexpr void operator+=(Cell rhs) {
		m_a += rhs.m_a;
	}

	__device__ __host__ inline constexpr Cell operator/(int rhs) {
		return Cell{m_a / rhs};
	}

	// Atomically deposit into this cell
	__device__ inline void deposit(void) {
		atomicAdd(&m_a, 1);
	}

	int x = 2;

private:
	int m_a = 0;
};
