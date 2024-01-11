#pragma once

#include <cuda_runtime.h>

template<typename T>
struct Vec2 {
	T x, y;

	__host__ __device__ Vec2<T> operator+(Vec2<T> rhs) const {
		return {x + rhs.x, y + rhs.y};
	}

	__host__ __device__ Vec2<T> operator*(T rhs) const {
		return {x * rhs, y * rhs};
	}

	__host__ __device__ void operator+=(Vec2<T> rhs) {
		x += rhs.x;
		y += rhs.y;
	}
};

using IVec2 = Vec2<int>;
using UVec2 = Vec2<unsigned int>;
using FVec2 = Vec2<float>;
