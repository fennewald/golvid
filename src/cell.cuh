#pragma once

#include "src/pixel.cuh"

#include <cuda_runtime.h>


class Cell {
public:
	__device__ constexpr Cell(void) = default;

	__device__ Pixel color(void) const {
		
	}

private:
	int m_x = 0;
};
