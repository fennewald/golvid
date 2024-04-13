#include "src/exception.hh"
#include "src/medium.cuh"

#include "src/cuda_util.cuh"
#include "src/params.hh"
#include "src/pixel.cuh"

__device__ Cell Medium::at_f(float2 idx) {
	return at(int2{__float2int_rd(idx.x), __float2int_rd(idx.y)});
}

__device__ Cell Medium::avg_at(int2 idx) {
	auto res = Cell{};

	static constexpr int kernel_dim = 3;
	static constexpr int n_cells = kernel_dim * kernel_dim;
	static constexpr int max_val = (kernel_dim - 1) / 2;
	static constexpr int min_val = max_val * -1;
	static_assert(kernel_dim % 2 == 1, "kernel dim must be odd");

#pragma unroll
	for (int dy = min_val; dy <= max_val; ++dy) {
		int y = idx.y + dy;
#pragma unroll
		for (int dx = min_val; dx <= max_val; ++dx) {
			int x = idx.x + dx;
			res += at({x, y});
		}
	}
	return res / n_cells;
}


__global__ void k_init(Medium m) {
	auto idx = IDX_2;
	if (Medium::contains(idx)) m.get_raw(idx)->clear();
}

__host__ void Medium::init(void) {
	cudaError_t res =
	    cudaMemset2D(m_cells, m_pitch, 0, params::width, params::height);
	if (res != cudaSuccess)
		throw Exception::format("Failed to clear Medium, {}", res);

	k_init<<<params::cells_grid_dim, params::cells_block_dim>>>(*this);
	res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to initalize Medium, {}", res);
}


__global__ void k_step(Medium prev, Medium next) {
	auto idx = IDX_2;
	if (!Medium::contains(idx)) return;

	auto cell = prev.avg_at(idx);
	auto out = next.get_raw(idx);
	cell.decay();
	*out = cell;
}

__host__ void Medium::step(Medium into) const {
	k_step<<<params::cells_grid_dim, params::cells_block_dim>>>(*this, into);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to step Medium, {}", res);
}


__global__ void k_render(Medium m, Pixel * pixels, int pitch) {
	auto idx = IDX_2;
	if (!Medium::contains(idx)) return;

	Pixel * out = cu_util::pitch_ptr_raw(pixels, idx, pitch);
	*out = m.get_raw(idx)->color();

	if constexpr (params::hint_grid_dim != 0) {
		if ((idx.x % params::hint_grid_dim == params::hint_grid_dim - 1) ||
		    (idx.y % params::hint_grid_dim == params::hint_grid_dim - 1)) {
			*out = color::white;
		}
	}
}

__host__ void Medium::render(Pixel * pixels, int pitch) const {
	k_render<<<params::cells_grid_dim, params::cells_block_dim>>>(
	    *this, pixels, pitch);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to render Medium, {}", res);
}
