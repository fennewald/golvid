#include "src/sim.cuh"

#include "src/params.hh"
#include "src/vec.cuh"


/*******************************************************************************/

__global__ void
k_render(const Cell * cells, int cell_pitch, Pixel * pixels, int pix_pitch) {
	auto coords = idx_2();
	if (coords.x >= static_cast<int>(params::width)) return;
	if (coords.y >= static_cast<int>(params::height)) return;

	render_cell(
	    pitch_ptr(cells, coords, cell_pitch),
	    pitch_ptr(pixels, coords, pix_pitch));
}

__host__ void
render(const Cell * cells, int cell_pitch, Pixel * pixels, int pix_pitch) {
	k_render<<<params::cells_grid_dim, params::cells_block_dim>>>(
	    cells, cell_pitch, pixels, pix_pitch);
}

__global__ void media_step(const Cell * prev, Cell * next, int pitch) {
	auto coords = idx_2();
	if (coords.x >= static_cast<int>(params::width)) return;
	if (coords.y >= static_cast<int>(params::height)) return;

	Cell * output = pitch_ptr(next, coords, pitch);
	Cell   c = get_avg_cell(prev, coords, pitch);
	// Cell c = *pitch_ptr(prev, coords, pitch);

	decay(&c);
	*output = c;
}

__host__ void step(
    Cell ** prev,
    Cell ** next,
    int     cell_pitch,
    float * x,
    float * y,
    float * dir,
    Pixel * pixels,
    int     pix_pitch) {
	agent_step<<<params::agent_grid_dim, params::agent_block_dim>>>(
	    x, y, dir, *prev, cell_pitch);

	deposit<<<params::agent_grid_dim, params::agent_block_dim>>>(
	    x, y, *prev, cell_pitch);

	media_step<<<params::cells_grid_dim, params::cells_block_dim>>>(
	    *prev, *next, cell_pitch);

	render(*next, cell_pitch, pixels, pix_pitch);

	Cell * tmp = *prev;
	*prev = *next;
	*next = tmp;
}
