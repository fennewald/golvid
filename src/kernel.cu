#include <cstdint>

struct dimensions {
	unsigned int width, height;
};

struct pixel {
	unsigned int r, g, b;
};


__device__ bool get(const uint8_t * base, uint32_t x, uint32_t y) {
	// TODO: impl
	return true;
}

__global__ void kernel(const uint8_t * previous, uint8_t * next, pixel * pixels) {
	next[0] = 12;
}
