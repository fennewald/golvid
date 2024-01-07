#pragma once

#include <cstdint>

#include "src/pixel.hh"

struct res_t {
	unsigned int width, height;
};

void step(
    uint8_t ** prev,
    uint8_t ** next,
    res_t      res,
    int        pitch,
    pixel_t *  pixels,
    int        pix_pitch);

void initalize(
    uint8_t * w0, uint8_t * w1, res_t res, int pitch, pixel_t * pixels, int pix_pitch);
