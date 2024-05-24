#pragma once

#include "src/util.hh"
#include "src/pixel.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace params {

// Angle of the sensor relative to the direction the agent is facing
static constexpr float sensor_angle_deg = 45.0;
// Angle the agent turns when moving towards food
static constexpr float agent_turn_deg = 30.0;
// Distance of the sensor from the agent's body
static constexpr float sensor_distance = 10;
// Distance an agent steps each turn
static constexpr float agent_step_size = 1.0;
// Decay factory
static constexpr float decay_factor = 0.9;
// Amount to deposit each step
static constexpr int deposit_amount = 2<<20;
// Grid line step (0 to disable)
static constexpr size_t hint_grid_dim = 0;
// Factor to lengthen diagonal probes by
static constexpr float angle_correction_factor = 1; // 1.41421356237;
// Length of debug agent vectors
static constexpr float agent_dir_hint_len = 2.0;
// Hard turn threshold
static constexpr int hard_turn = 2<<17;
// Brightness factor
static constexpr double brightness = 32.0;


static constexpr float max_brightness_cell = brightness * deposit_amount;
static constexpr float sensor_angle_distance = sensor_distance * angle_correction_factor;
static constexpr float sensor_angle_rad = util::deg_to_rad(sensor_angle_deg);
static constexpr float agent_turn_rad = util::deg_to_rad(agent_turn_deg);


// Number of agents in the simulation
static constexpr size_t n_agents = 200000;
// Width of the simulation
static constexpr int width = 320;
// Height of the simulation
static constexpr int height = 512;


static constexpr size_t sized_agent_block_dim = 512;
static constexpr size_t agent_block_dim =
    std::min(sized_agent_block_dim, n_agents);
static constexpr size_t agent_grid_dim =
    util::ceil_div(n_agents, agent_block_dim);

static constexpr size_t cells_block_width = 32;
static constexpr size_t cells_block_height = 32;
static constexpr size_t cells_grid_width =
    util::ceil_div(width, cells_block_width);
static constexpr size_t cells_grid_height =
    util::ceil_div(height, cells_block_height);
static constexpr dim3 cells_block_dim =
    dim3{cells_block_width, cells_block_height};
static constexpr dim3 cells_grid_dim = dim3{cells_grid_width, cells_grid_height};

}  // namespace params
