#pragma once

#include "src/util.hh"

#include <cuda_runtime.h>

#include <cstddef>

namespace params {

// Angle of the sensor relative to the direction the agent is facing
inline constexpr float sensor_angle_deg = 45;
// Angle the agent turns when moving towards food
inline constexpr float agent_turn_deg = sensor_angle_deg;
// Distance of the sensor from the agent's body
inline constexpr float sensor_distance = 2;
// Distance an agent steps each turn
inline constexpr float agent_step_size = 2;


inline constexpr float sensor_angle_rad = util::deg_to_rad(sensor_angle_deg);
inline constexpr float agent_turn_rad = util::deg_to_rad(agent_turn_deg);


// Number of agents in the simulation
inline constexpr size_t n_agents = 2;
// Width of the simulation
inline constexpr int width = 200;
// Height of the simulation
inline constexpr int height = 200;


inline constexpr size_t agent_block_dim = 128;
inline constexpr size_t agent_grid_dim =
    util::ceil_div(n_agents, agent_block_dim);

inline constexpr size_t cells_block_width = 32;
inline constexpr size_t cells_block_height = 32;
inline constexpr size_t cells_grid_width =
    util::ceil_div(width, cells_block_width);
inline constexpr size_t cells_grid_height =
    util::ceil_div(height, cells_block_height);
inline constexpr dim3 cells_block_dim =
    dim3{cells_block_width, cells_block_height};
inline constexpr dim3 cells_grid_dim = dim3{cells_grid_width, cells_grid_height};

}  // namespace params
