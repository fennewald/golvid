#pragma once

#include "src/cuda_util.cuh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/util.hh"

#include <cuda_runtime.h>

class Agent {
public:
	__device__ constexpr Agent(float x, float y, float dir) :
	    m_x{x}, m_y{y}, m_dir{dir} {}

	__device__ inline float x(void) const { return m_x; }
	__device__ inline float y(void) const { return m_y; }
	__device__ inline float dir(void) const { return m_dir; }

	__device__ inline float2 coords(void) const { return float2{m_x, m_y}; }
	__device__ inline int2   icoords(void) const

	__device__ inline Cell sense_l(Medium medium) const {
		return sense_with_dt(medium, params::sensor_angle_rad, params::sensor_angle_distance);
	}
	__device__ inline Cell sense_c(Medium medium) const {
		return sense_with_dt(medium, 0);
	}
	__device__ inline Cell sense_r(Medium medium) const {
		return sense_with_dt(medium, -1 * params::sensor_angle_rad, params::sensor_angle_distance);
	}

	__device__ inline void turn_left(void) {
		m_dir = m_dir + params::agent_turn_rad;
	}
	__device__ inline void turn_right(void) {
		m_dir = m_dir - params::agent_turn_rad;
	}

	__device__ inline void move(void) {
		float dx, dy;
		sincosf(m_dir, &dx, &dy);
		m_x = fmodf(m_x + (dx * params::agent_step_size), params::width);
		m_y = fmodf(m_y + (dy * params::agent_step_size), params::height);
	}

private:
	__device__ inline Cell sense_with_dt(Medium medium, float d, float distance = params::sensor_distance) const {
		float2 p;
		sincosf(m_dir + d, &p.y, &p.x);
		p.x = (p.x * distance) + x();
		p.y = (p.y * distance) + y();
		return *medium.get(int2{static_cast<int>(p.x), static_cast<int>(p.y)});
	}

	float m_x;
	float m_y;
	float m_dir;
};
