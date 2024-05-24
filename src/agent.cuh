#pragma once

#include "src/cuda_util.cuh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/util.hh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>

class Agent {
public:
	__device__ Agent(float x, float y, float dx, float dy) :
	    m_x{x}, m_y{y}, m_dx{dx}, m_dy{dy} {}

	__device__ inline float x(void) const { return m_x; }
	__device__ inline float y(void) const { return m_y; }
	__device__ inline float dx(void) const { return m_dx; }
	__device__ inline float dy(void) const { return m_dy; }

	__device__ inline float2 coords(void) const { return float2{m_x, m_y}; }
	__device__ inline float2 dir(void) const { return float2{m_dx, m_dy}; }

	__device__ inline Cell sense_l(Medium medium) const {
		return sense_with_dt(
		    medium, params::sensor_angle_rad, params::sensor_angle_distance);
	}
	__device__ inline Cell sense_c(Medium medium) const {
		return sense_with_dt(medium, 0);
	}
	__device__ inline Cell sense_r(Medium medium) const {
		return sense_with_dt(
		    medium, -1 * params::sensor_angle_rad, params::sensor_angle_distance);
	}

	__device__ inline float2 rotated(float rad) const {
		float2 res;
		res.x = (dx() * cosf(rad)) + (dy() * sinf(rad));
		res.y = (-1 * dx() * sinf(rad)) + (dy() * cosf(rad));
		return res;
	}

	__device__ inline void rotate(float rad) {
		auto res = rotated(rad);
		m_dx = res.x;
		m_dy = res.y;
	}

	__device__ inline long rand() {
		return curand(&m_rng);
	}

	__device__ inline void turn_left(void) { rotate(params::agent_turn_rad); }
	__device__ inline void turn_right(void) {
		rotate(-1 * params::agent_turn_rad);
	}
	__device__ inline void turn_rand(void) {
		unsigned int x = curand(&m_rng);
		if ((x & 1) == 1) {
			turn_left();
		} else {
			turn_right();
		}
	}


	__device__ inline void move(void) {
		m_x += m_dx * params::agent_step_size;
		m_y += m_dy * params::agent_step_size;

		if (m_x < 0) {
			m_x *= -1;
			m_dx *= -1;
		}
		if (m_y < 0) {
			m_y *= -1;
			m_dy *= -1;
		}
		if (m_x >= params::width) {
			m_x = (2 * params::width) - m_x;
			m_dx *= -1;
		}
		if (m_y >= params::height) {
			m_y = (2 * params::height) - m_y;
			m_dy *= -1;
		}

		// m_x = fmodf(m_x + (dx * params::agent_step_size), params::width);
		// m_y = fmodf(m_y + (dy * params::agent_step_size), params::height);
	}

private:
	__device__ inline Cell sense_with_dt(
	    Medium medium, float d, float distance = params::sensor_distance) const {
		auto r = rotated(d);
		auto s = float2{r.x * distance + m_x, r.y * distance + m_y};

		return *medium.get(cu_util::round_f2(s));
	}

	float m_x;
	float m_y;
	float m_dx;
	float m_dy;
	// Rng state
	curandState m_rng;
};
