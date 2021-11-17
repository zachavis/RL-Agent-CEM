#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tensor.h"

#define REWARDS_DEPTH_SIZE 64
#define FEATURE_DEPTH_SIZE 64

// GOAL X
#define GOAL_X 25.0f
#define GOAL_Y 0.0f

__global__ void RewardsKernel(Tensor X, Tensor U, float * R, int t);

__global__ void FeatureKernel(Tensor X, Tensor X_prime);
