#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tensor.h"

#define FEATURE_DEPTH_SIZE 512

// GOAL X
#define GOAL_X 25
#define GOAL_Y 0

__global__ void FeatureKernel(Tensor X, Tensor X_prime);
