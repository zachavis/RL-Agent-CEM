#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tensor.h"

#define DYNAMICS_DEPTH_SIZE 64
#define TIME_DELTA 0.1f

__global__ void CarDynamicsKernel(Tensor X, Tensor U, Tensor X_prime);