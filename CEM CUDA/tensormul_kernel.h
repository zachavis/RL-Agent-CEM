#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tensor.h"

__global__ void TensorMulKernel(Tensor M, Tensor N, Tensor P);

__global__ void TensorAddKernel(Tensor M, Tensor N, Tensor P);

__global__ void TensorReLUKernel(Tensor M, Tensor P);
