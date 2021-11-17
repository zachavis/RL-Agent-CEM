#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "matrixmul.h"

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);
