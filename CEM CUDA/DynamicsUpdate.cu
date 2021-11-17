#pragma once


#ifndef _DYNAMICS_KERNEL_H_
#define _DYNAMICS_KERNEL_H_

#include "DynamicsUpdate.h"
#include <math.h>
#include <device_functions.h>


// Expects X is a 3 x 1 x N Tensor,
// Expects U is a 4 x 1 x N Tensor,
// Rturns X' is a 3 x 1 x N Tensor,
__global__ void CarDynamicsKernel(Tensor X, Tensor U, Tensor X_prime)
{
    //int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z;

	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	//int Row = by * BLOCK_SIZE + ty;
	//int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * DYNAMICS_DEPTH_SIZE + tz;

	float Pvalue = 0;

    int XSize = X.height * X.width;
    int USize = U.height * U.width;
    int X_primeSize = X_prime.height * X_prime.width;

    float X_s[3] = {0};
    float U_s[4] = {0};

    if (Slice < X_prime.depth)
    {
        X_s[0] = X.elements[Slice * XSize + 0 * X.width];
        X_s[1] = X.elements[Slice * XSize + 1 * X.width];
        X_s[2] = X.elements[Slice * XSize + 2 * X.width];
    
        U_s[0] = U.elements[Slice * USize + 0 * U.width];
        U_s[1] = U.elements[Slice * USize + 1 * U.width];
        U_s[2] = U.elements[Slice * USize + 2 * U.width];
        U_s[3] = U.elements[Slice * USize + 3 * U.width];
        
        U_s[0] = fmin(40.f,      U_s[0]);
        U_s[0] = fmax(-40.f,     U_s[0]);
        U_s[1] = fmin(3.14159f,  U_s[1]);
        U_s[1] = fmax(-3.14159f, U_s[1]);

    
        X_prime.elements[Slice * X_primeSize + 0 * X_prime.width] =   X_s[0] + TIME_DELTA * U_s[0] *cos(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 1 * X_prime.width] =   X_s[1] + TIME_DELTA * U_s[0] *sin(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 2 * X_prime.width] =   X_s[2] + TIME_DELTA * U_s[1];

    }

}


#endif // #ifndef _DYNAMICS_KERNEL_H_