#pragma once


#ifndef _REWARDS_KERNEL_H_
#define _REWARDS_KERNEL_H_

#include "reward_kernel.h"
#include <math.h>
#include <device_functions.h>

__global__ void RewardsKernel(Tensor X, Tensor U, float * R, int t)
{
    //int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z;

	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	//int Row = by * BLOCK_SIZE + ty;
	//int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * REWARDS_DEPTH_SIZE + tz;

	float Pvalue = 0;

    int XSize = X.height * X.width;
    int USize = U.height * U.width;
    // int X_primeSize = X_prime.height * X_prime.width;

    float X_s[3] = {0};
    float U_s[4] = {0};

    if (Slice < X.depth)
    {
        X_s[0] = X.elements[Slice * XSize + 0 * X.width];
        X_s[1] = X.elements[Slice * XSize + 1 * X.width];
        X_s[2] = X.elements[Slice * XSize + 2 * X.width];
    
        U_s[0] = U.elements[Slice * USize + 0 * U.width];
        U_s[1] = U.elements[Slice * USize + 1 * U.width];
        U_s[2] = U.elements[Slice * USize + 2 * U.width];
        U_s[3] = U.elements[Slice * USize + 3 * U.width];

        float dist_x = GOAL_X - X_s[0];
        float dist_y = GOAL_Y - X_s[1];

        float goal_pointing = sinf(-X_s[2]) * dist_x + cosf(-X_s[2]) * dist_y;
/*
        float */
            
        float penalty = fabsf(goal_pointing);// + .01 * (U_s[0]*U_s[0] + U_s[1]*U_s[1]);

        R[Slice] += powf(.95,t) * (1.0f / (sqrtf(dist_x * dist_x + dist_y * dist_y) + 0.1f) - penalty);

        /*X_prime.elements[Slice * X_primeSize + 0 * X_prime.width] =   X_s[0] + U_s[0] *cos(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 1 * X_prime.width] =   X_s[1] + U_s[0] *sin(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 2 * X_prime.width] =   X_s[2] + U_s[1];*/

    }
}



__global__ void FeatureKernel(Tensor X, Tensor X_prime)
{
    //int bx = blockIdx.x;
	//int by = blockIdx.y;
	int bz = blockIdx.z;

	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	//int Row = by * BLOCK_SIZE + ty;
	//int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * REWARDS_DEPTH_SIZE + tz;

	float Pvalue = 0;

    int XSize = X.height * X.width;
    //int USize = U.height * U.width;
    int X_primeSize = X_prime.height * X_prime.width;

    float X_s[3] = {0};
    float U_s[4] = {0};

    if (Slice < X_prime.depth)
    {
        X_s[0] = X.elements[Slice * XSize + 0 * X.width];
        X_s[1] = X.elements[Slice * XSize + 1 * X.width];
        X_s[2] = X.elements[Slice * XSize + 2 * X.width];
    
        //U_s[0] = U.elements[Slice * USize + 0 * U.width];
        //U_s[1] = U.elements[Slice * USize + 1 * U.width];
        //U_s[2] = U.elements[Slice * USize + 2 * U.width];
        //U_s[3] = U.elements[Slice * USize + 3 * U.width];

        float dist_x = GOAL_X - X_s[0];
        float dist_y = GOAL_Y - X_s[1];

        //R[Slice] += 1.0f / (sqrt(dist_x * dist_x + dist_y * dist_y) + 0.1f);

        X_prime.elements[Slice * X_primeSize + 0 * X_prime.width] =   cosf(-X_s[2]) * dist_x - sinf(-X_s[2]) * dist_y; // X_s[0] + U_s[0] *cos(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 1 * X_prime.width] =   sinf(-X_s[2]) * dist_x + cosf(-X_s[2]) * dist_y; //X_s[1] + U_s[0] *sin(X_s[2]);
        X_prime.elements[Slice * X_primeSize + 2 * X_prime.width] =   X_s[2];

    }
}
//
//Eigen::Vector3f GenFeature(Eigen::Vector3f x, Eigen::Vector3f g)
//{
//    Eigen::Vector2f diff = g.head<2>() - x.head<2>();
//    float rotdifx = cos(-x[2]) * diff[0] - sin(-x[2]) * diff[1];
//    float rotdify = sin(-x[2]) * diff[0] + cos(-x[2]) * diff[1];
//
//    return Eigen::Vector3f(rotdifx,rotdify,x[2]);
//}

#endif