#pragma once


#ifndef _TENSORMUL_KERNEL_H_
#define _TENSORMUL_KERNEL_H_

#include "tensormul_kernel.h"
#include <math.h>
#include <device_functions.h>

//// Matrix multiplication kernel thread specification
//__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
//{
//  //Multiply the two matrices
//
//  // Each thread will be computing the dot product of the ijth pair of vectors in A and B
//  unsigned i = threadIdx.y + blockIdx.y * blockDim.y;
//  unsigned j = threadIdx.x + blockIdx.x * blockDim.x;
//  
//  if (i < 4 && j < 16)
//  {
//	  float sum = 0;
//	  for (int k = 0; k < 32; ++k)
//	  {
//			sum += M.elements[i * 32 + k] * N.elements[k * 4 + j];
//	  }
//	  P.elements[i * 4 + j] = sum;
//  }  
//  
//}



/* Matrix multiplication: C = A * B.
 * Device code.
 */


#define MEMORYSAFE


// Matrix multiplication kernel thread specification
__global__ void TensorMulKernel(Tensor M_d, Tensor N_d, Tensor P_d)
{
	__shared__ float M_s[BLOCK_SIZE][BLOCK_SIZE][DEPTH_SIZE];
	__shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE][DEPTH_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	int Row = by * BLOCK_SIZE + ty;
	int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * DEPTH_SIZE + tz;

	float Pvalue = 0;

    int MSize = M_d.height * M_d.width;
    int NSize = N_d.height * N_d.width;
    int PSize = P_d.height * P_d.width;
	
	// For each phase
	
	for (int m = 0; m < ceilf(float(M_d.width)/float(BLOCK_SIZE)); ++m)
	{
		
		int mcol = m * BLOCK_SIZE + tx;
		int nrow = m * BLOCK_SIZE + ty;
		
#ifdef MEMORYSAFE
		// Allocate shared memory, with zeros in spaces outside matrix for correct dot product (more divergent)

        if (mcol < M_d.width && Row < M_d.height && Slice < M_d.depth)
        {
            M_s[ty][tx][tz] = M_d.elements[Slice * MSize + Row * M_d.width + mcol];
        }
        else
        {
            M_s[ty][tx][tz] = 0;
        }

        if (nrow < N_d.height && Col < N_d.width && Slice < N_d.depth)
        {
            N_s[ty][tx][tz] = N_d.elements[Slice * NSize + nrow * N_d.width + Col];
        }
        else
        {
            N_s[ty][tx][tz] = 0;
        }

		//M_s[ty][tx] = (mcol < M_d.width && Row < M_d.height)? M_d.elements[Row * M_d.width + mcol] : 0;
        //N_s[ty][tx] = (nrow < N_d.height && Col < N_d.width) ? N_d.elements[nrow * N_d.width + Col] : 0;
#else	
		// fill shared arrays with OOB garbage in memory (less divergent)
		M_s[ty][tx] = M_d.elements[Row * M_d.width + mcol];
		N_s[ty][tx] = N_d.elements[nrow * N_d.width + Col];
#endif
		__syncthreads();
		
		// Calculate partial dot product
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
#ifndef MEMORYSAFE
			if (m * TILE_WIDTH + k < M_d.width) // stop before garbage; don't need this if we're padding zeros in the shared matrices
#endif		
			Pvalue += M_s[ty][k][tz] * N_s[k][tx][tz];
		}
		// Make sure block is fully used by all other threads
		__syncthreads();
	}
	
	// Since other elements in the block depend on data this thread will load, we can't early-terminate
	if( Row < P_d.height && Col < P_d.width && Slice < P_d.depth) 
	{
		P_d.elements[Slice * PSize  +  Row*P_d.width + Col] = Pvalue;
	}
	

}




__global__ void TensorAddKernel(Tensor M_d, Tensor N_d, Tensor P_d)
{
    int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	int Row = by * BLOCK_SIZE + ty;
	int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * DEPTH_SIZE + tz;


    int MSize = M_d.height * M_d.width;
    int NSize = N_d.height * N_d.width;
    int PSize = P_d.height * P_d.width;


    if( Row < P_d.height && Col < P_d.width && Slice < P_d.depth) 
	{
        float Pvalue = M_d.elements[Slice * MSize + Row*M_d.width + Col] +  N_d.elements[Slice * NSize + Row*N_d.width + Col];
		P_d.elements[Slice * PSize  +  Row*P_d.width + Col] = Pvalue;
	}

}




__global__ void TensorReLUKernel(Tensor M_d, Tensor P_d)
{
    int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	int Row = by * BLOCK_SIZE + ty;
	int Col = bx * BLOCK_SIZE + tx;
    int Slice = bz * DEPTH_SIZE + tz;

	float Pvalue = 0;

    int MSize = M_d.height * M_d.width;
    int PSize = P_d.height * P_d.width;

    if( Row < P_d.height && Col < P_d.width && Slice < P_d.depth) 
	{
        float Pvalue = M_d.elements[Slice * MSize + Row*M_d.width + Col];
        Pvalue = fmaxf(Pvalue,0.0f);//(Pvalue > 0) ? Pvalue : 0;
		P_d.elements[Slice * PSize  +  Row*P_d.width + Col] = Pvalue;
	}
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_





//#endif // #ifndef _MATRIXMUL_KERNEL_H_
