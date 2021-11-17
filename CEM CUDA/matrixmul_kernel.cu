/* Matrix multiplication: P = M * N.
 * Device code.
 */
#pragma once

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "matrixmul_kernel.h"
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
__global__ void MatrixMulKernel(Matrix M_d, Matrix N_d, Matrix P_d)
{
	__shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float N_s[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	
	// For each phase
		
	for (int m = 0; m < ceilf(float(M_d.width)/float(TILE_WIDTH)); ++m) //TODO don't assume width is a perfect multiple of tile_width
	{
		
		int mcol = m * TILE_WIDTH + tx;
		int nrow = m * TILE_WIDTH + ty;
		
#ifdef MEMORYSAFE
		// Allocate shared memory, with zeros in spaces outside matrix for correct dot product (more divergent)
		M_s[ty][tx] = (mcol < M_d.width && Row < M_d.height)? M_d.elements[Row * M_d.width + mcol] : 0;
        N_s[ty][tx] = (nrow < N_d.height && Col < N_d.width) ? N_d.elements[nrow * N_d.width + Col] : 0;
#else	
		// fill shared arrays with OOB garbage in memory (less divergent)
		M_s[ty][tx] = M_d.elements[Row * M_d.width + mcol];
		N_s[ty][tx] = N_d.elements[nrow * N_d.width + Col];
#endif
		__syncthreads();
		
		// Calculate partial dot product
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
#ifndef MEMORYSAFE
			if (m * TILE_WIDTH + k < M_d.width) // stop before garbage; don't need this if we're padding zeros in the shared matrices
#endif		
			Pvalue += M_s[ty][k] * N_s[k][tx];
		}
		// Make sure block is fully used by all other threads
		__syncthreads();
	}
	
	// Since other elements in the block depend on data this thread will load, we can't early-terminate
	if( Row < P_d.height && Col < P_d.width) 
	{
		P_d.elements[Row*P_d.width + Col] = Pvalue;
	}
	

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_





//#endif // #ifndef _MATRIXMUL_KERNEL_H_
