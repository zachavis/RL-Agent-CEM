#pragma once

#ifndef _TENSORMUL_H_
#define _TENSORMUL_H_

// Thread block size
#define BLOCK_SIZE 16
#define DEPTH_SIZE 4

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
//#define WM MATRIX_SIZE // Matrix M width
//#define HM MATRIX_SIZE // Matrix M height
//#define WN MATRIX_SIZE // Matrix N width
//#define HN WM  // Matrix N height
//#define WP WN  // Matrix P width 
//#define HP HM  // Matrix P height


// Matrix Structure declaration
typedef struct {
	//width of the matrix represented
    unsigned int width;
	//height of the matrix represented
    unsigned int height;

    // number of matrices stacked in tensor
    unsigned int depth;

	////number of elements between the beginnings of adjacent
	//// rows in the memory layout (useful for representing sub-matrices)
 //   unsigned int pitch;


	//Pointer to the first element of the matrix represented
    float* elements;
} Tensor;


#endif // _MATRIXMUL_H_

