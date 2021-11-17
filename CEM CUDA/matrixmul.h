
#pragma once

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define TILE_WIDTH 16
//#define BLOCK_SIZE TILE_WIDTH

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WM TILE_WIDTH // Matrix M width
#define HM TILE_WIDTH // Matrix M height
#define WN TILE_WIDTH // Matrix N width
#define HN WM  // Matrix N height
#define WP WN  // Matrix P width 
#define HP HM  // Matrix P height


// Matrix Structure declaration
typedef struct {
	//width of the matrix represented
    unsigned int width;
	//height of the matrix represented
    unsigned int height;
	//number of elements between the beginnings of adjacent
	// rows in the memory layout (useful for representing sub-matrices)
    unsigned int pitch;
	//Pointer to the first element of the matrix represented
    float* elements;
} Matrix;


#endif // _MATRIXMUL_H_

