
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>

#include <Eigen/Dense>

//
//#include "matrixmul.h"
#include "matrixmul_kernel.h"
//#include "tensormul.h"
#include "tensormul_kernel.h"
#include "DynamicsUpdate.h"
#include "reward_kernel.h"


#include <numeric> //iota
#include <algorithm> //min
#include <random>

#include <fstream>




// TODO: Move to top since global
std::default_random_engine generator (8980);
std::normal_distribution<float> distribution(0,1); // TODO check range

std::uniform_real_distribution<float> uniform(-1,1);
std::uniform_real_distribution<float> uniform01(0,1);

//#define DEBUG_TEXT

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//#define TIME_DELTA 0.1f;
//const float TIME_DELTA = 0.1f;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int Test1()
{
    Eigen::VectorXf test;
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


//#define MATRIX_SIZE 8


Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}
Tensor AllocateDeviceTensor(const Tensor T)
{
    Tensor Tdevice = T;
    int size = T.width * T.height * T.depth * sizeof(float);
    cudaMalloc((void**)&Tdevice.elements, size);
    return Tdevice;
}


void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
               cudaMemcpyHostToDevice);
}
void CopyToDeviceTensor(Tensor Tdevice, const Tensor Thost)
{
    int size = Thost.width * Thost.height * Thost.depth * sizeof(float);
    Tdevice.height = Thost.height;
    Tdevice.width = Thost.width;
    Tdevice.depth = Thost.depth;
    // Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Tdevice.elements, Thost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
               cudaMemcpyDeviceToHost);
}
// Copy a device matrix to a host matrix.
void CopyFromDeviceTensor(Tensor Thost, const Tensor Tdevice)
{
    int size = Tdevice.width * Tdevice.height * Tdevice.depth * sizeof(float);
    cudaMemcpy(Thost.elements, Tdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }
    return M;
}

bool CheckNAN(const float * elements, unsigned int size)
{
    bool flag = false;
    for (int i = 0; i < size; ++i)
    {
        
        if (isnan(elements[i]))
        {
            printf("uh oh! at %d\n",i);
            flag = true;
        }
            
    }
    return flag;
}



Tensor AllocateTensor(int height, int width, int depth, int init)
{
    Tensor T; 
    T.width  = width;
    T.height = height;
    T.depth  = depth;
    int size = T.width * T.height * T.depth;
    T.elements = NULL;

    T.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < size; i++)
    {
        T.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }
    return T;
}

//Tensor AllocateRandomDDStateTensor(int depth, int init)
//{
//    Tensor T; 
//    T.width  = 1;
//    T.height = 3;
//    T.depth  = depth;
//    int size = T.width * T.height * T.depth;
//    T.elements = NULL;
//
//    T.elements = (float*) malloc(size*sizeof(float));
//
//    for(unsigned int i = 0; i < size; i++)
//    {
//        float val = 0.0f;
//        if (i%0==0)
//        {
//
//        } 
//        else if(i%1==0)
//        {
//
//        }
//        else
//        {
//
//        }
//
//
//        T.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
//    }
//    return T;
//}

void SetStateTensorRandom(float * elements, int size)
{
    for (int i = 0; i < size; i+=3)
    {
        elements[i] = uniform01(generator) * -40;
        elements[i+1] = uniform(generator) * 40;
        elements[i+2] = uniform(generator) * 3.14159;
    }
}

Tensor AllocateTensorCenteredRandom(int height, int width, int depth, float scale)
{
    Tensor T; 
    T.width  = width;
    T.height = height;
    T.depth  = depth;
    int size = T.width * T.height * T.depth;
    T.elements = NULL;

    T.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < size; i++)
    {
        T.elements[i] = (rand() / (float)RAND_MAX) * scale * 2 - scale;
    }
    return T;
}

Tensor AllocateTensorFixed(int height, int width, int depth, float value)
{
    Tensor T; 
    T.width  = width;
    T.height = height;
    T.depth  = depth;
    int size = T.width * T.height * T.depth;
    T.elements = NULL;

    T.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < size; i++)
    {
        T.elements[i] = value;
    }
    return T;
}




// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a device matrix.
void FreeDeviceTensor(Tensor* T)
{
    cudaFree(T->elements);
    T->elements = NULL;
}



void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{

    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(ceil(float(P.width)/TILE_WIDTH),ceil(float(P.height)/TILE_WIDTH),1);
	//dim3 DG(1,1,1);
	dim3 DB(TILE_WIDTH,TILE_WIDTH,1);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    MatrixMulKernel<<<DG,DB>>>(Md,Nd,Pd);


    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);


}



void TensorMulOnDevice(const Tensor M, const Tensor N, Tensor P)
{
    // Load M and N to the device
    Tensor Md = AllocateDeviceTensor(M);
    CopyToDeviceTensor(Md, M);
    Tensor Nd = AllocateDeviceTensor(N);
    CopyToDeviceTensor(Nd, N);

    // Allocate P on the device
    Tensor Pd = AllocateDeviceTensor(P);
    CopyToDeviceTensor(Pd, P); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(ceil(float(P.width)/BLOCK_SIZE),ceil(float(P.height)/BLOCK_SIZE),ceil(float(P.depth)/DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    TensorMulKernel<<<DG,DB>>>(Md,Nd,Pd);


    // Read P from the device
    CopyFromDeviceTensor(P, Pd); 

    // Free device matrices
    FreeDeviceTensor(&Md);
    FreeDeviceTensor(&Nd);
    FreeDeviceTensor(&Pd);

}





extern "C"
////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wA         width of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            float sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                float a = A[i * wA + k];
                float b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

//! @param dA         depth of both matrices
void computeTensorGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int hB, unsigned int wB, unsigned int dA)
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        for (unsigned int i = 0; i < hA; ++i)
            for (unsigned int j = 0; j < wB; ++j) {
                float sum = 0;
                for (unsigned int k = 0; k < wA; ++k) {
                    float a = A[z * wA * hA + i * wA + k];
                    float b = B[z * wB * hB + k * wB + j];
                    sum += a * b;
                }
                C[z * wB * hA + i * wB + j] = (float)sum;
            }
    }
}



// returns true iff A and B have same elements in same order
bool CompareMatrices(Matrix A, Matrix B, float accuracy = 0.0001f) {
    unsigned int size = A.width * A.height;

    if ( (A.width != B.width) || (A.height != B.height) )
        return false;

    for (unsigned i = 0; i < size; i++)
        if (abs(A.elements[i] - B.elements[i]) > accuracy) 
            return false;
    return true;
}
// returns true iff A and B have same elements in same order
bool CompareTensors(Tensor A, Tensor B, float accuracy = 0.0001f) {
    bool flag = true;
    unsigned int size = A.width * A.height * A.depth;

    if ( (A.width != B.width) || (A.height != B.height) || (A.depth != B.depth) )
        return false;

    for (unsigned i = 0; i < size; i++)
    {
        if (abs(A.elements[i] - B.elements[i]) > accuracy) 
        {
            #ifdef DEBUG_TEXT
            printf("BAD AT [%d]: %f vs %f\n",i,A.elements[i],B.elements[i]);
            #endif  
            return false;//flag = false;//return false;
        }
    }
    return true;
}


bool CompareArrays(float * A, float * B, unsigned int size, float accuracy = 0.0001f) {
    
    for (unsigned i = 0; i < size; i++)
        if (abs(A[i] - B[i]) > accuracy)
        {
            #ifdef DEBUG_TEXT
            printf("BAD AT [%d]: %f vs %f\n",i,A[i],B[i]);
            #endif  
            return false;
        }
    return true;
}




int Test2()
{
    int A = 4;
    int B = 1025;
    int C = 16;

    Matrix M  = AllocateMatrix(A, B, 1);
    Matrix N  = AllocateMatrix(M.width, C, 1);
    Matrix P  = AllocateMatrix(M.height, N.width, 0);

    //Matrix dM = AllocateDeviceMatrix(M);

    printf("%d\n",M.height);
    //printf("%d\n",dM.height);



    MatrixMulOnDevice(M, N, P);

    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);

    // check if the device result is equivalent to the expected solution
    bool res = CompareMatrices(reference, P);
    printf("Test %s\n", res ? "PASSED" : "FAILED");
    
    for (int i = 0; i < P.height * P.width; ++i)
    {
        printf("%f, %f\n", reference.elements[i], P.elements[i] );
    }


    return 0;
}





void BatchMatrixMultiplyTest()
{

    // return Test1();
   
    //Eigen::MatrixXf eM = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eN = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eP;
    int A = 4;
    int B = 1025;
    int C = 16;
    int D = 10000;

    Tensor M  = AllocateTensor(A,           B,          D, 1);
    Tensor N  = AllocateTensor(M.width,     C,          D, 1);
    Tensor P  = AllocateTensor(M.height,    N.width,    D, 0);

    //Matrix dM = AllocateDeviceMatrix(M);

    //printf("%d\n",M.height);
    //printf("%d\n",dM.height);



    //MatrixMulOnDevice(M, N, P);
    TensorMulOnDevice(M,N,P);

    // compute the matrix multiplication on the CPU for comparison
    Tensor reference = AllocateTensor(P.height, P.width, P.depth, 0);
    computeTensorGold(reference.elements, M.elements, N.elements, M.height, M.width, N.height, N.width, M.depth);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareTensors(reference, P);
    
    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED", D);
    return;
}


void CarUpdateOnDevice(const Tensor X, const Tensor U, Tensor X_Prime)
{
    // Load M and N to the device
    Tensor Xd = AllocateDeviceTensor(X);
    CopyToDeviceTensor(Xd, X);
    Tensor Ud = AllocateDeviceTensor(U);
    CopyToDeviceTensor(Ud, U);

    // Allocate P on the device
    Tensor X_Primed = AllocateDeviceTensor(X_Prime);
    CopyToDeviceTensor(X_Primed, X_Prime); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(1,1,ceil(float(X_Prime.depth)/DYNAMICS_DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(1,1,DYNAMICS_DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    CarDynamicsKernel<<<DG,DB>>>(Xd,Ud,X_Primed);
    // cudaDeviceSynchronize();

    // Read P from the device
    CopyFromDeviceTensor(X_Prime, X_Primed); 

    // Free device matrices
    FreeDeviceTensor(&Xd);
    FreeDeviceTensor(&Ud);
    FreeDeviceTensor(&X_Prime);

}

void computeCarGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int hB, unsigned int wB, unsigned int dA)
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        //for (unsigned int i = 0; i < hA; ++i)
        //    for (unsigned int j = 0; j < wB; ++j) {
        //        float sum = 0;
        //        /*for (unsigned int k = 0; k < wA; ++k) {
        //            float a = A[z * wA * hA + i * wA + k];
        //            float b = B[z * wB * hB + k * wB + j];
        //            sum += a * b;
        //        }*/
        //        C[z * wB * hA + i * wB + j] = (float)sum;
        //    }

        float v = B[z*wB*hB+0*wB+0];
        float w = B[z*wB*hB+1*wB+0];
        if (true)//CLIP_CONTROL)
        {
            v = v > 40 ? 40 : v;
            v = v < -40 ? -40 : v;
            
            w = w > 3.14159 ? 3.14159 : w;
            w = w < -3.14159 ? -3.14159 : w;
        }
        
        C[z*wB*hA+0*wB+0] = A[z*wA*hA+0*wA] + TIME_DELTA * cos(A[z*wA*hA+2*wA+0]) * v;
        C[z*wB*hA+1*wB+0] = A[z*wA*hA+1*wA] + TIME_DELTA * sin(A[z*wA*hA+2*wA+0]) * v;
        C[z*wB*hA+2*wB+0] = A[z*wA*hA+2*wA] + TIME_DELTA * w;

        /*C[z*wB*hA+0*wB+0] = C[z*wB*hA+0*wB] + cos(C[z*wB*hA+2*wB+0]) * B[z*wB*hA+0*wB+0];
        C[z*wB*hA+1*wB+0] = C[z*wB*hA+1*wB] + sin(C[z*wB*hA+2*wB+0]) * B[z*wB*hA+0*wB+0];
        C[z*wB*hA+2*wB+0] = C[z*wB*hA+2*wB] + B[z*wB*hA+1*wB+0];*/
    }


}



void CarTest()
{

    int DEPTH = 10000;
    Tensor X = AllocateTensor(3,1,DEPTH,1);
    Tensor U = AllocateTensor(4,1,DEPTH,1);
    Tensor X_Prime   = AllocateTensor(X.height,     X.width,    DEPTH, 0);


    CarUpdateOnDevice(X,U,X_Prime);

    // compute the matrix multiplication on the CPU for comparison
    Tensor reference = AllocateTensor(X_Prime.height, X_Prime.width, X_Prime.depth, 0);
    computeCarGold(reference.elements,X.elements,U.elements,X.height,X.width,U.height,U.width,X.depth);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareTensors(reference, X_Prime);
    

    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    // printf("Depth: %d\n",DEPTH);
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED",DEPTH);

    
#ifdef DEBUG_TEXT

    for (int z = 0; z < 2; ++z){

        for (int i = 0; i < 3; ++i){
            printf("X[%d,1,%d]: %f\n",i,z,X.elements[z * X.width * X.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 2; ++i){
            printf("U[%d,1,%d]: %f\n",i,z,U.elements[z * U.width * U.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 3; ++i){
            printf("X'[%d,1,%d]: %f\n",i,z,X_Prime.elements[z * X_Prime.width * X_Prime.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 3; ++i){
            printf("R'[%d,1,%d]: %f\n",i,z,reference.elements[z * X_Prime.width * X_Prime.height + i]);
        }
        printf("\n");
    }

   /* for (int i = 0; i < U.depth * U.width * U.height; ++i)
    {
                printf("u %f\n",U.elements[i]);
    }

    for (int i = 0; i < X_Prime.depth * X_Prime.width * X_Prime.height; ++i)
    {
                printf("x' %f\n",X_Prime.elements[i]);
    }*/
#endif

    
    return;
}




void TensorAddOnDevice(const Tensor M, const Tensor N, Tensor P)
{
    // Load M and N to the device
    Tensor Md = AllocateDeviceTensor(M);
    CopyToDeviceTensor(Md, M);
    Tensor Nd = AllocateDeviceTensor(N);
    CopyToDeviceTensor(Nd, N);

    // Allocate P on the device
    Tensor Pd = AllocateDeviceTensor(P);
    CopyToDeviceTensor(Pd, P); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(ceil(float(P.width)/BLOCK_SIZE),ceil(float(P.height)/BLOCK_SIZE),ceil(float(P.depth)/DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    TensorAddKernel<<<DG,DB>>>(Md,Nd,Pd);


    // Read P from the device
    CopyFromDeviceTensor(P, Pd); 

    // Free device matrices
    FreeDeviceTensor(&Md);
    FreeDeviceTensor(&Nd);
    FreeDeviceTensor(&Pd);

}

void addTensorGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int dA)
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        for (unsigned int i = 0; i < hA; ++i)
            for (unsigned int j = 0; j < wA; ++j) {
                
                C[z * wA * hA + i * wA + j] = (float)(A[z * wA * hA + i * wA + j] + B[z * wA * hA + i * wA + j]);
            }
    }
}

void BatchMatrixAddTest()
{

    // return Test1();
   
    //Eigen::MatrixXf eM = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eN = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eP;
    int A = 4;
    int B = 1025;
    //int C = 16;
    int D = 10000;

    Tensor M  = AllocateTensor(A,           B,          D, 1);
    Tensor N  = AllocateTensor(M.height,     M.width,   D, 1);
    Tensor P  = AllocateTensor(M.height,    M.width,    D, 0);

    //Matrix dM = AllocateDeviceMatrix(M);

    //printf("%d\n",M.height);
    //printf("%d\n",dM.height);



    //MatrixMulOnDevice(M, N, P);
    //TensorMulOnDevice(M,N,P);
    TensorAddOnDevice(M,N,P);

    // compute the matrix multiplication on the CPU for comparison
    Tensor reference = AllocateTensor(P.height, P.width, P.depth, 0);
    addTensorGold(reference.elements, M.elements, N.elements, M.height, M.width, M.depth);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareTensors(reference, P);
    
    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED", D);
    return;
}




void TensorReLUOnDevice(const Tensor M, Tensor P)
{
    // Load M and N to the device
    Tensor Md = AllocateDeviceTensor(M);
    CopyToDeviceTensor(Md, M);

    // Allocate P on the device
    Tensor Pd = AllocateDeviceTensor(P);
    CopyToDeviceTensor(Pd, P); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(ceil(float(P.width)/BLOCK_SIZE),ceil(float(P.height)/BLOCK_SIZE),ceil(float(P.depth)/DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    TensorReLUKernel<<<DG,DB>>>(Md,Pd);


    // Read P from the device
    CopyFromDeviceTensor(P, Pd); 

    // Free device matrices
    FreeDeviceTensor(&Md);
    FreeDeviceTensor(&Pd);

}

void reluTensorGold(float* C, const float* A, unsigned int hA, unsigned int wA, unsigned int dA)
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        for (unsigned int i = 0; i < hA; ++i)
            for (unsigned int j = 0; j < wA; ++j) {
                float val = (float)A[z * wA * hA + i * wA + j];
                val = (val > 0)?val:0;
                C[z * wA * hA + i * wA + j] = val;
            }
    }
}

void BatchMatrixReLUTest()
{

    // return Test1();
   
    //Eigen::MatrixXf eM = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eN = Eigen::MatrixXf::Random(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXf eP;
    int A = 4;
    int B = 1025;
    //int C = 16;
    int D = 10000;

    Tensor M  = AllocateTensor(A,           B,          D, 1);
    Tensor P  = AllocateTensor(M.height,    M.width,    D, 0);

    //Matrix dM = AllocateDeviceMatrix(M);

    //printf("%d\n",M.height);
    //printf("%d\n",dM.height);



    //MatrixMulOnDevice(M, N, P);
    //TensorMulOnDevice(M,N,P);
    TensorReLUOnDevice(M,P);

    // compute the matrix multiplication on the CPU for comparison
    Tensor reference = AllocateTensor(P.height, P.width, P.depth, 0);
    reluTensorGold(reference.elements, M.elements, M.height, M.width, M.depth);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareTensors(reference, P);
    
    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED",D);
    return;
}




void RewardOnDevice(const Tensor X, const Tensor U, float * R, float t)
{
    // Load M and N to the device
    Tensor Xd = AllocateDeviceTensor(X);
    CopyToDeviceTensor(Xd, X);
    Tensor Ud = AllocateDeviceTensor(U);
    CopyToDeviceTensor(Ud, U);

    // Allocate R on the device
    float * Rd = R;
    int Rsize = X.depth * sizeof(float);
    cudaMalloc((void**)&Rd, Rsize);
    cudaMemcpy(Rd, R, Rsize, cudaMemcpyHostToDevice);

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(1,1,ceil(float(X.depth)/REWARDS_DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(1,1,REWARDS_DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    //CarDynamicsKernel<<<DG,DB>>>(Xd,Ud,X_Primed);
    // cudaDeviceSynchronize();
    RewardsKernel<<<DG,DB>>>(Xd,Ud,Rd,t);
    //RewardsKernel<<<DG,DB>>>(Xd,Ud,Rd);

    //printf("rrrr%f\n",R[0]);
    //
    //printf("rrrr%f\n",R[1]);

    // Read R from the device
    cudaMemcpy(R, Rd, Rsize, cudaMemcpyDeviceToHost);
    
    
    //printf("rrrr%f\n",R[0]);
    //
    //printf("rrrr%f\n",R[1]);

    // Free device matrices
    FreeDeviceTensor(&Xd);
    FreeDeviceTensor(&Ud);
    cudaFree(Rd);
    Rd = NULL;

}

void computeRewardGold(float* R, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int dA, int t) // TODO: Should have U information for extending method
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        float x = A[z*wA*hA+0*wA];
        float y = A[z*wA*hA+1*wA];
        float theta = A[z*wA*hA+2*wA];

        float dist_x = GOAL_X - x;
        float dist_y = GOAL_Y - y;

        float goal_pointing = sinf(-theta) * dist_x + cosf(-theta) * dist_y;
            
        float penalty = fabsf(goal_pointing);

        R[z] += powf(.95,t) * (1.0f / (sqrt(dist_x * dist_x + dist_y * dist_y) + 0.1f) - penalty);
    }
    /*for (unsigned int z = 0; z < dA; ++z)
    {
        float x = A[z*wA*hA+0*wA];
        float y = A[z*wA*hA+1*wA];

        float dist_x = GOAL_X - x;
        float dist_y = GOAL_Y - y;

        R[z] += 1.0f / (sqrt(dist_x * dist_x + dist_y * dist_y) + 0.1f);
    }*/
}

void RewardTest()
{

    int DEPTH = 10000;
    Tensor X = AllocateTensor(3,1,DEPTH,1);
    Tensor U = AllocateTensor(4,1,DEPTH,1);
    

    int size = X.depth;
    float * R = (float*) malloc(size*sizeof(float));
    for(unsigned int i = 0; i < size; i++)
    {
        R[i] = (1 == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }

    float * reference = (float*) malloc(size*sizeof(float));
    for(unsigned int i = 0; i < size; i++)
    {
        reference[i] = R[i];//(0 == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }

    RewardOnDevice(X,U,R,0);
    //CarUpdateOnDevice(X,U,X_Prime);

    // compute the matrix multiplication on the CPU for comparison
    //Tensor reference = AllocateTensor(X_Prime.height, X_Prime.width, X_Prime.depth, 0);
    
    

    //computeCarGold(reference.elements,X.elements,U.elements,X.height,X.width,U.height,U.width,X.depth);
    computeRewardGold(reference,X.elements,U.elements,X.height,X.width,X.depth,0);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareArrays(reference, R, size);
    
    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    //printf("Depth: %d\n",DEPTH);
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED", DEPTH);


#ifdef DEBUG_TEXT
    for (int z = 0; z < 2; ++z){

        for (int i = 0; i < 3; ++i){
            printf("X[%d,1,%d]: %f\n",i,z,X.elements[z * X.width * X.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 2; ++i){
            printf("U[%d,1,%d]: %f\n",i,z,U.elements[z * U.width * U.height + i]);
        }
        printf("\n");
        printf("R[%d]: %f\n",z,R[z]);
        printf("\n");
        printf("ref[%d]: %f\n",z,reference[z]);
        printf("\n");
    }
#endif
   /* for (int i = 0; i < U.depth * U.width * U.height; ++i)
    {
                printf("u %f\n",U.elements[i]);
    }

    for (int i = 0; i < X_Prime.depth * X_Prime.width * X_Prime.height; ++i)
    {
                printf("x' %f\n",X_Prime.elements[i]);
    }*/

    
    return;
}



void FeatureOnDevice(const Tensor X, Tensor X_Prime)
{
    // Load M and N to the device
    Tensor Xd = AllocateDeviceTensor(X);
    CopyToDeviceTensor(Xd, X);

    // Allocate P on the device
    Tensor X_Primed = AllocateDeviceTensor(X_Prime);
    CopyToDeviceTensor(X_Primed, X_Prime); // Clear memory

    // Setup the execution configuration
	//unsigned threadsPerBlock = 16;
	dim3 DG(1,1,ceil(float(X_Prime.depth)/FEATURE_DEPTH_SIZE));
	//dim3 DG(1,1,1);
	dim3 DB(1,1,FEATURE_DEPTH_SIZE);

	//printf("Calling kernel function with P.shape = (%d,%d)\n", P.height, P.width);
	//printf("DG = (%d,%d,%d)\n", DG.x, DG.y, DG.z);
	//printf("DB = (%d,%d,%d)\n", DB.x, DB.y, DB.z);
    // Launch the device computation threads!
    FeatureKernel<<<DG,DB>>>(Xd,X_Primed);
    // cudaDeviceSynchronize();

    // Read P from the device
    CopyFromDeviceTensor(X_Prime, X_Primed); 

    // Free device matrices
    FreeDeviceTensor(&Xd);
    FreeDeviceTensor(&X_Prime);

}

void computeFeatureGold(float* C, const float* A, unsigned int hA, unsigned int wA, unsigned int dA)
{
    for (unsigned int z = 0; z < dA; ++z)
    {
        //for (unsigned int i = 0; i < hA; ++i)
        //    for (unsigned int j = 0; j < wB; ++j) {
        //        float sum = 0;
        //        /*for (unsigned int k = 0; k < wA; ++k) {
        //            float a = A[z * wA * hA + i * wA + k];
        //            float b = B[z * wB * hB + k * wB + j];
        //            sum += a * b;
        //        }*/
        //        C[z * wB * hA + i * wB + j] = (float)sum;
        //    }

        float x     = A[z*wA*hA+0*wA];
        float y     = A[z*wA*hA+1*wA];
        float angle = A[z*wA*hA+2*wA+0];
        float diff_x = GOAL_X - x;
        float diff_y = GOAL_Y - y;
        
        C[z*wA*hA+0*wA+0] =     cos(-angle) * diff_x - sin(-angle) * diff_y; // X_s[0] + U_s[0] *cos(X_s[2]);
        C[z*wA*hA+1*wA+0] =     sin(-angle) * diff_x + cos(-angle) * diff_y; //X_s[1] + U_s[0] *sin(X_s[2]);
        C[z*wA*hA+2*wA+0] =     angle;

        /*C[z*wB*hA+0*wB+0] = C[z*wB*hA+0*wB] + cos(C[z*wB*hA+2*wB+0]) * B[z*wB*hA+0*wB+0];
        C[z*wB*hA+1*wB+0] = C[z*wB*hA+1*wB] + sin(C[z*wB*hA+2*wB+0]) * B[z*wB*hA+0*wB+0];
        C[z*wB*hA+2*wB+0] = C[z*wB*hA+2*wB] + B[z*wB*hA+1*wB+0];*/
    }


}



void FeatureTest()
{

    int DEPTH = 10000;
    Tensor X         = AllocateTensor(3, 1, DEPTH, 1);
    Tensor X_Prime   = AllocateTensor(3, 1, DEPTH, 0);


    FeatureOnDevice(X,X_Prime);

    // compute the matrix multiplication on the CPU for comparison
    Tensor reference = AllocateTensor(X_Prime.height, X_Prime.width, X_Prime.depth, 0);
    computeFeatureGold(reference.elements,X.elements,X.height,X.width,X.depth);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res = CompareTensors(reference, X_Prime);
    

    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    // printf("Depth: %d\n",DEPTH);
    printf("Test %s [%d]\n", res ? "PASSED" : "FAILED",DEPTH);

    
#ifdef DEBUG_TEXT

    for (int z = 0; z < 2; ++z){

        for (int i = 0; i < 3; ++i){
            printf("X[%d,1,%d]: %f\n",i,z,X.elements[z * X.width * X.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 3; ++i){
            printf("X'[%d,1,%d]: %f\n",i,z,X_Prime.elements[z * X_Prime.width * X_Prime.height + i]);
        }
        printf("\n");
    }

   /* for (int i = 0; i < U.depth * U.width * U.height; ++i)
    {
                printf("u %f\n",U.elements[i]);
    }

    for (int i = 0; i < X_Prime.depth * X_Prime.width * X_Prime.height; ++i)
    {
                printf("x' %f\n",X_Prime.elements[i]);
    }*/
#endif

    
    return;
}




void TestAllKernels()
{    
    printf("Running BMM kernel...");
    BatchMatrixMultiplyTest();
    cudaDeviceSynchronize();
    printf("Running BMA kernel...");
    BatchMatrixAddTest();
    cudaDeviceSynchronize();
    printf("Running BMR kernel...");
    BatchMatrixReLUTest();
    cudaDeviceSynchronize();
    printf("Running car kernel...");
    CarTest();
    cudaDeviceSynchronize();
    printf("Running reward kernel...");
    RewardTest();
    cudaDeviceSynchronize();
    printf("Running feature kernel...");
    FeatureTest();
    cudaDeviceSynchronize();
}






void StepOnDevice(Tensor X, Tensor X_feat, Tensor U, Tensor X_Prime, Tensor W1, Tensor B1, Tensor Hidden, Tensor W2, Tensor B2, float * R)
{

    

    //  X_prime = CarDynamics( X, (W2*relu(W1*X + B1) + B2) )
    //  Rewards += Reward(X,X_prime)
    //  X = X_prime

    // Load to device
    Tensor X_d = AllocateDeviceTensor(X);
    CopyToDeviceTensor(X_d, X);

    Tensor X_feat_d = AllocateDeviceTensor(X_feat);
    CopyToDeviceTensor(X_feat_d, X_feat);

    Tensor U_d = AllocateDeviceTensor(U);
    CopyToDeviceTensor(U_d, U);

    Tensor X_Prime_d = AllocateDeviceTensor(X_Prime);
    CopyToDeviceTensor(X_Prime_d, X_Prime);

    Tensor W1_d = AllocateDeviceTensor(W1);
    CopyToDeviceTensor(W1_d, W1);

    Tensor B1_d = AllocateDeviceTensor(B1);
    CopyToDeviceTensor(B1_d, B1);

    Tensor Hidden_d = AllocateDeviceTensor(Hidden);
    CopyToDeviceTensor(Hidden_d, Hidden);

    Tensor W2_d = AllocateDeviceTensor(W2);
    CopyToDeviceTensor(W2_d, W2);

    Tensor B2_d = AllocateDeviceTensor(B2);
    CopyToDeviceTensor(B2_d, B2);


    float * R_d = R;
    int Rsize = X_Prime.depth * sizeof(float);
    cudaMalloc((void**)&R_d, Rsize);
    cudaMemcpy(R_d, R, Rsize, cudaMemcpyHostToDevice);
    

    // Execute on device

    dim3 DG(1,1,ceil(float(X_d.depth)/FEATURE_DEPTH_SIZE));
	dim3 DB(1,1,FEATURE_DEPTH_SIZE);
    FeatureKernel<<<DG,DB>>>(X_d,X_feat_d);
    cudaDeviceSynchronize();

	DG = dim3(ceil(float(Hidden.width)/BLOCK_SIZE),ceil(float(Hidden.height)/BLOCK_SIZE),ceil(float(Hidden.depth)/DEPTH_SIZE));
	DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
    TensorMulKernel<<<DG,DB>>>(W1_d,X_feat_d,Hidden_d);
    cudaDeviceSynchronize();
    TensorAddKernel<<<DG,DB>>>(Hidden_d,B1_d,Hidden_d);
    cudaDeviceSynchronize();

    TensorReLUKernel<<<DG,DB>>>(Hidden_d,Hidden_d);
    cudaDeviceSynchronize();

	DG = dim3(ceil(float(U_d.width)/BLOCK_SIZE),ceil(float(U_d.height)/BLOCK_SIZE),ceil(float(U_d.depth)/DEPTH_SIZE));
	DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
    TensorMulKernel<<<DG,DB>>>(W2_d,Hidden_d,U_d);
    cudaDeviceSynchronize();
    TensorAddKernel<<<DG,DB>>>(U_d,B2_d,U_d);
    cudaDeviceSynchronize();

    DG = dim3(1,1,ceil(float(X_Prime.depth)/DYNAMICS_DEPTH_SIZE));
	DB = dim3(1,1,DYNAMICS_DEPTH_SIZE);
    CarDynamicsKernel<<<DG,DB>>>(X_d,U_d,X_Prime_d);
    cudaDeviceSynchronize();
   
    DG = dim3(1,1,ceil(float(X_Prime.depth)/REWARDS_DEPTH_SIZE));
	DB = dim3(1,1,REWARDS_DEPTH_SIZE);
    RewardsKernel<<<DG,DB>>>(X_Prime_d,U_d,R_d,0);
    cudaDeviceSynchronize();

    // Swap pointers
    Tensor Temp = X_d;
    X_d = X_Prime_d;
    X_Prime_d = Temp;

    DG = dim3(1,1,ceil(float(X_d.depth)/FEATURE_DEPTH_SIZE));
	DB = dim3(1,1,FEATURE_DEPTH_SIZE);
    FeatureKernel<<<DG,DB>>>(X_d,X_feat_d);
    cudaDeviceSynchronize();

	DG = dim3(ceil(float(Hidden.width)/BLOCK_SIZE),ceil(float(Hidden.height)/BLOCK_SIZE),ceil(float(Hidden.depth)/DEPTH_SIZE));
	DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
    TensorMulKernel<<<DG,DB>>>(W1_d,X_feat_d,Hidden_d);
    cudaDeviceSynchronize();
    TensorAddKernel<<<DG,DB>>>(Hidden_d,B1_d,Hidden_d);
    cudaDeviceSynchronize();

    TensorReLUKernel<<<DG,DB>>>(Hidden_d,Hidden_d);
    cudaDeviceSynchronize();

	DG = dim3(ceil(float(U_d.width)/BLOCK_SIZE),ceil(float(U_d.height)/BLOCK_SIZE),ceil(float(U_d.depth)/DEPTH_SIZE));
	DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
    TensorMulKernel<<<DG,DB>>>(W2_d,Hidden_d,U_d);
    cudaDeviceSynchronize();
    TensorAddKernel<<<DG,DB>>>(U_d,B2_d,U_d);
    cudaDeviceSynchronize();

    DG = dim3(1,1,ceil(float(X_Prime.depth)/DYNAMICS_DEPTH_SIZE));
	DB = dim3(1,1,DYNAMICS_DEPTH_SIZE);
    CarDynamicsKernel<<<DG,DB>>>(X_d,U_d,X_Prime_d);
    cudaDeviceSynchronize();
   
    DG = dim3(1,1,ceil(float(X_Prime.depth)/REWARDS_DEPTH_SIZE));
	DB = dim3(1,1,REWARDS_DEPTH_SIZE);
    RewardsKernel<<<DG,DB>>>(X_Prime_d,U_d,R_d,1);
    cudaDeviceSynchronize();


    
    CopyFromDeviceTensor(X_feat, X_feat_d);
    CopyFromDeviceTensor(Hidden, Hidden_d); 
    CopyFromDeviceTensor(U, U_d); 
    CopyFromDeviceTensor(X_Prime, X_Prime_d); 
    cudaMemcpy(R, R_d, Rsize, cudaMemcpyDeviceToHost);

    //// Read P from the device
    //CopyFromDeviceTensor(X_Prime, X_Primed); 

    //// Free device matrices
    //FreeDeviceTensor(&Xd);
    //FreeDeviceTensor(&X_Prime);
      
    FreeDeviceTensor(&X_d);
    FreeDeviceTensor(&X_feat_d);
    FreeDeviceTensor(&U_d);
    FreeDeviceTensor(&X_Prime_d);
    FreeDeviceTensor(&W1_d);
    FreeDeviceTensor(&B1_d);
    FreeDeviceTensor(&Hidden_d);
    FreeDeviceTensor(&W2_d);
    FreeDeviceTensor(&B2_d);
    cudaFree(R_d);
    R_d = NULL;

}


void computeStepGold(Tensor X, Tensor X_feat, Tensor U, Tensor X_Prime, Tensor W1, Tensor B1, Tensor Hidden, Tensor W2, Tensor B2, float * R)
{

 //   dim3 DG(1,1,ceil(float(X_Prime.depth)/FEATURE_DEPTH_SIZE));
	//dim3 DB(1,1,FEATURE_DEPTH_SIZE);
 //   FeatureKernel<<<DG,DB>>>(X_d,X_feat_d);

	//DG = dim3(ceil(float(Hidden.width)/BLOCK_SIZE),ceil(float(Hidden.height)/BLOCK_SIZE),ceil(float(Hidden.depth)/DEPTH_SIZE));
	//DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
 //   TensorMulKernel<<<DG,DB>>>(W1_d,X_feat_d,Hidden_d);
 //   TensorAddKernel<<<DG,DB>>>(Hidden_d,B1_d,Hidden_d);

 //   TensorReLUKernel<<<DG,DB>>>(Hidden_d,Hidden_d);

	//DG = dim3(ceil(float(U_d.width)/BLOCK_SIZE),ceil(float(U_d.height)/BLOCK_SIZE),ceil(float(U_d.depth)/DEPTH_SIZE));
	//DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
 //   TensorMulKernel<<<DG,DB>>>(W2_d,Hidden_d,U_d);
 //   TensorAddKernel<<<DG,DB>>>(X_Prime_d,B2_d,U_d);

 //   DG = dim3(1,1,ceil(float(X_Prime.depth)/DYNAMICS_DEPTH_SIZE));
	//DB = dim3(1,1,DYNAMICS_DEPTH_SIZE);
 //   CarDynamicsKernel<<<DG,DB>>>(X_d,U_d,X_Prime_d);
 //  
 //   DG = dim3(1,1,ceil(float(X_Prime.depth)/REWARDS_DEPTH_SIZE));
	//DB = dim3(1,1,REWARDS_DEPTH_SIZE);
 //   RewardsKernel<<<DG,DB>>>(X_d,U_d,R_d);


    // COMPUTATION
    computeFeatureGold(X_feat.elements,X.elements,X.height,X.width,X.depth);

    computeTensorGold(Hidden.elements,W1.elements,X_feat.elements,W1.height,W1.width,X_feat.height,X_feat.width,W1.depth);
    addTensorGold(Hidden.elements,B1.elements,Hidden.elements,B1.height,B1.width,B1.depth);

    reluTensorGold(Hidden.elements,Hidden.elements,Hidden.height,Hidden.width,Hidden.depth);

    computeTensorGold(U.elements,W2.elements,Hidden.elements,W2.height,W2.width,Hidden.height,Hidden.width,W2.depth);
    addTensorGold(U.elements,B2.elements,U.elements,B2.height,B2.width,B2.depth);

    computeCarGold(X_Prime.elements,X.elements,U.elements,X.height,X.width,U.height,U.width,X.depth);

    computeRewardGold(R,X_Prime.elements,U.elements,X_Prime.height,X_Prime.width,X_Prime.depth,0);


    computeFeatureGold(X_feat.elements,X_Prime.elements,X_Prime.height,X_Prime.width,X_Prime.depth);

    computeTensorGold(Hidden.elements,W1.elements,X_feat.elements,W1.height,W1.width,X_feat.height,X_feat.width,W1.depth);
    addTensorGold(Hidden.elements,B1.elements,Hidden.elements,B1.height,B1.width,B1.depth);

    reluTensorGold(Hidden.elements,Hidden.elements,Hidden.height,Hidden.width,Hidden.depth);

    computeTensorGold(U.elements,W2.elements,Hidden.elements,W2.height,W2.width,Hidden.height,Hidden.width,W2.depth);
    addTensorGold(U.elements,B2.elements,U.elements,B2.height,B2.width,B2.depth);

    computeCarGold(X_Prime.elements,X_Prime.elements,U.elements,X_Prime.height,X_Prime.width,U.height,U.width,X_Prime.depth);

    computeRewardGold(R,X_Prime.elements,U.elements,X_Prime.height,X_Prime.width,X_Prime.depth,1);

}


void StepTest()
{
    printf("Performing 2 step trajectory with deep policy test...\n");
    int DEPTH = 10000;
    int SIZE_IN = 3;
    int SIZE_HIDDEN = 200;
    int SIZE_OUT = 4;
    Tensor X        = AllocateTensor(SIZE_IN, 1, DEPTH, 1); // TODO: generate these within a feasible range
    Tensor X_feat   = AllocateTensor(SIZE_IN, 1, DEPTH, 1);
    Tensor U        = AllocateTensor(SIZE_OUT, 1, DEPTH, 1);
    Tensor X_Prime  = AllocateTensor(SIZE_IN, 1, DEPTH, 0);
    
    Tensor W1  = AllocateTensorCenteredRandom(SIZE_HIDDEN, SIZE_IN,   DEPTH,  .01);
    Tensor B1  = AllocateTensorCenteredRandom(SIZE_HIDDEN,    1,      DEPTH,  .01);
    Tensor Hidden = AllocateTensor(SIZE_HIDDEN, 1,      DEPTH,  1);
    
    Tensor W2  = AllocateTensorCenteredRandom(SIZE_OUT, SIZE_HIDDEN,  DEPTH,  .01);
    Tensor B2  = AllocateTensorCenteredRandom(SIZE_OUT, 1,            DEPTH,  .01);

    float * R = (float*) malloc(DEPTH*sizeof(float));
    for(unsigned int i = 0; i < DEPTH; i++)
    {
        R[i] = 0; //(1 == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }


    // compute the matrix multiplication on the CPU for comparison
    Tensor reference_X = AllocateTensor(X.height, X.width, X.depth, 0);
    Tensor reference_X_feat = AllocateTensor(X_feat.height, X_feat.width, X_feat.depth, 0);
    Tensor reference_U = AllocateTensor(U.height, U.width, U.depth, 0);
    Tensor reference_X_Prime = AllocateTensor(X_Prime.height, X_Prime.width, X_Prime.depth, 0);
    
    Tensor reference_W1 = AllocateTensor(W1.height, W1.width, W1.depth, 0);
    Tensor reference_B1 = AllocateTensor(B1.height, B1.width, B1.depth, 0);
    Tensor reference_Hidden = AllocateTensor(Hidden.height, Hidden.width, Hidden.depth, 0);
    Tensor reference_W2 = AllocateTensor(W2.height, W2.width, W2.depth, 0);
    Tensor reference_B2 = AllocateTensor(B2.height, B2.width, B2.depth, 0);
    float * reference_R = (float*) malloc(DEPTH*sizeof(float));
    for(unsigned int i = 0; i < DEPTH; i++)
    {
        reference_R[i] = R[i];//(0 == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }






    StepOnDevice(X,X_feat,U,X_Prime,W1,B1,Hidden,W2,B2,R);

    //computeRewardGold(reference_R,X.elements,X.height,X.width,X.depth);

    computeStepGold(X, reference_X_feat, reference_U, reference_X_Prime, W1, B1, reference_Hidden, W2, B2, reference_R);
    
    cudaDeviceSynchronize();
    // check if the device result is equivalent to the expected solution
    bool res_X_feat     = CompareTensors(reference_X_feat, X_feat, .1);
    bool res_Hidden     = CompareTensors(reference_Hidden, Hidden, .1);
    bool res_U          = CompareTensors(reference_U, U, .1);
    bool res_X_Prime    = CompareTensors(reference_X_Prime, X_Prime, .1);
    bool res_R          = CompareArrays(reference_R, R, DEPTH, .1);
    


    /*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    {
        printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    }*/
    // printf("Depth: %d\n",DEPTH);
    printf("Test X_feat %s [%d]\n", res_X_feat ? "PASSED" : "FAILED",DEPTH);
    printf("Test Hidden %s [%d]\n", res_Hidden ? "PASSED" : "FAILED",DEPTH);
    printf("Test U %s [%d]\n", res_U ? "PASSED" : "FAILED",DEPTH);
    printf("Test X' %s [%d]\n", res_X_Prime ? "PASSED" : "FAILED",DEPTH);
    printf("Test R %s [%d]\n", res_R ? "PASSED" : "FAILED",DEPTH);
    printf("Test %s [%d]\n", res_X_feat && res_Hidden && res_U && res_X_Prime && res_R ? "PASSED" : "FAILED" , DEPTH );



    /*
    TensorMulOnDevice(W1,X,Hidden);
    computeTensorGold(reference_Hidden,X,B,)*/



    
#ifdef DEBUG_TEXT

     for (int z = 0; z < 2; ++z){
        printf("R[%d]: %f\n",z,R[z]);
        printf("\n");
        printf("Rref[%d]: %f\n",z,reference_R[z]);
        printf("\n");
        
        for (int i = 0; i < 3; ++i){
            printf("X'[%d,1,%d]: %f\n",i,z,X_Prime.elements[z * X_Prime.width * X_Prime.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 3; ++i){
            printf("X'ref[%d,1,%d]: %f\n",i,z,reference_X_Prime.elements[z * X_Prime.width * X_Prime.height + i]);
        }
        printf("\n");

        for (int i = 0; i < 4; ++i){
            printf("U[%d,1,%d]: %f\n",i,z,U.elements[z * U.width * U.height + i]);
        }
        printf("\n");
        for (int i = 0; i < 4; ++i){
            printf("Uref[%d,1,%d]: %f\n",i,z,reference_U.elements[z * U.width * U.height + i]);
        }
        printf("\n");
     }

   /* for (int i = 0; i < U.depth * U.width * U.height; ++i)
    {
                printf("u %f\n",U.elements[i]);
    }

    for (int i = 0; i < X_Prime.depth * X_Prime.width * X_Prime.height; ++i)
    {
                printf("x' %f\n",X_Prime.elements[i]);
    }*/
#endif

    
    return;
}



void RandomSampleDistribution(float * thetas, const float* theta_mean, const float* theta_stddev, unsigned int size, unsigned int mean_size)
{
    for (int i = 0; i < size; ++i)
    {
        float rand = distribution(generator);
        if ( isnan(rand) || isnan(theta_mean[i % mean_size]) || isnan(theta_stddev[i % mean_size]) )
            printf("OOPS at %d [%f,%f,%f]\n", i,rand,theta_mean[i],theta_stddev[i]);
        thetas[i] = theta_mean[i % mean_size] + rand * theta_stddev[i % mean_size];
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}

void ZeroTensor(float * elements, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        elements[i] = 0;
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}

void PlusEqualsTensor(float * A, const float* B, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        A[i] += B[i];
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}

void PlusEqualsTensor(float * A, const float scalar, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        A[i] += scalar;
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}


void MinusEqualsTensor(float * A, const float* B, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        A[i] -= B[i];
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}

void StdDevTensor(float * stddev, const float* newmean, const float* parameter, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        stddev[i] += (newmean[i] - parameter[i]) * (newmean[i] - parameter[i]);
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}

void MultEqualsTensor(float * A, const float scalar, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        A[i] *= scalar;
        //printf("%f\n",thetas[i]);
        //printf("%f + %f = %f\n",theta_mean.elements[i],rand * theta_stddev.elements[i],thetas.elements[i]);
    }
}


// sorts in decreasing order for CEM
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
std::vector<unsigned int> sort_indexes(const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<unsigned int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](unsigned int i1, unsigned int i2) {return v[i1] > v[i2];});

  return idx;
}

Tensor CEM(int blank, Tensor & theta_mean, int batch_size, int n_iter, float elite_frac, float initial_std=0.1f);

int main()
{

    ///*printf("Running BMM Test...\n");
    //BatchMatrixMultiplyTest();*/
    //
    //int IN_SIZE = 200;
    //int HIDDEN_SIZE = 1;
    //int OUT_SIZE = 4;
    //int TRIALS = 10000;

    //Tensor W1  = AllocateTensor(4,       200,       TRIALS, 1);
    //Tensor X   = AllocateTensor(W1.width,     1,          TRIALS, 1);
    //Tensor P   = AllocateTensor(W1.height,     X.width,    TRIALS, 0);

    //TensorMulOnDevice(W1,X,P);

    //// compute the matrix multiplication on the CPU for comparison
    //Tensor reference = AllocateTensor(P.height, P.width, P.depth, 0);
    //computeTensorGold(reference.elements, W1.elements, X.elements, W1.height, W1.width, X.height, X.width, W1.depth);

    //// check if the device result is equivalent to the expected solution
    //bool res = CompareTensors(reference, P);
    //
    ///*for (int i = 0; i < P.height * P.width * P.depth; ++i)
    //{
    //    printf("%i : %f, %f\n", i, reference.elements[i], P.elements[i] );
    //}*/
    //printf("Test %s\n", res ? "PASSED" : "FAILED");
    //printf("Result size: %d rows, %d columns", P.height, P.width);

    //CarTest();
    //printf("Running reward kernel...\n");
    //RewardTest();

    //
    
    //printf("doing something\n");

    ////RewardTest();
    printf("Testing kernels\n");
    TestAllKernels();
    printf("Testing step\n");
    StepTest();
    ////CarTest();
    //printf("did something\n");


    
    int DEPTH = 100;
    int SIZE_IN = 3;
    int SIZE_HIDDEN = 200;
    int SIZE_OUT = 4;

    float initial_stddev = .1f;
    //printf("Running random sample\n");
    Tensor W1_mean =        AllocateTensorFixed(SIZE_HIDDEN,SIZE_IN,1,0);
    int W1_mean_size =      W1_mean.height * W1_mean.width;
    Tensor W1_stddev =      AllocateTensorCenteredRandom(SIZE_HIDDEN,SIZE_IN,1,initial_stddev);
    int W1_stddev_size =    W1_stddev.height * W1_stddev.width;
    Tensor W1 =             AllocateTensor(SIZE_HIDDEN,SIZE_IN,DEPTH,0);
    int W1_size =           W1.height*W1.width * W1.depth;

    Tensor B1_mean =        AllocateTensorFixed(SIZE_HIDDEN,1,1,0);
    int B1_mean_size =      B1_mean.height * B1_mean.width;
    Tensor B1_stddev =      AllocateTensorCenteredRandom(SIZE_HIDDEN,1,1,initial_stddev);
    int B1_stddev_size =    B1_stddev.height * B1_stddev.width;
    Tensor B1  =            AllocateTensor(SIZE_HIDDEN, 1, DEPTH,  0);
    int B1_size =           B1.height*B1.width * B1.depth;
    
    Tensor W2_mean =        AllocateTensorFixed(SIZE_OUT,SIZE_HIDDEN,1,0);
    int W2_mean_size =      W2_mean.height * W2_mean.width;
    Tensor W2_stddev =      AllocateTensorCenteredRandom(SIZE_OUT,SIZE_HIDDEN,1,initial_stddev);
    int W2_stddev_size =    W2_stddev.height * W2_stddev.width;
    Tensor W2 =             AllocateTensor(SIZE_OUT,SIZE_HIDDEN,DEPTH,0);
    int W2_size =           W2.height*W2.width * W2.depth;
    
    Tensor B2_mean =        AllocateTensorFixed(SIZE_OUT,1,1,0);
    int B2_mean_size =      B2_mean.height * B2_mean.width;
    Tensor B2_stddev =      AllocateTensorCenteredRandom(SIZE_OUT,1,1,initial_stddev);
    int B2_stddev_size =    B2_stddev.height * B2_stddev.width;
    Tensor B2  =            AllocateTensor(SIZE_OUT, 1, DEPTH,  0);
    int B2_size =           B2.height*B2.width * B2.depth;

    
    float * R = (float*) malloc(DEPTH*sizeof(float));
    for(unsigned int i = 0; i < DEPTH; i++)
    {
        R[i] = 0; //(rand() / (float)RAND_MAX);
    }


    // Allocate space on GPU
    // Allocate host Tensors
    Tensor X        = AllocateTensor(SIZE_IN, 1, DEPTH, 1); // TODO: generate these within a feasible range
    Tensor X_feat   = AllocateTensor(SIZE_IN, 1, DEPTH, 1);
    Tensor U        = AllocateTensor(SIZE_OUT, 1, DEPTH, 1);
    Tensor X_Prime  = AllocateTensor(SIZE_IN, 1, DEPTH, 0);
    
    //Tensor W1  = AllocateTensorCenteredRandom(SIZE_HIDDEN, SIZE_IN,   DEPTH,  .01);
    //Tensor B1  = AllocateTensorCenteredRandom(SIZE_HIDDEN,    1,      DEPTH,  .01);
    Tensor Hidden = AllocateTensor(SIZE_HIDDEN, 1,      DEPTH,  1);
    
    //Tensor W2  = AllocateTensorCenteredRandom(SIZE_OUT, SIZE_HIDDEN,  DEPTH,  .01);
    //Tensor B2  = AllocateTensorCenteredRandom(SIZE_OUT, 1,            DEPTH,  .01);



    // Load to device
    SetStateTensorRandom(X.elements,X.height*X.width*X.depth);
    Tensor X_d = AllocateDeviceTensor(X);
    CopyToDeviceTensor(X_d, X);

    Tensor X_feat_d = AllocateDeviceTensor(X_feat);
    CopyToDeviceTensor(X_feat_d, X_feat);

    Tensor U_d = AllocateDeviceTensor(U);
    CopyToDeviceTensor(U_d, U);

    Tensor X_Prime_d = AllocateDeviceTensor(X_Prime);
    CopyToDeviceTensor(X_Prime_d, X_Prime);

    Tensor W1_d = AllocateDeviceTensor(W1);
    CopyToDeviceTensor(W1_d, W1);

    Tensor B1_d = AllocateDeviceTensor(B1);
    CopyToDeviceTensor(B1_d, B1);

    Tensor Hidden_d = AllocateDeviceTensor(Hidden);
    CopyToDeviceTensor(Hidden_d, Hidden);

    Tensor W2_d = AllocateDeviceTensor(W2);
    CopyToDeviceTensor(W2_d, W2);

    Tensor B2_d = AllocateDeviceTensor(B2);
    CopyToDeviceTensor(B2_d, B2);


    float * R_d = R;
    int Rsize = X_Prime.depth * sizeof(float);
    cudaMalloc((void**)&R_d, Rsize);
    cudaMemcpy(R_d, R, Rsize, cudaMemcpyHostToDevice);
    

    unsigned int CEM_EPOCHS = 3500;
    unsigned int CEM_NUM_TRAJECTORY = 150;
    unsigned int CEM_N_ELITE = 10; // 10 from 100

    for (int epoch = 0; epoch < CEM_EPOCHS; ++epoch)
    {
#ifdef DEBUG_TEXT
        printf("epoch %d\n",epoch+1);
#endif
        // Set new starting positions TODO: randomize
        SetStateTensorRandom(X.elements,X.height*X.width*X.depth);
        CopyToDeviceTensor(X_d, X);

        // Zero Rewards
        for(unsigned int i = 0; i < DEPTH; i++)
        {
            R[i] = 0; //(rand() / (float)RAND_MAX);
        }
        cudaMemcpy(R_d, R, Rsize, cudaMemcpyHostToDevice);


#ifdef DEBUG_TEXT
        printf("Random sample\n");
#endif
        // Draw DEPTH parameters from theta_mean and theta_stddev
        RandomSampleDistribution(W1.elements,W1_mean.elements,W1_stddev.elements, W1_size, W1_mean_size);
        RandomSampleDistribution(W2.elements,W2_mean.elements,W2_stddev.elements, W2_size, W2_mean_size);
        RandomSampleDistribution(B1.elements,B1_mean.elements,B1_stddev.elements, B1_size, B1_mean_size);
        RandomSampleDistribution(B2.elements,B2_mean.elements,B2_stddev.elements, B2_size, B2_mean_size);

    
        // Send new parameters to GPU
#ifdef DEBUG_TEXT
        printf("Sending to GPU\n");
#endif
        CopyToDeviceTensor(W1_d, W1);
        CopyToDeviceTensor(B1_d, B1);
        CopyToDeviceTensor(W2_d, W2);
        CopyToDeviceTensor(B2_d, B2);


        // Run trajectories
#ifdef DEBUG_TEXT
        printf("Running trajectories\n");
#endif
        for (int t = 0; t < CEM_NUM_TRAJECTORY; ++t)
        {
            dim3 DG(1,1,ceil(float(X_d.depth)/FEATURE_DEPTH_SIZE));
	        dim3 DB(1,1,FEATURE_DEPTH_SIZE);
            FeatureKernel<<<DG,DB>>>(X_d,X_feat_d);
            cudaDeviceSynchronize();

	        DG = dim3(ceil(float(Hidden.width)/BLOCK_SIZE),ceil(float(Hidden.height)/BLOCK_SIZE),ceil(float(Hidden.depth)/DEPTH_SIZE));
	        DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
            TensorMulKernel<<<DG,DB>>>(W1_d,X_feat_d,Hidden_d);
            cudaDeviceSynchronize();
            TensorAddKernel<<<DG,DB>>>(Hidden_d,B1_d,Hidden_d);
            cudaDeviceSynchronize();

            TensorReLUKernel<<<DG,DB>>>(Hidden_d,Hidden_d);
            cudaDeviceSynchronize();

	        DG = dim3(ceil(float(U_d.width)/BLOCK_SIZE),ceil(float(U_d.height)/BLOCK_SIZE),ceil(float(U_d.depth)/DEPTH_SIZE));
	        DB = dim3(BLOCK_SIZE,BLOCK_SIZE,DEPTH_SIZE);
            TensorMulKernel<<<DG,DB>>>(W2_d,Hidden_d,U_d);
            cudaDeviceSynchronize();
            TensorAddKernel<<<DG,DB>>>(U_d,B2_d,U_d);
            cudaDeviceSynchronize();

            DG = dim3(1,1,ceil(float(X_Prime.depth)/DYNAMICS_DEPTH_SIZE));
	        DB = dim3(1,1,DYNAMICS_DEPTH_SIZE);
            CarDynamicsKernel<<<DG,DB>>>(X_d,U_d,X_Prime_d);
            cudaDeviceSynchronize();
   
            DG = dim3(1,1,ceil(float(X_Prime.depth)/REWARDS_DEPTH_SIZE));
	        DB = dim3(1,1,REWARDS_DEPTH_SIZE);
            RewardsKernel<<<DG,DB>>>(X_Prime_d,U_d,R_d,t);
            cudaDeviceSynchronize();

            // Swap pointers
            Tensor Temp = X_d;
            X_d = X_Prime_d;
            X_Prime_d = Temp;
        } // End Trajectory

        //cudaDeviceSynchronize();


        //CopyFromDeviceTensor(X_feat, X_feat_d);
        //CopyFromDeviceTensor(Hidden, Hidden_d); 
        //CopyFromDeviceTensor(U, U_d); 
        //CopyFromDeviceTensor(X_Prime, X_Prime_d); 

#ifdef DEBUG_TEXT
        printf("Sorting rewards\n");
#endif
        // Get rewards off GPU

        //for(int i = 0; i < DEPTH; ++i)
        //{
        //    printf("R %f\n",R[i]);
        //}
        cudaMemcpy(R, R_d, Rsize, cudaMemcpyDeviceToHost);
        // Vectorize
        std::vector<float> vec_rewards(R, R + DEPTH);
        // Sort from highest to lowest
        std::vector<unsigned int> elite = sort_indexes(vec_rewards);

        int first_reasonable = 0;
        /*for(int i = 0; i < DEPTH; ++i)
        {
            if(R[elite[i]] < 100000)
            {
                first_reasonable = i;
                break;
            }
        }*/


        printf("%d/%d, Best reward: %f\n", epoch+1, CEM_EPOCHS, R[elite[first_reasonable]]);//,first_reasonable);
        /*if (R[elite[first_reasonable]] > 100000)
        {
            for(int i = 0; i < DEPTH; ++i)
                printf("(%d,%d) = %f\n",i,elite[i],R[elite[i]]);
            break;
        }*/


#ifdef DEBUG_TEXT
        printf("Zero dists\n");
#endif
        // Zero old distributions
        ZeroTensor(W1_mean.elements,W1_mean_size);
        ZeroTensor(W2_mean.elements,W2_mean_size);
        ZeroTensor(B1_mean.elements,B1_mean_size);
        ZeroTensor(B2_mean.elements,B2_mean_size);

        ZeroTensor(W1_stddev.elements,W1_stddev_size);
        ZeroTensor(W2_stddev.elements,W2_stddev_size);
        ZeroTensor(B1_stddev.elements,B1_stddev_size);
        ZeroTensor(B2_stddev.elements,B2_stddev_size);
    

#ifdef DEBUG_TEXT
        printf("fitting mean\n");
#endif
        // Fit new mean
        int starting = first_reasonable;
        int stopping = CEM_N_ELITE + starting;
        if (stopping >= DEPTH)
        {
            printf("CRITICAL FAILURE");
            return;
        }
        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            PlusEqualsTensor(W1_mean.elements, W1.elements + elite[i] * W1_mean_size, W1_mean_size);
        }
        MultEqualsTensor(W1_mean.elements, 1.0/float(CEM_N_ELITE),W1_mean_size);

        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            PlusEqualsTensor(W2_mean.elements,W2.elements + elite[i] * W2_mean_size, W2_mean_size);
        }
        MultEqualsTensor(W2_mean.elements, 1.0/float(CEM_N_ELITE),W2_mean_size);

        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            PlusEqualsTensor(B1_mean.elements,B1.elements + elite[i] * B1_mean_size, B1_mean_size);
        }
        MultEqualsTensor(B1_mean.elements, 1.0/float(CEM_N_ELITE),B1_mean_size);

        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            PlusEqualsTensor(B2_mean.elements,B2.elements + elite[i] * B2_mean_size, B2_mean_size);
        }
        MultEqualsTensor(B2_mean.elements, 1.0/float(CEM_N_ELITE),B2_mean_size);

    
#ifdef DEBUG_TEXT
        printf("fitting stddev\n");
#endif
        // Fit new stddev
        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            StdDevTensor(W1_stddev.elements, W1_mean.elements, W1.elements + elite[i]*W1_stddev_size, W1_stddev_size);
        }
        MultEqualsTensor(W1_stddev.elements, 1.0/float(CEM_N_ELITE-1),W1_stddev_size);
        PlusEqualsTensor(W1_stddev.elements, 0.1/(epoch*.005+1), W1_stddev_size);
        
        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            StdDevTensor(W2_stddev.elements, W2_mean.elements, W2.elements + elite[i]*W2_stddev_size, W2_stddev_size);
        }
        MultEqualsTensor(W2_stddev.elements, 1.0/float(CEM_N_ELITE-1),W2_stddev_size);
        PlusEqualsTensor(W2_stddev.elements, 0.1/(epoch*.005+1), W2_stddev_size);
        
        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            StdDevTensor(B1_stddev.elements, B1_mean.elements, B1.elements + elite[i]*B1_stddev_size, B1_stddev_size);
        }
        MultEqualsTensor(B1_stddev.elements, 1.0/float(CEM_N_ELITE-1),B1_stddev_size);
        PlusEqualsTensor(B1_stddev.elements, 0.1/(epoch*.005+1), B1_stddev_size);
        
        for(int i = 0; i < CEM_N_ELITE; ++i)
        {
            StdDevTensor(B2_stddev.elements, B2_mean.elements, B2.elements + elite[i]*B2_stddev_size, B2_stddev_size);
        }
        MultEqualsTensor(B2_stddev.elements, 1.0/float(CEM_N_ELITE-1),B2_stddev_size);
        PlusEqualsTensor(B2_stddev.elements, 0.1/(epoch*.005+1), B2_stddev_size);
        
    } // end CEM



    printf("saving to disk...\n");
    std::ofstream outfile;
    outfile.open("dd_parameters.m");
    for (int i = 0; i < W1_mean_size; ++i)
        outfile << W1_mean.elements[i] << std::endl;
    for (int i = 0; i < B1_mean_size; ++i)
        outfile << B1_mean.elements[i] << std::endl;
    for (int i = 0; i < W2_mean_size; ++i)
        outfile << W2_mean.elements[i] << std::endl;
    for (int i = 0; i < B2_mean_size; ++i)
        outfile << B2_mean.elements[i] << std::endl;
    outfile.close();
    printf("saved!\n");
      
    FreeDeviceTensor(&X_d);
    FreeDeviceTensor(&X_feat_d);
    FreeDeviceTensor(&U_d);
    FreeDeviceTensor(&X_Prime_d);
    FreeDeviceTensor(&W1_d);
    FreeDeviceTensor(&B1_d);
    FreeDeviceTensor(&Hidden_d);
    FreeDeviceTensor(&W2_d);
    FreeDeviceTensor(&B2_d);
    cudaFree(R_d);
    R_d = NULL;

    
    ////  X_prime = CarDynamics( X, (W2*relu(W1*X + B1) + B2) )
    ////  Rewards += Reward(X,X_prime)
    ////  X = X_prime

    

    return 0;
    
}







//Tensor CEM(int blank, Tensor & theta_mean, int batch_size, int n_iter, float elite_frac, float initial_std=0.1f)
//{
//    // std::ofstream outfile;
//    // outfile.open("new_results.csv");
//    int n_elite = int(roundf(elite_frac * batch_size));
//    
//    Eigen::VectorXf theta_std = Eigen::VectorXf::Ones(theta_mean.size()) * initial_std; // TODO: Make static
//
//    for( int i = 0; i < n_iter; ++i )
//    {
//        std::vector<Eigen::VectorXf> thetas(batch_size);
//        std::vector<float> rewards(batch_size);
//        //std::vector<unsigned int> elite(batch_size);
//        //std::cout << "thetas: " << thetas.size() << std::endl;
//        //
//        if (true)
//        {
//            #pragma omp parallel for
//            for (int k = 0; k < thetas.size(); ++k)
//            {
//                thetas.at(k) = theta_mean.binaryExpr(theta_std, &AddNoise);
//                rewards.at(k) = EvaluatePolicy(thetas.at(k),TRAJECTORY_SAMPLES,EVALUATION_SAMPLES,DT); // Reward(theta,Eigen::Vector3f::Zero());
//            
//                //++k;
//            }
//        } 
//        else
//        {
//            //int k = 0;
//            for (int k = 0; k < thetas.size(); ++k)
//            {
//                thetas.at(k) = theta_mean.binaryExpr(theta_std, &AddNoise);
//                rewards.at(k) = EvaluatePolicy(thetas.at(k),TRAJECTORY_SAMPLES,EVALUATION_SAMPLES,DT); // Reward(theta,Eigen::Vector3f::Zero());
//                //++k;
//            }
//        }
//
//        //for (Eigen::VectorXf & theta : thetas)
//        //{
//        //    theta = theta_mean.binaryExpr(theta_std, &AddNoise);
//        //    rewards.at(k) = EvaluatePolicy(theta,TRAJECTORY_SAMPLES,30,DT); // Reward(theta,Eigen::Vector3f::Zero());
//        //    
//        //    ++k;
//        //}
//
//        std::vector<unsigned int> elite = sort_indexes(rewards);
//        std::cout << i+1 << "/" << n_iter << ", Best reward: " << rewards[elite[0]] << std::endl;
//        outfile << rewards[elite[0]] << std::endl;
//        Eigen::VectorXf new_mean = Eigen::VectorXf::Zero(theta_mean.size());
//        Eigen::VectorXf new_std = Eigen::VectorXf::Zero(theta_mean.size());
//        for(int e = 0; e < n_elite; ++e)
//        {
//            new_mean += thetas[elite[e]];
//        }
//        new_mean *= 1.0f/n_elite;
//
//
//        // Assuming statistical independence between each variable
//        for(int j = 0; j < new_mean.size(); ++j)
//        {
//            for(int e = 0; e < n_elite; ++e)
//            {
//                new_std[j] += powf((new_mean[j]-thetas[elite[e]][j]),2.0);
//            }
//            new_std[j] /= (n_elite-1); // TODO: maybe add a little bit to avoid vanishing std_dev (.1/(i+1))
//            new_std[j] += (.1/(i*.005f+1));
//        }
//
//
//        // Set up new distribution
//        theta_mean = new_mean;
//        theta_std = new_std;
//        
//        
//
//        /*for(int e = 0; e < n_elite; ++e)
//        {
//            Eigen::VectorXf displacement = thetas[elite[e]] - new_mean;
//            new_std += displacement.squaredNorm();
//        }*/
//    }
//    
//    //outfile.close();
//    //std::cout << th_std.size() << std::endl;
//    ////Add noise to batch_size samples 
//    //ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
//    ////Evaluate each sample
//    //ys = np.array([f(th,evalation_samples) for th in ths])
//    //// Keep top n_elite best samples
//    //elite_inds = ys.argsort()[::-1][:n_elite]
//    //elite_ths = ths[elite_inds]
//    //// Compute the mean and std-dev of best samples
//    //th_mean = np.median(elite_ths,axis=0)
//    //th_std = elite_ths.std(axis=0)
//    ////  some extra noise
//    //th_std += cem_noise_factor/(iter+1)
//    ////Return results 
//    return theta_mean;
//}









// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
