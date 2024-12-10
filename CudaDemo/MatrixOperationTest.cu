#include "MatrixOperationTest.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
        * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
cudaError_t MatMulGlobal(const Matrix A, const Matrix B, Matrix C)
{
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cuda_status = cudaMalloc(&d_A.elements, size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 1 failed!");
        goto Error;
    }

    cuda_status = cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 1 failed!");
        goto Error;
    }

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cuda_status = cudaMalloc(&d_B.elements, size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 2 failed!");
        goto Error;
    }

    cuda_status = cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 2 failed!");
        goto Error;
    }
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cuda_status = cudaMalloc(&d_C.elements, size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc 3 failed!");
        goto Error;
    }

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    // Read C from device memory
    cuda_status = cudaMemcpy(C.elements, d_C.elements, size,
        cudaMemcpyDeviceToHost);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy 3 failed!");
        goto Error;
    }

Error:
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    return cuda_status;
}


// Get a matrix element
__device__ float GetElement(const SMatrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(SMatrix A, int row, int col,
    float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ SMatrix GetSubMatrix(SMatrix A, int row, int col)
{
    SMatrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
        + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulSharedKernel(SMatrix A, SMatrix B, SMatrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    SMatrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    int iter_num = (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < iter_num; ++m) {

        // Get sub-matrix Asub of A
        SMatrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        SMatrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
cudaError_t MatMulShared(const SMatrix A, const SMatrix B, SMatrix C)
{
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Load A and B to device memory
    SMatrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);
    SMatrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice);

    // Allocate C in device memory
    SMatrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    MatMulSharedKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
        cudaMemcpyDeviceToHost);

Error:
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    return cuda_status;
}

int MatrixOperationTest::TestGlobalMatMul() {
    Matrix A, B, C;
    const int width_1 = 300;
    const int width_2 = 400;
    const int height_1 = 400;
    const int height_2 = 300;
    A.width = width_1;
    A.height = height_1;
    A.elements = new float[width_1 * height_1]();

    B.width = width_2;
    B.height = height_2;
    B.elements = new float[width_2 * height_2]();

    C.width = width_2;
    C.height = height_1;
    C.elements = new float[width_2 * height_1]();

    for (int i = 0; i < height_1; i++) {
        for (int j = 0; j < width_1; j++) {
            A.elements[i * width_1 + j] = i + j;
        }
    }
    for (int i = 0; i < height_2; i++) {
        for (int j = 0; j < width_2; j++) {
            B.elements[i * width_2 + j] = i - j;
        }
    }

    // Add vectors in parallel.
    cudaError_t cuda_status = MatMulGlobal(A, B, C);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "MatMul failed!");
        return 1;
    }

    printf("Matrix elements: {%f,%f,%f,%f,%f}\n",
        C.elements[0], C.elements[1], C.elements[2], C.elements[3], C.elements[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

int MatrixOperationTest::TestSharedMatMul() {
    SMatrix A, B, C;
    const int width_1 = 300;
    const int width_2 = 400;
    const int height_1 = 400;
    const int height_2 = 300;
    A.width = width_1;
    A.height = height_1;
    A.stride = width_1;
    A.elements = new float[width_1 * height_1]();

    B.width = width_2;
    B.height = height_2;
    B.stride = width_2;
    B.elements = new float[width_2 * height_2]();

    C.width = width_2;
    C.height = height_1;
    C.stride = width_2;
    C.elements = new float[width_2 * height_1]();

    for (int i = 0; i < height_1; i++) {
        for (int j = 0; j < width_1; j++) {
            A.elements[i * width_1 + j] = i + j;
        }
    }
    for (int i = 0; i < height_2; i++) {
        for (int j = 0; j < width_2; j++) {
            B.elements[i * width_2 + j] = i - j;
        }
    }

    // Add vectors in parallel.
    cudaError_t cuda_status = MatMulShared(A, B, C);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "MatMul failed!");
        return 1;
    }

    printf("Matrix elements: {%f,%f,%f,%f,%f}\n",
        C.elements[0], C.elements[1], C.elements[2], C.elements[3], C.elements[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}