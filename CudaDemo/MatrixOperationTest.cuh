#pragma once

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} SMatrix;

class MatrixOperationTest {
public:
    int TestGlobalMatMul();
    int TestSharedMatMul();

//private:
//    void MatMul(const Matrix A, const Matrix B, Matrix C);
};