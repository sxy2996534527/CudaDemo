#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaGraphTest {
public:
	int TestCreateGraphWithStreamCapture();
	int TestCrossStreamDependencyAndEvent();
	int TestUpdateGlobalGraph();

private:
	cudaError_t CreateGraphWithAPI();
	cudaError_t CreateGraphWithStreamCapture();
	cudaError_t CrossStreamDependencyAndEvent();
	cudaError_t UpdateGlobalGraph();
};