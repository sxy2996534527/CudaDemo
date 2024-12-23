#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaEvent {
public:
	int TestEventElapsedTime();
	int TestCPUElapsedTime();
private:
	cudaError_t EventElapsedTime();
};