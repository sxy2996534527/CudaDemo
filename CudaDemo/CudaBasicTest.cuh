#pragma once

class CudaBasicTest {
public:
	int TestIntAdd();
	int TestFloatAdd();
	int TestProcess2DArray();
	cudaError_t GetDeviceInfo();
};