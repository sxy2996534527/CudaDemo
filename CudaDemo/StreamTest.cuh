#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void SimpleAddKernel(float* outputDevPtr, float* inputDevPtr, int data_size);
void CUDART_CB MyCallback(void* data);

class StreamTest {
public:
	StreamTest();
	~StreamTest();
	
	int TestSimpleAsyncStream(); 
	int CountPinnedMem();
	int TestLaunchHostFuncInStream();
	int TestCreateStreamWithPriority();

private:
	cudaError_t CreateStream();
	cudaError_t DestroyStream();
	cudaError_t SetL2PersistingAccessAttr(int stream_id);
	cudaError_t AsyncMemoryCopy();
	cudaError_t LaunchHostFuncInStream();
	cudaError_t CreateStreamWithPriority();

private:
	cudaStream_t m_stream[2];
	size_t m_data_size;
	float* m_host_data;
};