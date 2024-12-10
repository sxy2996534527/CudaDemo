#include "CudaEvent.cuh"

#include <stdio.h>
#include <iostream>
#include <chrono>

__global__ void SimpleAddKernel1(float* outputDevPtr, float* inputDevPtr, int data_size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < data_size)
		outputDevPtr[i] = inputDevPtr[i] + 1;
}

cudaError_t CudaEvent::EventElapsedTime() {
	cudaError_t cuda_status;

	size_t size = 51200;
	size_t data_size = size * sizeof(float);
	float* inputHost;
	cuda_status = cudaMallocHost(&inputHost, 2 * data_size);

	// Assign values to the pinned memory
	for (size_t i = 0; i < 2 * size; i++) {
		inputHost[i] = static_cast<float>(i * 1.5f);
	}

	float* inputDev;
	float* outputDev;
	cuda_status = cudaMalloc(&inputDev, 2 * data_size);
	cuda_status = cudaMalloc(&outputDev, 2 * data_size);

	cudaStream_t m_stream[2];
	for (int i = 0; i < 2; ++i)
		cudaStreamCreate(&m_stream[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < 2; ++i) {
		cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
			data_size, cudaMemcpyHostToDevice, m_stream[i]);
		SimpleAddKernel1 << <100, 512, 0, m_stream[i] >> >
			(outputDev + i * size, inputDev + i * size, size);
		cudaMemcpyAsync(inputHost + i * size, outputDev + i * size,
			data_size, cudaMemcpyDeviceToHost, m_stream[i]);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Kernel excution time: %fms\n", elapsedTime);

Error:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFreeHost(inputHost);
	cudaFree(inputDev);
	cudaFree(outputDev);
	return cuda_status;
}

int CudaEvent::TestEventElapsedTime() {
	cudaError_t cuda_status = EventElapsedTime();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "EventElapsedTime failed: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	return 0;
}
int CudaEvent::TestCPUElapsedTime() {
	float* inputHost = new float[102400]();
	for (size_t i = 0; i < 102400; i++) {
		inputHost[i] = static_cast<float>(i * 1.5f);
	}
	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 102400; i++) {
		inputHost[i] += 1;
	}

	// Record end time
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate the duration in microseconds
	auto time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

	// Output the time in seconds and microseconds
	std::cout << "Execution time: " << time_used2.count() << " seconds" << std::endl;
	std::cout << "Execution time: " << time_used2.count() * 1000 << " ms" << std::endl;
	
	return 0;
}
