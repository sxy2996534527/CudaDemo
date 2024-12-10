#include "DeviceInfo.cuh"

#include <stdio.h>

__global__ void DeviceDemoKernel(float* inputDevPtr, int data_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < data_size)
        inputDevPtr[i] = inputDevPtr[i] + 1;
}

cudaError_t DeviceInfo::GetDeviceList() {
    cudaError_t cuda_status;
    int deviceCount;
    cuda_status = cudaGetDeviceCount(&deviceCount);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Enumerate device failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
Error:
    return cuda_status;
}

cudaError_t DeviceInfo::SetDevice() {
    cudaError_t cuda_status;

    size_t size = 1024 * sizeof(float);
    
    cuda_status = cudaSetDevice(0);            // Set device 0 as current
    float* p0;
    cuda_status = cudaMalloc(&p0, size);       // Allocate memory on device 0
    DeviceDemoKernel << <1000, 128 >> > (p0, 1024); // Launch kernel on device 0
    
    cuda_status = cudaSetDevice(1);            // Set device 1 as current
    float* p1;
    cuda_status = cudaMalloc(&p1, size);       // Allocate memory on device 1
    DeviceDemoKernel << <1000, 128 >> > (p1, 1024); // Launch kernel on device 1

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Set different device failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
Error:
    return cuda_status;
}

cudaError_t DeviceInfo::SetStreamOnMultiDevice() {
    cudaError_t cuda_status;

    size_t size = 1024 * sizeof(float);

    cuda_status = cudaSetDevice(0);               // Set device 0 as current
    float* p0;
    cuda_status = cudaMalloc(&p0, size);       // Allocate memory on device 0
    cudaStream_t s0;
    cuda_status = cudaStreamCreate(&s0);          // Create stream s0 on device 0
    DeviceDemoKernel << <1000, 128, 0, s0 >> > (p0, 1024); // Launch kernel on device 0 in s0

    cuda_status = cudaSetDevice(1);               // Set device 1 as current
    float* p1;
    cuda_status = cudaMalloc(&p1, size);       // Allocate memory on device 1
    cudaStream_t s1;
    cuda_status = cudaStreamCreate(&s1);          // Create stream s1 on device 1
    DeviceDemoKernel << <1000, 128, 0, s1 >> > (p1, 1024); // Launch kernel on device 1 in s1

    // This kernel launch will fail:
    DeviceDemoKernel << <1000, 128, 0, s0 >> > (p1, 1024); // Launch kernel on device 1 in s0

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Set stream on multi-device failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
Error:
    return cuda_status;
}

cudaError_t DeviceInfo::P2PMemoryAccess() {
    cudaError_t cuda_status;
    
    cudaSetDevice(0);                   // Set device 0 as current
    float* p0;
    size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);              // Allocate memory on device 0
    DeviceDemoKernel << <1000, 128 >> > (p0, 1024);        // Launch kernel on device 0
    cudaSetDevice(1);                   // Set device 1 as current
    cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                       // with device 0

    // Launch kernel on device 1
    // This kernel launch can access memory on device 0 at address p0
    DeviceDemoKernel << <1000, 128 >> > (p0, 1024);

    cudaDeviceSynchronize();

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "P2PMemoryAccess on multi-device failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
Error:
    return cuda_status;
}

int DeviceInfo::TestGetDeviceList() {
    cudaError_t cuda_status = GetDeviceList();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "GetDeviceList failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}

int DeviceInfo::TestSetDevice() {
    cudaError_t cuda_status = SetDevice();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "SetDevice failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}

int DeviceInfo::TestSetStreamOnMultiDevice() {
    cudaError_t cuda_status = SetStreamOnMultiDevice();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "SetStreamOnMultiDevice failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}