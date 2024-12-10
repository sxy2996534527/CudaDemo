#include "StreamTest.cuh"

#include <stdio.h>
#include <iostream>

__global__ void SimpleAddKernel(float* outputDevPtr, float* inputDevPtr, int data_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < data_size)
        outputDevPtr[i] = inputDevPtr[i] + 1;
}

void CUDART_CB MyCallback(void* data) {
    printf("Inside callback %d\n", (size_t)data);
}

StreamTest::StreamTest() {
    m_data_size = 51200;
}

StreamTest::~StreamTest() {

}

cudaError_t StreamTest::CreateStream() {
    cudaError_t cuda_status;

    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&m_stream[i]);
    //float* hostPtr;
    //cudaMallocHost(&hostPtr, 2 * size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Create stream failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

Error:
    return cuda_status;
}

cudaError_t StreamTest::DestroyStream() {
    cudaError_t cuda_status;

    for (int i = 0; i < 2; ++i)
        cudaStreamDestroy(m_stream[i]);
    //float* hostPtr;
    //cudaMallocHost(&hostPtr, 2 * size);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Destroy stream failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

Error:
    return cuda_status;
}

cudaError_t StreamTest::SetL2PersistingAccessAttr(int stream_id) {
    cudaError_t cuda_status;

    cudaDeviceProp prop;                                                                        // CUDA device properties variable
    cuda_status = cudaGetDeviceProperties(&prop, 0);                                                 // Query GPU properties
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        goto Error;
    }
    
    size_t l2_size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    cuda_status = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetLimit failed!");
        goto Error;
    }

    int num_bytes = 512;
    size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

    //size_t data_size = size * sizeof(float);
    //// Allocate vectors in device memory
    //float* d_A;
    //cuda_status = cudaMalloc(&d_A, data_size);
    //// Copy vectors from host memory to device memory
    //cuda_status = cudaMemcpy(d_A, h_data, data_size, cudaMemcpyHostToDevice);

    cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
    //stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_A);               // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio = 0.6;                                        // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;               // Persistence Property
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;                // Type of access property on cache miss

    cuda_status = cudaStreamSetAttribute(m_stream[stream_id], cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaStreamSetAttribute failed!");
        goto Error;
    }
    //for (int i = 0; i < 10; i++) {
    //    cuda_kernelA << <grid_size, block_size, 0, stream >> > (data1);                                 // This data1 is used by a kernel multiple times
    //}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
    //cuda_kernelB << <grid_size, block_size, 0, stream >> > (data1);                                     // A different kernel in the same stream can also benefit
    //                                                                                            // from the persistence of data1

    //stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
    //cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
    //cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2 

    //cuda_kernelC << <grid_size, block_size, 0, stream >> > (data2);                                     // data2 can now benefit from full L2 in normal mode

Error:
    return cuda_status;
}

cudaError_t StreamTest::AsyncMemoryCopy() {
    cudaError_t cuda_status;

    size_t data_size = m_data_size * sizeof(float);
    cuda_status = cudaMallocHost(&m_host_data, 2 * data_size);

    // Assign values to the pinned memory
    for (size_t i = 0; i < 2 * m_data_size; i++) {
        m_host_data[i] = static_cast<float>(i * 1.5f);
    }


    float* inputDevPtr;
    float* outputDevPtr;
    cuda_status = cudaMalloc(&inputDevPtr, 2 * data_size);
    cuda_status = cudaMalloc(&outputDevPtr, 2 * data_size);

    for (int i = 0; i < 2; ++i) {
        cuda_status = cudaMemcpyAsync(inputDevPtr + i * m_data_size, m_host_data + i * m_data_size,
            data_size, cudaMemcpyHostToDevice, m_stream[i]);
        SimpleAddKernel << <100, 512, 0, m_stream[i] >> >
            (outputDevPtr + i * m_data_size, inputDevPtr + i * m_data_size, m_data_size);
        cuda_status = cudaMemcpyAsync(m_host_data + i * m_data_size, outputDevPtr + i * m_data_size,
            data_size, cudaMemcpyDeviceToHost, m_stream[i]);
    }
    //cuda_status = cudaMemcpyAsync(inputDevPtr, m_host_data,
    //    data_size, cudaMemcpyHostToDevice, m_stream[0]);
    //SimpleAddKernel << <100, 512, 0, m_stream[0] >> >
    //    (outputDevPtr, inputDevPtr, m_data_size);
    //cuda_status = cudaMemcpyAsync(m_host_data, outputDevPtr,
    //    data_size, cudaMemcpyDeviceToHost, m_stream[0]);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Execute kernel failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

Error:
    cudaFreeHost(m_host_data);
    cudaFree(inputDevPtr);
    cudaFree(outputDevPtr);
    return cuda_status;
}

cudaError_t StreamTest::LaunchHostFuncInStream() {
    cudaError_t cuda_status;

    size_t data_size = m_data_size * sizeof(float);

    float** hostPtr = new float* [2];
    for (int i = 0; i < 2; i++) {
        cuda_status = cudaMallocHost(&hostPtr[i], data_size);
    }

    // Assign values to the pinned memory
    for (size_t i = 0; i < m_data_size; i++) {
        hostPtr[0][i] = static_cast<float>(i * 0.5f);
        hostPtr[1][i] = static_cast<float>(i * 1.5f);
    }

    float** devPtrIn = new float*[2];
    float** devPtrOut = new float* [2];
    for (int i = 0; i < 2; i++) {
        cuda_status = cudaMalloc(&devPtrIn[i], data_size);
        cuda_status = cudaMalloc(&devPtrOut[i], data_size);
    }

    for (size_t i = 0; i < 2; ++i) {
        cudaMemcpyAsync(devPtrIn[i], hostPtr[i], data_size, cudaMemcpyHostToDevice, m_stream[i]);
        SimpleAddKernel << <100, 512, 0, m_stream[i] >> > (devPtrOut[i], devPtrIn[i], m_data_size);
        cudaMemcpyAsync(hostPtr[i], devPtrOut[i], data_size, cudaMemcpyDeviceToHost, m_stream[i]);
        cudaLaunchHostFunc(m_stream[i], MyCallback, (void*)i);
    }


    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "LaunchHostFuncInStream: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

Error:
    for (int i = 0; i < 2; i++) {
        cudaFreeHost(hostPtr[i]);
        cudaFree(devPtrIn[i]);
        cudaFree(devPtrOut[i]);
    }

    return cuda_status;
}

cudaError_t StreamTest::CreateStreamWithPriority() {
    cudaError_t cuda_status;
    // get the range of stream priorities for this device
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    // create streams with highest and lowest available priorities
    cudaStreamCreateWithPriority(&m_stream[0], cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&m_stream[1], cudaStreamNonBlocking, priority_low);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "LaunchHostFuncInStream: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
Error:
    //cudaStreamDestroy(m_stream[0]);
    //cudaStreamDestroy(m_stream[1]);
    return cuda_status;
}

int StreamTest::TestSimpleAsyncStream() {
    cudaError_t cuda_status = CreateStream();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CreateStream failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    cuda_status = AsyncMemoryCopy();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "TestSimpleAsyncStream failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}

int StreamTest::CountPinnedMem() {
    size_t stepSize = 100 * 1024 * 1024;  // Start with 100 MB increments
    size_t maxAllocSize = 0;
    float* pinnedMemory = nullptr;

    int flag = 0;
    while (!flag) {
        cudaError_t err = cudaMallocHost((void**)&pinnedMemory, maxAllocSize + stepSize);
        if (err == cudaSuccess) {
            std::cout << "Already allocated pinned memory: " << maxAllocSize / (1024.0 * 1024.0) << " MB" << std::endl;
            maxAllocSize += stepSize;
            cudaFreeHost(pinnedMemory);  // Free the memory after successful allocation
        }
        else {
            std::cout << "Failed to allocate additional pinned memory." << std::endl;
            flag = 1;
        }
    }

    std::cout << "Approximate maximum pinned memory: " << maxAllocSize / (1024.0 * 1024.0) << " MB" << std::endl;

    return 0;
}

int StreamTest::TestLaunchHostFuncInStream() {
    cudaError_t cuda_status = CreateStream();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CreateStream failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    cuda_status = LaunchHostFuncInStream();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "TestLaunchHostFuncInStream failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}

int StreamTest::TestCreateStreamWithPriority() {
    cudaError_t cuda_status = CreateStreamWithPriority();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CreateStreamWithPriority failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    return 0;
}