#include "CudaBasicTest.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__constant__ float constData[256];
__device__ float devData;
__device__ float* devPointer;

__global__ void AddKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Device code
__global__ void Array2DKernel(float* devPtr,
    size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

__global__ void Add2DArrayKernel(float* d_data, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate the pointer to the current row considering the pitch
        float* row = (float*)((char*)d_data + y * pitch);
        float value = row[x];  // Access the element at (x, y)

        // You can process the value as needed
        // For example, just for demonstration, let's add a constant
        row[x] = value + 1;  // Example modification
    }
}


// Device code
__global__ void Array3DKernel(cudaPitchedPtr devPitchedPtr,
    int width, int height, int depth)
{
    char* devPtr = (char *)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}

cudaError_t CudaBasicTest::GetDeviceInfo() {
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    int device;
    cuda_status = cudaGetDevice(&device);  // Get the current device

    int max_threads_block = 0;
    int num_multiprocessors = 0;
    int max_threads_multiprocessors = 0;
    int max_block_dim_x = 0;
    int max_block_dim_y = 0;
    int max_block_dim_z = 0;
    int max_grid_dim_x = 0;
    int max_grid_dim_y = 0;
    int max_grid_dim_z = 0;
    int max_shared_memory_block = 0;
    cuda_status = cudaDeviceGetAttribute(&max_threads_block, cudaDevAttrMaxThreadsPerBlock, device);
    cuda_status = cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, device);
    cuda_status = cudaDeviceGetAttribute(&max_threads_multiprocessors, cudaDevAttrMaxThreadsPerMultiProcessor, device);

    cuda_status = cudaDeviceGetAttribute(&max_block_dim_x, cudaDevAttrMaxBlockDimX, device);
    cuda_status = cudaDeviceGetAttribute(&max_block_dim_y, cudaDevAttrMaxBlockDimY, device);
    cuda_status = cudaDeviceGetAttribute(&max_block_dim_z, cudaDevAttrMaxBlockDimZ, device);

    cuda_status = cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device);
    cuda_status = cudaDeviceGetAttribute(&max_grid_dim_y, cudaDevAttrMaxGridDimY, device);
    cuda_status = cudaDeviceGetAttribute(&max_grid_dim_z, cudaDevAttrMaxGridDimZ, device);
    cuda_status = cudaDeviceGetAttribute(&max_shared_memory_block, cudaDevAttrMaxSharedMemoryPerBlock, device);

    std::cout << "Max threads per block: " << max_threads_block << std::endl;
    std::cout << "Number of multiprocessors on the device: " << num_multiprocessors << std::endl;
    std::cout << "Maximum number of threads per multiprocessor: " << max_threads_multiprocessors << std::endl;
    std::cout << "Maximum block dimension in the X axis: " << max_block_dim_x << std::endl;
    std::cout << "Maximum block dimension in the Y axis: " << max_block_dim_y << std::endl;
    std::cout << "Maximum block dimension in the Z axis: " << max_block_dim_z << std::endl;

    std::cout << "Maximum grid dimension in the X axis: " << max_grid_dim_x << std::endl;
    std::cout << "Maximum grid dimension in the Y axis: " << max_grid_dim_y << std::endl;
    std::cout << "Maximum grid dimension in the Z axis: " << max_grid_dim_z << std::endl;

    std::cout << "Maximum shared memory available per block in bytes: " << max_shared_memory_block << std::endl;

Error:
    return cuda_status;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t AddWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
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
    AddKernel << <1, size >> > (dev_c, dev_a, dev_b);

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

cudaError_t AddFloatWithCuda(float* C, float* A, float* B, unsigned int size) {
    size_t data_size = size * sizeof(float);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    int device;
    cudaGetDevice(&device);  // Get the current device

    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;

    // Allocate vectors in device memory
    float* d_A;
    cudaStatus = cudaMalloc(&d_A, data_size);
    float* d_B;
    cudaStatus = cudaMalloc(&d_B, data_size);
    float* d_C;
    cudaStatus = cudaMalloc(&d_C, data_size);

    // Copy vectors from host memory to device memory
    cudaStatus = cudaMemcpy(d_A, A, data_size, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_B, B, data_size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (size + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, size);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaStatus = cudaMemcpy(C, d_C, data_size, cudaMemcpyDeviceToHost);

Error:
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return cudaStatus;
}

cudaError_t Process2DArray(float* h_data, int width, int height) {
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Host code
    //int width = 64, height = 64;
    float* dev_ptr;
    size_t pitch;
    cuda_status = cudaMallocPitch(&dev_ptr, &pitch,
        width * sizeof(float), height);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch failed!");
        goto Error;
    }
    size_t s_pitch = width * sizeof(float);
    cuda_status = cudaMemcpy2D(dev_ptr, pitch, h_data, s_pitch, width, height, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2D failed!");
        goto Error;
    }

    //define block size and launch kernel
    dim3 block_dim(32, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    //Array2DKernel << <100, 512 >> > (devPtr, pitch, width, height);
    Add2DArrayKernel << < grid_dim, block_dim >> > (dev_ptr, pitch, width, height);

    // Step 5: Check for kernel launch errors
    cudaDeviceSynchronize();  // Ensure the kernel has finished executing
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cuda_status) << std::endl;
        goto Error;
    }

    // Step 6: Copy the data back from the device to the host
    cudaMemcpy2D(h_data, s_pitch, dev_ptr, pitch, width, height, cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_ptr);
    return cuda_status;
}

cudaError_t Test3DArray() {
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Host code
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float),
        height, depth);
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    Array3DKernel << <100, 512 >> > (devPitchedPtr, width, height, depth);

Error:
    return cuda_status;
}

__global__ void PrintKernel() {

    printf("%f\n", devData);
}

cudaError_t AccessGlobalData() {
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    float data[256];
    cuda_status = cudaMemcpyToSymbol(constData, data, sizeof(data));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "1 cudaMemcpyToSymbol Failed!");
        goto Error;
    }

    cuda_status = cudaMemcpyFromSymbol(data, constData, sizeof(data));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "1 cudaMemcpyFromSymbol Failed!");
        goto Error;
    }

    float value = 3.14f;
    cuda_status = cudaMemcpyToSymbol(devData, &value, sizeof(float));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "2 cudaMemcpyToSymbol Failed!");
        goto Error;
    }
    PrintKernel << <1, 1 >> > ();

    float* ptr;
    (&ptr, 256 * sizeof(float));
    cuda_status = cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "3 cudaMemcpyToSymbol Failed!");
        goto Error;
    }

Error:
    return cuda_status;
}

cudaError_t SetL2PersistingAccessAttr(float* h_data, unsigned int size) {
    cudaError_t cuda_status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStream_t stream;
    cuda_status = cudaStreamCreate(&stream);                                                                  // Create CUDA stream
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed!");
        goto Error;
    }

    cudaDeviceProp prop;                                                                        // CUDA device properties variable
    cudaGetDeviceProperties(&prop, 0);                                                 // Query GPU properties
    size_t l2_size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

    int num_bytes = 512;
    size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

    size_t data_size = size * sizeof(float);
    // Allocate vectors in device memory
    float* d_A;
    cuda_status = cudaMalloc(&d_A, data_size);
    // Copy vectors from host memory to device memory
    cuda_status = cudaMemcpy(d_A, h_data, data_size, cudaMemcpyHostToDevice);

    cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_A);               // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio = 0.6;                                        // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;               // Persistence Property
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;                // Type of access property on cache miss

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

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

int CudaBasicTest::TestIntAdd() {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = AddWithCuda(c, a, b, arraySize);
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

int CudaBasicTest::TestFloatAdd() {
    const int arraySize = 500;
    float a[arraySize] = { 0 };
    float b[arraySize] = { 0 };
    float c[arraySize] = { 0 };

    for (int i = 0; i < arraySize; i++) {
        a[i] = i * 0.5f;
        b[i] = i * 1.5f;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = AddFloatWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AddFloatWithCuda failed!");
        return 1;
    }

    printf("{%f,%f,%f,%f,%f} + {%f,%f,%f,%f,%f} = {%f,%f,%f,%f,%f}\n",
        a[0], a[1], a[2], a[3], a[4],
        b[0], b[1], b[2], b[3], b[4],
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

int CudaBasicTest::TestProcess2DArray() {
    const int width = 500;
    const int height = 200;
    const int arr_size = width * height;
    float a[arr_size] = { 0 };

    for (int i = 0; i < arr_size; i++) {
        a[i] = i * 0.5f;
    }

    // Add vectors in parallel.
    cudaError_t cuda_status = Process2DArray(a, width, height);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Process2DArray failed!");
        return 1;
    }

    printf("{%f,%f,%f,%f,%f}\n",
        a[0], a[1], a[2], a[3], a[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}