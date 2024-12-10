#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class DeviceInfo {
public:
	int TestGetDeviceList();
	int TestSetDevice();
	int TestSetStreamOnMultiDevice();
private:
	cudaError_t GetDeviceList();
	cudaError_t SetDevice();
	cudaError_t SetStreamOnMultiDevice();

	//根据系统属性，特别是 PCIe 或 NVLINK 拓扑结构，设备能够相互寻址对方的内存
	cudaError_t P2PMemoryAccess();
};