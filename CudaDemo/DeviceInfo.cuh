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

	//����ϵͳ���ԣ��ر��� PCIe �� NVLINK ���˽ṹ���豸�ܹ��໥Ѱַ�Է����ڴ�
	cudaError_t P2PMemoryAccess();
};