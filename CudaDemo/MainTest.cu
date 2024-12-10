#include "CudaBasicTest.cuh"
#include "MatrixOperationTest.cuh"
#include "StreamTest.cuh"
#include "CudaGraphTest.cuh"
#include "CudaEvent.cuh"
#include "DeviceInfo.cuh"
#include "TextureSurfaceMemory.h"
#include "CudaUserObject.cuh"
#include "SimpleTest.h"

int main()
{
	//SimpleTest simple_test;
	//simple_test.SimplePointerTest();

	//CudaBasicTest basic_test;
	//basic_test.TestProcess2DArray();
	//basic_test.GetDeviceInfo();

	//MatrixOperationTest matrix_test;
	//matrix_test.TestGlobalMatMul();
	//matrix_test.TestSharedMatMul();

	//StreamTest stream_test;
	//stream_test.TestSimpleAsyncStream();
	//stream_test.CountPinnedMem();
	//stream_test.TestLaunchHostFuncInStream();
	//stream_test.TestCreateStreamWithPriority();

	//CudaGraphTest graph_test;
	//graph_test.TestCreateGraphWithStreamCapture();
	//graph_test.TestCrossStreamDependencyAndEvent();
	//graph_test.TestUpdateGlobalGraph();
	
	//CudaEvent cuda_event;
	//cuda_event.TestEventElapsedTime();
	//cuda_event.TestCPUElapsedTime();

	//DeviceInfo dev_info;
	//dev_info.TestGetDeviceList();
	//dev_info.TestSetDevice();
	//dev_info.TestSetStreamOnMultiDevice();
	
	//TextureSurfaceMemory tex_surf_mem;
	//tex_surf_mem.TestTextureMemory();
	//tex_surf_mem.TestSurfaceMemory();
	
	//CudaUserObject object_cu;
	//object_cu.ManageGraph();

	return 0;
}