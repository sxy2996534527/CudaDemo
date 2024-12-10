#include "CudaGraphTest.cuh"

#include <stdio.h>

__global__ void AddKernel1(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void AddKernel2(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void AddKernel3(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void AddKernel4(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

cudaError_t CudaGraphTest::CreateGraphWithAPI() {
	cudaError_t cuda_status;

	// Create the graph - it starts out empty
	cudaGraph_t graph;
	cudaGraphCreate(&graph, 0);

	// For the purpose of this example, we'll create
	// the nodes separately from the dependencies to
	// demonstrate that it can be done in two stages.
	// Note that dependencies can also be specified 
	// at node creation. 
	cudaGraphNode_t a, b, c, d;
	cudaKernelNodeParams nodeParams;
	cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
	cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
	cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
	cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

	// Now set up dependencies on each node
	cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
	cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
	cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
	cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "CreateGraphWithAPI failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}
Error:
	return cuda_status;
}

cudaError_t CudaGraphTest::CreateGraphWithStreamCapture() {
	cudaError_t cuda_status;

	//test data
	int size = 512;
	int data_size = 512 * sizeof(int);
	int* a = new int[size]();
	int* b = new int[size]();
	int* c = new int[size]();

	for (int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	int* d_a = 0;
	int* d_b = 0;
	int* d_c = 0;
	cudaMalloc(&d_a, data_size);
	cudaMalloc(&d_b, data_size);
	cudaMalloc(&d_c, data_size);

	cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, data_size, cudaMemcpyHostToDevice);

	cudaStream_t stream;
	cuda_status = cudaStreamCreate(&stream);

	cudaGraph_t graph;

	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

	AddKernel1 << < 1, size, 0, stream >> > (d_c, d_a, d_b);
	AddKernel2 << < 1, size, 0, stream >> > (d_c, d_a, d_b);
	//libraryCall(stream);
	AddKernel3 << < 1, size, 0, stream >> > (d_c, d_a, d_b);

	cudaStreamEndCapture(stream, &graph);
	
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "CreateGraphWithAPI failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}

	// Instantiate and launch the graph
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
	cudaGraphLaunch(graphExec, stream);

	// Wait for completion
	cudaStreamSynchronize(stream);

	// Copy output vector from GPU buffer to host memory.
	cuda_status = cudaMemcpy(c, d_c, data_size, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	// Clean up
	cudaGraphExecDestroy(graphExec);
	cudaGraphDestroy(graph);
	cudaStreamDestroy(stream);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] a;
	delete[] b;
	delete[] c;

	return cuda_status;
}

cudaError_t CudaGraphTest::CrossStreamDependencyAndEvent() {
	cudaError_t cuda_status;

	//test data
	int size = 512;
	int data_size = 512 * sizeof(int);
	int* a = new int[size]();
	int* b = new int[size]();
	int* c = new int[size]();

	for (int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	int* d_a = 0;
	int* d_b = 0;
	int* d_c = 0;
	cudaMalloc(&d_a, data_size);
	cudaMalloc(&d_b, data_size);
	cudaMalloc(&d_c, data_size);

	cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, data_size, cudaMemcpyHostToDevice);

	cudaStream_t stream1, stream2;
	cuda_status = cudaStreamCreate(&stream1);
	cuda_status = cudaStreamCreate(&stream2);

	// stream1 is the origin stream
	cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

	AddKernel1 << < 1, size, 0, stream1 >> > (d_c, d_a, d_b);

	// Fork into stream2
	cudaEvent_t event1;
	cudaEventCreate(&event1);
	cudaEventRecord(event1, stream1);
	cudaStreamWaitEvent(stream2, event1);

	AddKernel2 << < 1, size, 0, stream1 >> > (d_c, d_a, d_b);
	AddKernel3 << < 1, size, 0, stream2 >> > (d_c, d_a, d_b);

	// Join stream2 back to origin stream (stream1)
	cudaEvent_t event2;
	cudaEventCreate(&event2);
	cudaEventRecord(event2, stream2);
	cudaStreamWaitEvent(stream1, event2);

	AddKernel4 << < 1, size, 0, stream1 >> > (d_c, d_a, d_b);

	// End capture in the origin stream
	cudaGraph_t graph;
	cudaStreamEndCapture(stream1, &graph);

	// stream1 and stream2 no longer in capture mode  

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "CrossStreamDependencyAndEvent failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}

	// Instantiate and launch the graph
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
	cudaGraphLaunch(graphExec, stream1);

	// Wait for completion
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	// Copy output vector from GPU buffer to host memory.
	cuda_status = cudaMemcpy(c, d_c, data_size, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	// Clean up
	cudaGraphExecDestroy(graphExec);
	cudaGraphDestroy(graph);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] a;
	delete[] b;
	delete[] c;

	return cuda_status;
}

cudaError_t CudaGraphTest::UpdateGlobalGraph() {
	cudaError_t cuda_status;

	cudaGraphExec_t graphExec = NULL;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//test data
	int size = 512;
	int data_size = 512 * sizeof(int);
	int* a = new int[size]();
	int* b = new int[size]();
	int* c = new int[size]();

	for (int i = 0; i < size; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	int* d_a = 0;
	int* d_b = 0;
	int* d_c = 0;
	cudaMalloc(&d_a, data_size);
	cudaMalloc(&d_b, data_size);
	cudaMalloc(&d_c, data_size);

	cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, data_size, cudaMemcpyHostToDevice);

	for (int i = 0; i < 10; i++) {
		cudaGraph_t graph;
		cudaGraphExecUpdateResult updateResult;
		cudaGraphNode_t errorNode;

		// In this example we use stream capture to create the graph.
		// You can also use the Graph API to produce a graph.
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

		// Call a user-defined, stream based workload, for example
		AddKernel1 << < 1, size, 0, stream >> > (d_c, d_a, d_b);
		AddKernel2 << < 1, size, 0, stream >> > (d_c, d_a, d_b);
		//libraryCall(stream);
		AddKernel3 << < 1, size, 0, stream >> > (d_c, d_a, d_b);

		cudaStreamEndCapture(stream, &graph);

		// If we've already instantiated the graph, try to update it directly
		// and avoid the instantiation overhead
		if (graphExec != NULL) {
			// If the graph fails to update, errorNode will be set to the
			// node causing the failure and updateResult will be set to a
			// reason code.
			cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
		}

		// Instantiate during the first iteration or whenever the update
		// fails for any reason
		if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

			// If a previous update failed, destroy the cudaGraphExec_t
			// before re-instantiating it
			if (graphExec != NULL) {
				cudaGraphExecDestroy(graphExec);
			}
			// Instantiate graphExec from graph. The error node and
			// error message parameters are unused here.
			cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
		}

		cudaGraphDestroy(graph);
		cudaGraphLaunch(graphExec, stream);
		cudaStreamSynchronize(stream);
	}
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "UpdateGlobalGraph failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cuda_status = cudaMemcpy(c, d_c, data_size, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	// Clean up
	cudaGraphExecDestroy(graphExec);
	//cudaGraphDestroy(graph);
	cudaStreamDestroy(stream);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] a;
	delete[] b;
	delete[] c;
	return cuda_status;
}

int CudaGraphTest::TestCreateGraphWithStreamCapture() {
	cudaError_t cuda_status = CreateGraphWithStreamCapture();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "CreateGraphWithStreamCapture failed: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	return 0;
}

int CudaGraphTest::TestCrossStreamDependencyAndEvent() {
	cudaError_t cuda_status = CrossStreamDependencyAndEvent();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "CrossStreamDependencyAndEvent failed: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	return 0;
}


int CudaGraphTest::TestUpdateGlobalGraph() {
	cudaError_t cuda_status = UpdateGlobalGraph();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "UpdateGlobalGraph failed: %s\n", cudaGetErrorString(cuda_status));
		return 1;
	}

	return 0;
}