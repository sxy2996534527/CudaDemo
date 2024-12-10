#include "CudaUserObject.cuh"

//// Library API with pool allocation
//void libraryWork(cudaStream_t stream) {
//    auto& resource = pool.claimTemporaryResource();
//    resource.waitOnReadyEventInStream(stream);
//    launchWork(stream, resource);
//    resource.recordReadyEvent(stream);
//}
//// Library API with asynchronous resource deletion
//void libraryWork(cudaStream_t stream) {
//    Resource* resource = new Resource(...);
//    launchWork(stream, resource);
//    cudaStreamAddCallback(
//        stream,
//        [](cudaStream_t, cudaError_t, void* resource) {
//            delete static_cast<Resource*>(resource);
//        },
//        resource,
//            0);
//    // Error handling considerations not shown
//}

int CudaUserObject::ManageGraph() {
    //cudaGraph_t graph;  // Preexisting graph

    //Object* object = new Object;  // C++ object with possibly nontrivial destructor
    //cudaUserObject_t cuObject;
    //cudaUserObjectCreate(
    //    &cuObject,
    //    object,  // Here we use a CUDA-provided template wrapper for this API,
    //             // which supplies a callback to delete the C++ object pointer
    //    1,  // Initial refcount
    //    cudaUserObjectNoDestructorSync  // Acknowledge that the callback cannot be
    //                                    // waited on via CUDA
    //);
    //cudaGraphRetainUserObject(
    //    graph,
    //    cuObject,
    //    1,  // Number of references
    //    cudaGraphUserObjectMove  // Transfer a reference owned by the caller (do
    //                             // not modify the total reference count)
    //);
    //// No more references owned by this thread; no need to call release API
    //cudaGraphExec_t graphExec;
    //cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // Will retain a
    //                                                               // new reference
    //cudaGraphDestroy(graph);  // graphExec still owns a reference
    //cudaGraphLaunch(graphExec, 0);  // Async launch has access to the user objects
    //cudaGraphExecDestroy(graphExec);  // Launch is not synchronized; the release
    //                                  // will be deferred if needed
    //cudaStreamSynchronize(0);  // After the launch is synchronized, the remaining
    //                           // reference is released and the destructor will
    //                           // execute. Note this happens asynchronously.
    //// If the destructor callback had signaled a synchronization object, it would
    //// be safe to wait on it at this point.
    return 0;
}