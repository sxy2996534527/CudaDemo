#pragma once

//在某些异步函数调用之后检查异步错误的唯一方法是：在调用之后通过调用 cudaDeviceSynchronize()（或异步并发执行中描述的其他任何同步机制）来检查。
//
//运行时为每个初始化为 cudaSuccess 的主机线程维护一个错误变量，并在发生错误时用错误代码覆盖（无论是参数验证错误还是异步错误）。 cudaPeekAtLastError() 会返回此变量。 cudaGetLastError() 返回此变量并将其重置为 cudaSuccess。
//
//内核启动不返回任何错误代码，因此必须在内核启动后立即调用 cudaPeekAtLastError() 或 cudaGetLastError() 以确认任何启动前错误。
//为了确保 cudaPeekAtLastError() 或 cudaGetLastError() 返回的任何错误不是来自内核启动之前的调用，必须确保在内核启动之前将 CUDA 运行时的错误变量设置为 cudaSuccess，例如，可以在内核启动之前调用cudaGetLastError() 。内核启动是异步的，因此要检查异步错误，应用程序必须在内核启动和 cudaPeekAtLastError() 或 cudaGetLastError() 的调用之间进行同步。
//
//请注意，cudaStreamQuery() 和 cudaEventQuery() 可能返回 cudaErrorNotReady ，它不被视为错误，因此 cudaPeekAtLastError() 或 cudaGetLastError() 不会报告。