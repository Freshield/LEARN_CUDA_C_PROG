#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int iDev = 0;
    struct cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);

    printf("device %d: %s\n", iDev, iProp.name);
    printf("mulprocessors number: %d\n", iProp.multiProcessorCount);
    printf("constant memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
    printf("shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("registers per block: %d\n", iProp.regsPerBlock);
    printf("warp size %d\n", iProp.warpSize);
    printf("threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}