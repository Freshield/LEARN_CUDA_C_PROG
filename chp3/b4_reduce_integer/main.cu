#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n){
        return;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0){
            idata[tid] = idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

int main() {
    printf("Hello, World!\n");
    return 0;
}