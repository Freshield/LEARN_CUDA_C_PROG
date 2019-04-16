#include <stdio.h>
#include <cuda_runtime.h>
#include "../../common/common.h"

int recursizeReduce(int *data, int const size){
    if (size == 1){
        return data[0];
    }

    int const stride = size / 2;

    for (int i = 0; i < stride; ++i) {
        data[i] += data[i + stride];
    }

    return recursizeReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n){
        return;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceNeightboredLess(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n){
        return;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x){
            idata[index] += idata[index + stride];
        }

        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }

}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n){
        return;
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if (idx + 3 * blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrollingWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    struct cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, dev);
    printf("starting device %d: %s", dev, iProp.name);
    cudaSetDevice(dev);

    bool bResult = false;
    double iStart, iElaps;

    int size = 1 << 24;
    printf(" with array size %d\n", size);

    int blocksize = 512;

    if (argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x -1) / block.x, 1);
    printf(" grid %d block%d\n", grid.x, block.x);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    for (int i = 0; i < size; ++i) {
        h_idata[i] = (int)(rand() & 0xFF);
    }

    int gpu_sum = 0;
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    memcpy(tmp, h_idata, bytes);
    iStart = seconds();
    int cpu_sum = recursizeReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);
    //gpu kernel 1
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    //gpu kernel 2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeightboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored less elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    //gpu kernel 3
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu neighbored leaved elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    //gpu kernel 4
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/2 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x/2; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu unrolling 2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/2, block.x);
    //gpu kernel 5
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling4<<<grid.x/4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x/4; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu unrolling 4 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/4, block.x);
    //gpu kernel 6
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x/8; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu unrolling 8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);
    //gpu kernel 7
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrollingWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x/8; ++i) {
        gpu_sum += h_odata[i];
    }
    printf("gpu unrolling wraps 8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    bResult = (gpu_sum == cpu_sum);
    if (!bResult){
        printf("Test failed!\n");
    }

    return EXIT_SUCCESS;
}