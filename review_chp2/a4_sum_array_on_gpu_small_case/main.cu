#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; ++i) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Array do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match){
        printf("Array match.\n\n");
    }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {

    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    double iStart, iElaps;
    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;
    printf("initData use %.6f\n", iElaps);

//    memset(hostRef, 0, nBytes);
//    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1)/block.x);

    iStart = cpuSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    printf("GPU sum use %.6f\n", iElaps);


    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;
    printf("Host sum use %.3f\n", iElaps);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);


    return 0;
}