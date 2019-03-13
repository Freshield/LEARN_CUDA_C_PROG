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

void initialInt(int *ip, int size){

    for (int i = 0; i < size; ++i) {
        ip[i] = i;
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

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            printf("%3d", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate(%d,%d)"
           "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
           blockIdx.y, ix, iy, idx, A[idx]);
}

int main() {

    int dev = 0;
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;

    size_t nBytes = nxy * sizeof(float);

    int *h_A;
    h_A = (int *)malloc(nBytes);

    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_A;
    cudaMalloc((void **)&d_A, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printThreadIndex<<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    free(h_A);

    cudaDeviceReset();

    return 0;
}