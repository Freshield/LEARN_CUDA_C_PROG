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

void sumMatrixOnHost2D(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;ib += nx;ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;

    if ((ix < nx) && (iy < ny)){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void initialDataOnGPU(float *A, int nx, int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;

    if ((ix < nx) && (iy < ny)){
        A[idx] = idx;
    }
}

__global__ void sumMatrixOnGPU1D(float *A, float *B, float *C, int nx, int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx){
        for (int iy = 0; iy < ny; ++iy) {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main() {

    int dev = 0;
    cudaSetDevice(dev);

    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    printf("Vector size %d\n", nxy);

    size_t nBytes = nxy * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

//    memset(hostRef, 0, nBytes);
//    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    int dimx = 32;
    int dimy = 1;
    dim3 block(dimx, dimy);
//    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    dim3 grid((nx+block.x-1)/block.x, 1);

    double iStart, iElaps;
    iStart = cpuSecond();
//    initialData(h_A, nxy);
//    initialData(h_B, nxy);
    initialDataOnGPU<<<grid, block>>>(d_A, nx, ny);
    initialDataOnGPU<<<grid, block>>>(d_B, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("initData use %.6f\n", iElaps);

    cudaMemcpy(h_A, d_A, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, nBytes, cudaMemcpyDeviceToHost);

    iStart = cpuSecond();
//    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    sumMatrixOnGPU1D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Execution configuration <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
    printf("GPU sum use %.6f\n", iElaps);


    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    iStart = cpuSecond();
    sumMatrixOnHost2D(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("Host sum use %.3f\n", iElaps);

    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);


    return 0;
}