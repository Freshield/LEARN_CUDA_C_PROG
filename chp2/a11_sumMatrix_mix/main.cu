#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)\
{\
    cudaError_t error = call;\
    if (error != cudaSuccess){\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(-10*error);\
    }\
}\

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; ++i) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match){
        printf("Arrays match.\n\n");
    } else{
        printf("Arrays do not match.\n\n");
    }
}

void initialInt(int *ip, int size){
    for (int i = 0; i < size; ++i) {
        ip[i] = i;
    }
}

void initialData(float *ip, const int size){

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

__global__ void initialDataGPU(float *ip, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    int size = nx * ny;
    if (idx < size){
        ip[idx] = (float)(size * idx) / 10.0f;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC,
                                 int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny){
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC,
                                 int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx){
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    if (ix < nx && iy < ny){
        unsigned int idx = iy * nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int dev = 0;
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1<<14;
    int ny = 1<<14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx,ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    int dimx = 256;
    int dimy = 1;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, ny);

    double iStart = cpuSecond();
    initialDataGPU<<<nxy, 1>>>(d_MatA, nx, ny);
    initialDataGPU<<<nxy, 1>>>(d_MatB, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("initial on GPU use %f\n", iElaps);

    iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    iElaps = cpuSecond() - iStart;
    printf("initial use %f\n", iElaps);


    iStart = cpuSecond();
//    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
//    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(h_A, d_MatA, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_MatB, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("run on host use %f\n", iElaps);

    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return (0);

}