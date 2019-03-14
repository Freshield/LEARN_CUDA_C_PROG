#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void initialDataGPU(float *ip, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    int size = nx * ny;
    if (idx < size){
        ip[idx] = (float)(idx) - 3000.0f;
    }
}

__global__ void del_fake_shadow(float *ip, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    int size = nx * ny;
    if (idx < size){
        if (ip[idx] < -1000.0f){
            ip[idx] = -1000.0f;
        } else if (ip[idx] > 1000.0f){
            ip[idx] = 1000.0f;
        }
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void printMatrix(float *C, const int nx, const int ny){
    float *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for (int iy = 0; iy < 10; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            printf("%.2f\n", ic[ix]);
        }
        ic += nx;
    }
}


int main() {

    //prepare part
    int dev = 0;
    cudaSetDevice(dev);

    int nx = 512;
    int ny = 512;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *h_A, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);

    int dimx = 1;
    int dimy = 1;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, ny);

    initialDataGPU<<<grid, block>>>(d_A, nx, ny);
    cudaMemcpy(h_A, d_A, nxy, cudaMemcpyDeviceToHost);

    double iStart = cpuSecond();
    del_fake_shadow<<<grid, block>>>(d_A, nx, ny);
    cudaMemcpy(gpuRef, d_A, nxy, cudaMemcpyDeviceToHost);
    double iElapse = cpuSecond() - iStart;
    printf("del fake use %f", iElapse);


    return 0;
}