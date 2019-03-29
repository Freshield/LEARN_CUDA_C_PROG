#include <stdio.h>
#include <cuda_runtime.h>
#include "../../common/common.h"

__global__ void initial_data(float *ip, int nx, int ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix + iy * nx;

    if (ix < nx && iy < ny){
        ip[idx] = idx;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix + iy * nx;

    if (ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(float *C, const int nx, const int ny){
    float *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            printf("%.2f ", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    printf("starting");

    int dimx, dimy;
    if (argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    } else{
        dimx = 32;
        dimy = 32;
    }



    int dev = 0;
    struct cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, dev);
    printf("Using Device %d: %s\n", dev, iProp.name);
    cudaSetDevice(dev);

    int nx = 16 * 1024;
    int ny = 16 * 1024;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("nx %d, ny %d\n", nx, ny);
    printf("nxy %d\n", nxy);
    printf("nBytes %d\n", nBytes);

    float *gpuRef;
    gpuRef = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    initial_data<<<grid, block>>>(d_A, nx, ny);
    initial_data<<<grid, block>>>(d_B, nx, ny);

    double iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = (seconds() - iStart) * 1000;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, iElaps);


    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

//    printMatrix(gpuRef, nx, ny);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(gpuRef);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}