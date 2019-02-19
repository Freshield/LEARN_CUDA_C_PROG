#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!");
}

int main(void)
{
    printf("Hello World from CPU!");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}