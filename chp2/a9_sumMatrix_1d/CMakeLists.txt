cmake_minimum_required(VERSION 3.10)

project(a9_sumMatrix_1d)

# 设置CUDA版本
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)

# 找CUDA
find_package(CUDA REQUIRED)

# 添加*.cu源文件
file(GLOB CU main.cu)

# 添加包含目录
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
CUDA_ADD_EXECUTABLE(a9_sumMatrix_1d main.cu)

# 顺序是先创建c项目，然后改cmake，更新cmake，然后改main的名字
# nvprof --unified-memory-profiling off ./main