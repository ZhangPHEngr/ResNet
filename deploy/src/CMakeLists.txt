cmake_minimum_required(VERSION 3.22.1)
project(resnet)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")

#CUDA 向CUDA的目录
# directories都是只提供目录，具体索引h文件是在代码中声明编译时查找，索引so文件是在link时声明链接时查找
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES} /usr/local/cuda/lib64)

# TRT 指向TensorRT的目录
include_directories(
        /usr/local/TensorRT-8.4.0.6/include/
        /usr/local/TensorRT-8.4.0.6/samples/common/
)
link_directories(/usr/local/TensorRT-8.4.0.6/lib)

# 相关的cpp文件引入
file(GLOB SRC_LIST "/usr/local/TensorRT-8.4.0.6/samples/common/*.cpp")

# 生成可执行文件并link so
add_executable(demo demo.cpp ${SRC_LIST})
target_link_libraries(demo ${CUDA_LIBRARIES} nvcaffe_parser nvinfer nvonnxparser)