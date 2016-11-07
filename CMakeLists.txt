CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

PROJECT(easydl)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")
find_package(CUDA REQUIRED)

# include directories
INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/include/
	/usr/local/include
	/usr/include
    ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include
	)

# lib directories
LINK_DIRECTORIES(
	${PROJECT_BINARY_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	/usr/local/lib
	/usr/lib
	/usr/lib64
    ${PROJECT_SOURCE_DIR}/third_party/googletest/build/googlemock/gtest
	)

# for link
set(SYSLIB
	pthread
	glog
	gflags
	cublas
	cudart
	curand
    gtest
	)

add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -fPIC -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC")

ADD_SUBDIRECTORY(util)
ADD_SUBDIRECTORY(tensor)
ADD_SUBDIRECTORY(operator)

ENABLE_TESTING()
#ADD_TEST(test_singleton ${PROJECT_BINARY_DIR}/bin/test_singleton)
#ADD_TEST(test_xlog ${PROJECT_BINARY_DIR}/bin/test_xlog)
