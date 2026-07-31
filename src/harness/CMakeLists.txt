# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# Build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(PYTHON3_CMD "python3" CACHE STRING "Python 3 command")

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})


# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Set sm versions
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.4)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_87,code=sm_87")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.8)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_90,code=sm_90")
endif()

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Set the path to the Triton library
unset(TRITON_DIR CACHE)
set(TRITON_DIR ../../build/triton-inference-server)

# Set NVTX library path
#unset(NV_TOOLS_EXT_LIB CACHE)
#set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Build the harness for the Triton harness and for DLRM if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)


######### FFI Utilities for Python ###########
pybind11_add_module(FFIUtils ffi_utils/ffi_utils.cpp)
target_link_libraries(FFIUtils PRIVATE ${LOADGEN_LIB})
target_include_directories(FFIUtils
    PUBLIC
    ${LOADGEN_INCLUDE_DIR})
set_target_properties(FFIUtils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${FFI_UTILS_DIR})


######### DEFAULT HARNESS ########
if (${SOC_SM})
    if (${SOC_SM} STREQUAL "87")
        add_compile_definitions(IS_ORIN)
    endif()
endif()

# Add the LWIS subdirectory (which will generate a static LWIS library)
add_subdirectory(lwis)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cc
    common/logger.cpp
)

target_link_libraries(harness_default
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    lwis
    ${LOADGEN_LIB}
    numa
)

target_include_directories(harness_default
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
)

######### DEFAULT PY_HARNESS ########
add_custom_target(py_harness_default ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building default py_harness..."
    COMMAND ${CMAKE_COMMAND} -E env CXX_FLAGS=${CMAKE_CXX_FLAGS}
            ${PYTHON3_CMD} setup.py build_ext
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/harness_default
)

add_dependencies(py_harness_default lwis)

# Remove the build directory after building py_harness_default
add_custom_command(TARGET py_harness_default POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/harness_default/build
)



######### DLRMv2 HARNESS ########
if (${IS_SOC})
    execute_process(COMMAND echo "Skipping DLRMv2 harness for SoC system")
else()
    execute_process(COMMAND echo "Building DLRMv2 harness...")
    add_executable(harness_dlrm_v2
            harness_dlrm_v2/main_dlrm_v2.cpp
            harness_dlrm_v2/dlrm_v2_server.cpp
            harness_dlrm_v2/batch_maker.cpp
            harness_dlrm_v2/dlrm_v2_kernels.cu
            common/logger.cpp
    )

    target_link_libraries(harness_dlrm_v2
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
            ${NV_TOOLS_EXT_LIB}
            numa
    )

    target_include_directories(harness_dlrm_v2
            PUBLIC
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                common
                harness_dlrm_v2
    )
endif()
