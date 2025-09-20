project(Genie_runtime)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
# Enable -fPIC globally for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime)
include_directories(${ORT_PATH})


FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)


pybind11_add_module(T2SOnnxCPURuntime src/T2SOnnxCPURuntime.cpp)

set(ORT_PATH "/home/eleven/miniconda3/envs/genie/lib/python3.10/site-packages/onnxruntime/capi" CACHE PATH "Path to onnxruntime C API library")
set(ORT_LIBRARY_NAME "libonnxruntime.so.1.22.1" CACHE STRING "Name of the onnxruntime shared library")

message("ORT_PATH: ${ORT_PATH}, ORT_LIBRARY_NAME: ${ORT_LIBRARY_NAME}")
target_link_libraries(T2SOnnxCPURuntime PRIVATE ${ORT_PATH}/${ORT_LIBRARY_NAME})
set_target_properties(T2SOnnxCPURuntime PROPERTIES
    INSTALL_RPATH "${ORT_PATH}"  
    BUILD_WITH_INSTALL_RPATH TRUE  
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..
)

#For pip install -e . to debug
add_custom_command(TARGET T2SOnnxCPURuntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:T2SOnnxCPURuntime> ${CMAKE_CURRENT_LIST_DIR}/..
)

if(UNIX) # Linux specific handling soname since onnxruntime pip package does not provide soft ln to libonnxruntime.so.1
    add_custom_command(
        TARGET T2SOnnxCPURuntime
        POST_BUILD
        COMMAND patchelf --replace-needed libonnxruntime.so.1 libonnxruntime.so.1.22.1 $<TARGET_FILE:T2SOnnxCPURuntime>
        COMMENT "Replacing dependency libonnxruntime.so.1 → libonnxruntime.so.1.22.1"
    )
endif()