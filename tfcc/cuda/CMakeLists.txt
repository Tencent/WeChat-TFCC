cmake_minimum_required(VERSION 3.14)

# check cuda
find_package(CUDA)
if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "nvcc compiler path")
else()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE FILEPATH "nvcc compiler path")
endif()

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(DIRECTORIES "exceptions" "framework" "interfaces" "utils" "kernel")

set(SRC_FILES "")

foreach(DIR ${DIRECTORIES})
    aux_source_directory(${DIR} SRC_FILES)
    set(DIR_ALL_FILES "")
    FILE(GLOB DIR_ALL_FILES ${DIR}/*)
    source_group(${DIR} ${DIR_ALL_FILES})
endforeach()

file(GLOB  cu  */*.cu)

add_definitions(-DWITH_CUDA_HEADER)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-warn-spills -Xptxas=-warn-lmem-usage -Xcudafe=--diag_suppress=unsigned_compare_with_zero")

# set cuda arch
if(${CMAKE_VERSION} VERSION_LESS 3.18)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_35,code=sm_35")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_53,code=sm_53")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=compute_62")
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
    endif()
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=compute_75")
    endif()
endif()
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62)
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 9.0)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 70)
    endif()
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 72 75)
    endif()
endif()
if (${CUDA_VERSION} VERSION_GREATER_EQUAL 10.0)
    add_definitions(-DTFCC_USE_TENSOR_CORE)
endif()

enable_language(CUDA)
include_directories(./)
add_library(tfcc_cuda ${SRC_FILES})
target_link_libraries(tfcc_cuda tfcc_core)

set(DEP_LIBRARYS "cublas" "cudnn" "culibos" "cudart" "cufft")
foreach(LIBRARY_NAME ${DEP_LIBRARYS})
    find_library(LIBRARY_PATH ${LIBRARY_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    if ("${LIBRARY_PATH}" MATCHES ".*NOTFOUND")
        message(SEND_ERROR "Library ${LIBRARY_NAME} not found.")
    endif()
    target_link_libraries(tfcc_cuda ${LIBRARY_PATH})
    unset(LIBRARY_PATH CACHE)
    unset(LIBRARY_PATH)
endforeach()

target_include_directories(
    tfcc_cuda
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)

# install
file(GLOB ROOT_HEADERS *.h)
set(HEADER_DIRECTORIES "exceptions" "framework" "utils")

install(TARGETS tfcc_cuda EXPORT ${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRECTORIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include PATTERN *.cpp EXCLUDE PATTERN *.cu EXCLUDE PATTERN kernel EXCLUDE)
install(FILES ${ROOT_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
