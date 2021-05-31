cmake_minimum_required(VERSION 3.14)
project(tfcc_runtime)

# set cmake find path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include(Utils)

# c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# cflag
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-noexcept-type")
    endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TFCC_EXTRA_CXX_FLAGS}")

message("* Current build type is : ${CMAKE_BUILD_TYPE}")
find_package(tfcc)

# srcs
search_srcs(
    SRC_FILES
    "framework"
    "exceptions"
    "operations"
    "utils"
    "operations/base"
    "operations/math"
    "operations/nn"
    "operations/random"
    "operations/relation"
    "operations/fusion"
    "operations/rnn"
    "operations/signal"
)

get_filename_component(TFCC_ROOT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${TFCC_ROOT_DIR} tfcc_runtime)

add_library(tfcc_runtime ${SRC_FILES} ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(
    tfcc_runtime
    PUBLIC
    $<BUILD_INTERFACE:${TFCC_ROOT_DIR}>
    $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/protos>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
    $<INSTALL_INTERFACE:${Protobuf_INCLUDE_DIRS}>
)

target_link_libraries(tfcc_runtime tfcc_core -ldl)
if (${TFCC_WITH_MKL})
    target_link_libraries(tfcc_runtime tfcc_mkl)
endif()

if()
    target_link_libraries(tfcc_runtime tfcc_cuda)
endif()

target_link_libraries(tfcc_runtime ${Protobuf_LIBRARIES})

add_executable(tfcc_runtime_test main.cpp)
target_link_libraries(tfcc_runtime_test tfcc_runtime)

# install
# file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/framework/*.h)
set(HEADER_DIRECTORIES "framework" "utils")
set(PROTO_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/protos/tfcc_runtime/proto)

install(TARGETS tfcc_runtime EXPORT ${PROJECT_NAME})
install(DIRECTORY ${HEADER_DIRECTORIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tfcc_runtime PATTERN *.cpp EXCLUDE)
install(DIRECTORY "../3rd/xbyak" DESTINATION ${CMAKE_INSTALL_PREFIX}/include/3rd PATTERN *.cpp EXCLUDE)
install(DIRECTORY ${PROTO_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tfcc_runtime PATTERN *.cc EXCLUDE)
install(FILES "tfcc_runtime.h" DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

install(EXPORT ${PROJECT_NAME} DESTINATION cmake)
configure_file(cmake/tfcc_runtime-config.cmake.in ${PROJECT_BINARY_DIR}/tfcc_runtime-config.cmake @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/tfcc_runtime-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)