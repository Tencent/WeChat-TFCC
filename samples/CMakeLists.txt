cmake_minimum_required(VERSION 3.14)
project(tfcc_runtime_example)

# c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(tfcc_runtime)

add_executable(tfcc_runtime_example run_model.cpp)
target_link_libraries(tfcc_runtime_example tfcc_runtime)