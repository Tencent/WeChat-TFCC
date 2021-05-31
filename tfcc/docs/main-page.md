# README

### Language

[中文文档](@ref md_docs_main-page.ch)

-------------------------------------

### Introduction
TFCC is a c++ framework for deep learning inference. It can help developers who do not understand the CUDA and MKL programming to quickly write an efficient numerical computation code. Using this framework, models can be easily deployed both in CPU and NVIDIA GPU without any change.

### Install

##### Prepare `tfcc_core`
1. (Optional) Install `boost`(v1.65.0 or later).

##### Prepare `tfcc_mkl`
1. Install `MKL`(v2019.1.053 or later).
2. Install `DNNL`(v1.1 or later) with `DNNL_ENABLE_CONCURRENT_EXEC=ON` Flags.
3. (Optional) Get `libsvml.a` and `libirc.a` and save them in a same folder.

##### Prepare `tfcc_test`
1. Install gtest(v1.8.1 or later)

##### Configure
`TFCC` uses a CMake-based build system. You can use CMake options to control the build. Along with the standard CMake options such as CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE, you can pass `TFCC` specific options:

|Option                 | Possible Values (defaults in bold)   | Description
|:---                   | :---                                 | :---
|TFCC_WITH_MKL          | **ON**, OFF                          | Controls building the `tfcc_mkl`
|TFCC_WITH_CUDA         | **ON**, OFF                          | Controls building the `tfcc_cuda`
|TFCC_WITH_TEST         | **ON**, OFF                          | Controls building the `tfcc_test`
|TFCC_MKL_USE_AVX2      | **ON**, OFF                          | Controls supporting the avx2 instruction set
|TFCC_MKL_USE_AVX512    | **ON**, OFF                          | Controls supporting the avx512 instruction set
|TFCC_BUILD_SAMPLES     | ON, **OFF**                          | Controls building the samples
|TFCC_EXTRA_CXX_FLAGS   | *string*                             | C++ extra build flags
|TFCC_EXTRA_CUDA_FLAGS  | *string*                             | CUDA extra build flags
|BOOST_HOME             | *path*                               | Help cmake locate the library
|CUDA_HOME              | *path*:**/usr/local/cuda**           | Help cmake locate the library
|MKL_HOME               | *path*                               | Help cmake locate the library
|SVML_HOME              | *path*                               | Help cmake locate the library
|GTEST_HOME             | *path*                               | Help cmake locate the library
|DNNL_HOME              | *path*                               | Help cmake locate the library

##### Build
Configure CMake and create a makefile:
```
mkdir -p build && cd build && cmake $CMAKE_OPTIONS ..
```
Build the application:
```
make
```
Run test:
```
make check
```
Install:
```
make install
```

### Getting start
[A simple TFCC program](@ref md_docs_getting-start)

[Transform tensorflow model](@ref md_docs_trans-tf-model)
