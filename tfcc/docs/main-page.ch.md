# README

### 语言

[English document](index.html)

-------------------------------------

### 介绍
TFCC是一个深度学习前向计算框架。它可以帮助开发者在不了解cuda和mkl的情况下快速的进行开发。使用这个框架，程序可以方便的部署在gpu和cpu机器上。
### 安装

##### 编译`tfcc_core`的准备
1. (可选)安装`boost`(v1.65.0以上)。

##### 编译`tfcc_mkl`
1. 安装`MKL`(v2019.1.053以上)。
2. 安装`DNNL`(v1.1以上)并在编译的时候加上`DNNL_ENABLE_CONCURRENT_EXEC=ON`这个cmake参数。
3. (可选)获取`libsvml.a`和`libirc.a`并将他们保存到同一个目录。

##### 编译`tfcc_test`
1. 安装`gtest`(v1.8.1以上)。

##### Configure
`TFCC`使用cmake进行编译。你可以使用cmake变量来控制编译方式。除了cmake标准参数`CMAKE_INSTALL_PREFIX`和`CMAKE_BUILD_TYPE`，你还可以使用如下选项：

|Option                 | Possible Values (defaults in bold)   | Description
|:---                   | :---                                 | :---
|TFCC_WITH_MKL          | **ON**, OFF                          | 控制是否编译`tfcc_mkl`
|TFCC_WITH_CUDA         | **ON**, OFF                          | 控制是否编译`tfcc_cuda`
|TFCC_WITH_TEST         | **ON**, OFF                          | 控制是否编译`tfcc_test`
|TFCC_MKL_USE_AVX2      | **ON**, OFF                          | 控制是否编译avx2支持
|TFCC_MKL_USE_AVX512    | **ON**, OFF                          | 控制是否编译avx512支持
|TFCC_BUILD_SAMPLES     | ON, **OFF**                          | 控制是否编译示例程序
|TFCC_EXTRA_CXX_FLAGS   | *string*                             | 控制编译C++代码时的额外参数
|TFCC_EXTRA_CUDA_FLAGS  | *string*                             | 控制编译CUDA代码时的额外参数
|VECTORCLASS_HOME       | *path*                               | 帮助cmake定位库
|BOOST_HOME             | *path*                               | 帮助cmake定位库
|CUDA_HOME              | *path*:**/usr/local/cuda**           | 帮助cmake定位库
|MKL_HOME               | *path*                               | 帮助cmake定位库
|SVML_HOME              | *path*                               | 帮助cmake定位库
|GTEST_HOME             | *path*                               | 帮助cmake定位库
|DNNL_HOME              | *path*                               | 帮助cmake定位库

##### 编译
配置cmake并生成makefile:
```
mkdir -p build && cd build && cmake $CMAKE_OPTIONS ..
```
编译:
```
make
```
跑测试用例:
```
make check
```
安装:
```
make install
```

### 快速开始
[第一个tfcc程序](@ref md_docs_getting-start.ch)

[tensorflow模型改写成tfcc程序](@ref md_docs_trans-tf-model.ch)
