#!/bin/bash

INSTALL_PREFIX_PATH=$1

ROOT_PATH=`pwd`
mkdir -p .tmp

# download oneDNN
if [ ! -d $INSTALL_PREFIX_PATH ]; then
    mkdir -p $INSTALL_PREFIX_PATH
fi
cd $INSTALL_PREFIX_PATH
mkdir -p oneDNN
cd oneDNN
ONEDNN_INSTALL_PATH=`pwd`

cd $ROOT_PATH
cd .tmp
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git checkout v1.8.1
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$ONEDNN_INSTALL_PATH -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_ENABLE_CONCURRENT_EXEC=ON -DDNNL_BUILD_TESTS=OFF
make -j8 && make install

# download vectorclass
cd $INSTALL_PREFIX_PATH
git clone https://github.com/vectorclass/version1.git vectorclass
cd vectorclass
VECTORCLASS_INSTALL_PATH=`pwd`

# download protobuf
cd $INSTALL_PREFIX_PATH
mkdir -p protobuf
cd protobuf
PROTOBUF_INSTALL_PATH=`pwd`

cd $ROOT_PATH
cd .tmp
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.5.0
mkdir build
cd build
cmake ../cmake -DCMAKE_INSTALL_PREFIX=$PROTOBUF_INSTALL_PATH -Dprotobuf_BUILD_TESTS=OFF
make -j8 && make install

# build tfcc math library
cd $INSTALL_PREFIX_PATH
mkdir -p tfcc
cd tfcc
TFCC_INSTALL_PATH=`pwd`

cd $ROOT_PATH
cd tfcc
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$TFCC_INSTALL_PATH -DDNNL_HOME=$ONEDNN_INSTALL_PATH\
    -DVECTORCLASS_HOME=$VECTORCLASS_INSTALL_PATH -DTFCC_WITH_TEST=OFF -DTFCC_MKL_USE_AVX512=OFF
make -j8 && make install

# build runtime
cd $ROOT_PATH
cd tfcc_runtime
mkdir -p build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$PROTOBUF_INSTALL_PATH;$TFCC_INSTALL_PATH" -DCMAKE_INSTALL_PREFIX=$TFCC_INSTALL_PATH
make -j8 && make install
