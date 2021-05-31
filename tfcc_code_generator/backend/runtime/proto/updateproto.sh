#!/bin/bash

SHELL_FOLDER=$(dirname $(readlink -f "$0"))

cd ${SHELL_FOLDER}
rm -rf operations
rm -f *_pb2.py */*_pb2.py
protoc\
    --proto_path ../../../../\
    --python_out=.\
    ../../../../tfcc_runtime/proto/*.proto ../../../../tfcc_runtime/proto/operations/*.proto
mv tfcc_runtime/proto/* .
rmdir tfcc_runtime/proto
rmdir tfcc_runtime
sed -i 's/^from tfcc_runtime.proto import common_pb2/from .. import common_pb2/' operations/*_pb2.py
sed -i 's/^from tfcc_runtime.proto import common_pb2/from . import common_pb2/' model_pb2.py