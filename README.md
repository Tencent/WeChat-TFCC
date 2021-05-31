# TFCC

TFCC is a C++ deep learning inference framework.

TFCC provides the following toolkits that faciliate your development and deployment of your trained DL models:

|Porject|Source|Description|
|---|---|---|
|[TFCC](./tfcc/README.md)|`./tfcc`|The core of deep learning inference library. It provides friendly interfaces for model deployment, as well as the implementation of diverse operations in both MKL and CUDA environments. |
|TFCC Code Generator|`./tfcc_code_generator`|An automatic generator that can optimize the structure of your high-level models (tensorflows, pytorch, etc.) and generate the TFCC model.|
|TFCC Runtime|`./tfcc_runtime`|An runtime to load TFCC model and inference.|

# BUILD
Run

`./build.sh ${INSTALL_PREFIX_PATH}`
