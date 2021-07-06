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

# Quick Start
1. Convert Model

    The script `generator.py` can convert onnx model or tensorflow model to tfcc model. The docs [Convert ONNX Model](https://github.com/Tencent/WeChat-TFCC/blob/master/samples/ConvertONNXModel.md) and [Convert TF Model](https://github.com/Tencent/WeChat-TFCC/blob/master/samples/ConvertTFModel.md) show the details.

2. Load Model

    There is a simple way to load a model as following code:

    ```
    // load tfcc model to a string.
    std::string modelData = load_data_from_file(path);
    tfcc::runtime::Model model(modelData);
    ```

3. Inference

    Finally run the model
    ```
    tfcc::runtime::data::Inputs inputs;
    tfcc::runtime::data::Outputs outputs;

    // set inputs
    auto item = inputs.add_items();
    item->set_name("The input name");
    item->set_dtype(tfcc::runtime::common::FLOAT);
    std::vector<float> data = {1.0, 2.0};
    item->set_data(data.data(), data.size() * sizeof(float));

    model.run(inputs, outputs);
    ```

    [Complete code](https://github.com/Tencent/WeChat-TFCC/blob/master/samples/run_model.cpp)