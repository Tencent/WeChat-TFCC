# Introduction
This is a tutorial to convert a onnx model.

# Convert

Use this commend to convert

`cd tfcc/tfcc_code_generator && python3 generator.py --frontend=tensorflow --tf-model-path=${TF_MODEL_PATH} --tf-model-type=${MODEL_TYPE} --backend=runtime --output-path=model.tfccrt --summary=-`

The help of tf generator

`cd tfcc/tfcc_code_generator && python3 generator.py --frontend=tensorflow --backend -h`
# 