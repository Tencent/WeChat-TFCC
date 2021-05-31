# Introduction
This is a tutorial to convert a onnx model.


# Convert
1. install onnx with pip3

`pip3 install onnx`

2. get help of tf generator

`cd tfcc/tfcc_code_generator && python3 generator.py --frontend=onnx --backend -h`

3. convert

`cd tfcc/tfcc_code_generator && python3 generator.py --frontend=onnx --onnx-path=bertsquad-8.onnx --backend=runtime --output-path=model.tfccrt --summary=-`
