### 将tensorflow训练的模型改写成tfcc代码。
下面我们来将[tensorflow官方mnist例子](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py)改写成tfcc代码。

##### 模型数据导出
首先，我们先将模型frozen成`mnist.pb`。
然后运行
```
python tools/datatransfer.py --checkpoint=mnist.pb --out=mnist
```
来生成`mnist.npz`。

##### 改写代码
接着改写python函数`inference`
```
tfcc::Variable<float> inference(const tfcc::Tensor<float>& images, unsigned hidden1_units, unsigned hidden2_units)
{
    tfcc::Variable<float> hidden1, hidden2, logits;
    {
        auto scopeG = tfcc::Scope::scope("hidden1");
        auto& weights = tfcc::Constant<float>::getConstant("weights");
        auto& biases = tfcc::Constant<float>::getConstant("biases");
        hidden1 = tfcc::math::relu(tfcc::blas::matmul(images, weights) + biases);
    }

    {
        auto scopeG = tfcc::Scope::scope("hidden2");
        auto& weights = tfcc::Constant<float>::getConstant("weights");
        auto& biases = tfcc::Constant<float>::getConstant("biases");
        hidden2 = tfcc::math::relu(tfcc::blas::matmul(hidden1, weights) + biases);
    }

    {
        auto scopeG = tfcc::Scope::scope("softmax_linear");
        auto& weights = tfcc::Constant<float>::getConstant("weights");
        auto& biases = tfcc::Constant<float>::getConstant("biases");
        logits = tfcc::math::relu(tfcc::blas::matmul(hidden2, weights) + biases);
    }
    return logits;
}
```
然后加上对应的前处理后处理即可
