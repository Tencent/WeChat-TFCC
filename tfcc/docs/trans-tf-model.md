### Transform tensorflow code to TFCC
Now we transform the tensorflow mnist sample to TFCC code.

##### Export model
First, we need freeze tensorflow model to mnist.pb. Next, we run:
```
python tools/datatransfer.py --checkpoint=mnist.pb --out=mnist
```
to create mnist.npz.

##### Transform code
Then we transform python function inference to tfcc code.
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
