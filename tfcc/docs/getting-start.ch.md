### 最简单的TFCC程序
```
#include <iostream>

#include "tfcc.h"
#ifdef TFCC_WITH_MKL
#include "utils/tfcc_mklutils.h"
#else
#include "utils/tfcc_cudautils.h"
#endif

int main()
{
#ifdef TFCC_WITH_MKL
    tfcc::initialize_mkl();
#else
    tfcc::initialize_cuda();
#endif

    tfcc::Variable<float> a({2, 3});
    tfcc::Variable<float> b({2, 3});

    tfcc::data::set(a, {1, 2, 3, 4, 5, 6});
    tfcc::data::set(b, {2, 3, 4, 5, 6, 7});

    tfcc::Variable<float> c = a + b;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    return 0;
}
```
输出如下：
```
a: float (2,3,) [ 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 ] sum: 21.000000 avg: 3.500000 var: 2.916667
b: float (2,3,) [ 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000 ] sum: 27.000000 avg: 4.500000 var: 2.916667
c: float (2,3,) [ 3.000000 5.000000 7.000000 9.000000 11.000000 13.000000 ] sum: 48.000000 avg: 8.000000 var: 11.666667
```

上面就是一个最简单的tfcc计算矩阵加法的程序了。代码中的`tfcc::initialize_mkl()`和`tfcc::initialize_cuda()`是可以多次调用的，但在进行计算前，本线程必须调用至少一次。如果调用的是`tfcc::initialize_mkl()`则表示本线程内所有计算都在cpu上跑，如果调用的是`tfcc::initialize_cuda()`则表示本线程所有有计算都在gpu上跑。

### 使用常量
在大部分的场景下，我们会有很多程序运行期间其数值不会改变的矩阵，这种情况下每次调用`tfcc::data::set`函数无疑是低效的。要使用常量，我们先要做些准备。
首先，我们需要用python运行如下代码来生成`data.npz`。
```
import numpy as np

b = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float32)
b = b.reshape([2, 3])
data = {'b': b}
np.savez('data', **data)
```
接着我们修改上面的示例代码，将`b`改成常量：
```
#include <iostream>

#include "tfcc.h"
#ifdef TFCC_WITH_MKL
#include "utils/tfcc_mklutils.h"
#else
#include "utils/tfcc_cudautils.h"
#endif

int main(int argc, char* argv[])
{
#ifdef TFCC_WITH_MKL
    tfcc::initialize_mkl();
#else
    tfcc::initialize_cuda();
#endif

    if (argc < 2)
    {   
        std::cout << "Usage: " << argv[0] << " [model_path]" << std::endl;
        return 1;
    }   
    tfcc::NPZDataLoader loader(argv[1]);
    tfcc::DataLoader::setGlobalDefault(&loader);

    tfcc::Variable<float> a({2, 3});
    tfcc::data::set(a, {1, 2, 3, 4, 5, 6});

    tfcc::Constant<float>& b = tfcc::Constant<float>::getConstant("b");

    tfcc::Variable<float> c = a + b;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    return 0;
}
```

这里通过`Constant`类`getConstant`函数来获取了`data.npz`中的名字为`b`的常量。这个函数会在第一次调用的时候进行常量载入的操作，之后的每次调用，都只会从中cache将对应的`Constant`对象返回。

### 命名空间
很多时候，常量会有非常多个，这种时候，就需要命名空间(scope)进行管理。
同样，我们先运行如下python代码：
```
import numpy as np

b = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float32)
b = b.reshape([2, 3])
c = b + 1
d = c + 1
data = {'scope1/b': b, 'scope2/c': c, 'scope1/scope3/d': d}
np.savez('data', **data)
```
接着我们修改上面的示例代码：
```
#include <iostream>

#include "tfcc.h"
#ifdef TFCC_WITH_MKL
#include "utils/tfcc_mklutils.h"
#else
#include "utils/tfcc_cudautils.h"
#endif

int main(int argc, char* argv[])
{
#ifdef TFCC_WITH_MKL
    tfcc::initialize_mkl(1, 1);
#else
    tfcc::initialize_cuda();
#endif

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [model_path]" << std::endl;
        return 1;
    }
    tfcc::NPZDataLoader loader(argv[1]);
    tfcc::DataLoader::setGlobalDefault(&loader);

    tfcc::Variable<float> a({2, 3});
    tfcc::data::set(a, {1, 2, 3, 4, 5, 6});

    tfcc::Variable<float> result;
    {
        auto scope1 = tfcc::Scope::scope("scope1");
        tfcc::Constant<float>& b = tfcc::Constant<float>::getConstant("b");
        result = a + b;
        auto scope3 = tfcc::Scope::scope("scope3");
        tfcc::Constant<float>& d = tfcc::Constant<float>::getConstant("d");
        result = result * d;
    }

    {
        auto scope2 = tfcc::Scope::scope("scope2");
        tfcc::Constant<float>& c = tfcc::Constant<float>::getConstant("c");
        result = result - c;
    }

    std::cout << "a: " << a << std::endl;
    std::cout << "result: " << result << std::endl;
    return 0;
}
```
运行结果如下：
```
a: float (2,3,) [ 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 ] sum: 21.000000 avg: 3.500000 var: 2.916667
result: float (2,3,) [ 9.000000 21.000000 37.000000 57.000000 81.000000 109.000000 ] sum: 314.000000 avg: 52.333333 var: 1191.555556
```
在这里，程序会依次进入`tfcc::Scope::scope`函数所指定的`scope`，并且一旦该函数返回的对象被析构，则会退出该命名空间。
以上示例的完整代码在samples/helloworld里。
