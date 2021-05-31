### A simple TFCC program
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
Output:
```
a: float (2,3,) [ 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 ] sum: 21.000000 avg: 3.500000 var: 2.916667
b: float (2,3,) [ 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000 ] sum: 27.000000 avg: 4.500000 var: 2.916667
c: float (2,3,) [ 3.000000 5.000000 7.000000 9.000000 11.000000 13.000000 ] sum: 48.000000 avg: 8.000000 var: 11.666667
```

This is a simple matrix-add program using TFCC. The functions `tfcc::initialize_mkl` and `tfcc::initialize_cuda` must be run once or more before any matrix calculation. Depending on which functions `tfcc::initialize_mkl` and `tfcc::initialize_cuda` are called, TFCC use cpu or gpu to calculate.

### Use constant.
We need some constant-matrix in sometimes. Before using constant-value, we need run python code below to generate data.npz.
```
import numpy as np
b = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float32)
b = b.reshape([2, 3])
data = {'b': b}
np.savez('data', **data)
```
Now we change sample code:
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
We use function `Constant::getConstant` to get a constant value named by b. This function load data when it is called in first time. At other times, the function just return Constant object from cache.

### scope
In many times, we need scope to manage constant-value. Now we run python code below to generate data.npz
```
import numpy as np
b = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float32)
b = b.reshape([2, 3])
c = b + 1
d = c + 1
data = {'scope1/b': b, 'scope2/c': c, 'scope1/scope3/d': d}
np.savez('data', **data)
```
Then we change sample code:
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
Output:
```
a: float (2,3,) [ 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 ] sum: 21.000000 avg: 3.500000 var: 2.916667
result: float (2,3,) [ 9.000000 21.000000 37.000000 57.000000 81.000000 109.000000 ] sum: 314.000000 avg: 52.333333 var: 1191.555556
```
TFCC enter scope by function `tfcc::Scope::scope` specify in order, and once the value return by `tfcc::Scope::scope` has been destroyed, the scope is exited. You can see get the code at samples/helloworld
