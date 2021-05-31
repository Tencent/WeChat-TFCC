

#include "tfcc_cudaconvolutioninterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_cudnnruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudadevice.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_cudatypes.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "utils/tfcc_cudnnutils.h"

namespace tfcc {

/**
 * [s1, s2, s3] => [s1, s3, s2]
 */
template <class T>
static __global__ void _cuda_convolution_transpose(const T* a, unsigned s1, unsigned s2, unsigned s3, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = s1 * s2 * s3;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned ns1 = (i / (s2 * s3)) % s1;
    unsigned ns2 = (i / s3) % s2;
    unsigned ns3 = i % s3;

    unsigned pos = ns1 * s2 * s3 + ns3 * s2 + ns2;
    b[pos] = a[i];
  }
}

template <class T>
CUDAConvolutionInterface<T>::CUDAConvolutionInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAConvolutionInterface<T>::~CUDAConvolutionInterface() {
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc,
    const Tensor<T>& kernel,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth,
    unsigned dilateHeight, unsigned dilateWidth) {
  unsigned batch = input.shape(0);
  unsigned outChannels = kernel.shape(0);
  unsigned inChannels = kernel.shape(1);
  unsigned inHeight = input.shape(nhwc ? 1 : 2);
  unsigned inWidth = input.shape(nhwc ? 2 : 3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;
  cudnnDataType_t dataType = CUDATypeTraits<T>::getCUDNNType();

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  Variable<T> output(nhwc ? Shape({batch, outHeight, outWidth, outChannels}) : Shape({batch, outChannels, outHeight, outWidth}));

  cudnnTensorDescriptor_t inputDescriptor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&inputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnTensorDescriptorGuard inputGuard(&inputDescriptor);

  ret = cudnnSetTensor4dDescriptor(
      inputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      inChannels,
      inHeight,
      inWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // output tensor
  cudnnTensorDescriptor_t outputDescriptor;
  ret = cudnnCreateTensorDescriptor(&outputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnTensorDescriptorGuard outputGuard(&outputDescriptor);

  ret = cudnnSetTensor4dDescriptor(
      outputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      outChannels,
      outHeight,
      outWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // kernel tensor
  cudnnFilterDescriptor_t kernelDescriptor;
  ret = cudnnCreateFilterDescriptor(&kernelDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnFilterDescriptorGuard kernelGuard(&kernelDescriptor);

  ret = cudnnSetFilter4dDescriptor(
      kernelDescriptor,
      dataType,
      CUDNN_TENSOR_NCHW,
      outChannels,
      inChannels,
      kernelHeight,
      kernelWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // conv descriptor
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  ret = cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnConvolutionDescriptorGuard convGuard(&convolutionDescriptor);

  tfcc::CUDADevice* device = static_cast<tfcc::CUDADevice*>(tfcc::Device::getThreadDefault());
#ifdef TFCC_USE_TENSOR_CORE
  if (device->isTensorCoreEnabled()) {
    ret = cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);  //CUDNN_TENSOR_OP_MATH);
    if (ret != CUDNN_STATUS_SUCCESS)
      throw CUDNNRuntimeError(ret);
  }
#endif

  ret = cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingHeight,
      paddingWidth,
      strideHeight,
      strideWidth,
      dilateHeight,
      dilateWidth,
      CUDNN_CROSS_CORRELATION,
      dataType);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // conv algorithm
  cudnnConvolutionFwdAlgo_t convolutionAlgorithm;

  if (device->isTensorCoreEnabled()) {
    convolutionAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  } else {
    ret = cudnnGetConvolutionForwardAlgorithm(
        session->getImpl()->cudnnHandle(),
        inputDescriptor,
        kernelDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        0,
        &convolutionAlgorithm);
    if (ret != CUDNN_STATUS_SUCCESS)
      throw CUDNNRuntimeError(ret);
  }
  // alloc workspace memory
  size_t workspaceBytes = 0;
  ret = cudnnGetConvolutionForwardWorkspaceSize(
      session->getImpl()->cudnnHandle(),
      inputDescriptor,
      kernelDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      convolutionAlgorithm,
      &workspaceBytes);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  unsigned tmpSize = static_cast<unsigned>((workspaceBytes + sizeof(T) - 1) / sizeof(T));
  tmpSize = tmpSize == 0 ? 1 : tmpSize;
  Variable<T> tmp({
      tmpSize,
  });

  // run
  T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
  ret = cudnnConvolutionForward(
      session->getImpl()->cudnnHandle(),
      &alpha,
      inputDescriptor,
      input.data(),
      kernelDescriptor,
      kernel.data(),
      convolutionDescriptor,
      convolutionAlgorithm,
      tmp.data(),
      workspaceBytes,
      &beta,
      outputDescriptor,
      output.data());
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  return output;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc,
    const Tensor<T>& kernel,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth) {
  unsigned batch = input.shape(0);
  unsigned outChannels = kernel.shape(0);
  unsigned inChannels = kernel.shape(1);
  unsigned inHeight = input.shape(nhwc ? 1 : 2);
  unsigned inWidth = input.shape(nhwc ? 2 : 3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;
  cudnnDataType_t dataType = CUDATypeTraits<T>::getCUDNNType();

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  Variable<T> output(nhwc ? Shape({batch, outHeight, outWidth, outChannels}) : Shape({batch, outChannels, outHeight, outWidth}));

  cudnnTensorDescriptor_t inputDescriptor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&inputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnTensorDescriptorGuard inputGuard(&inputDescriptor);

  ret = cudnnSetTensor4dDescriptor(
      inputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      inChannels,
      inHeight,
      inWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // output tensor
  cudnnTensorDescriptor_t outputDescriptor;
  ret = cudnnCreateTensorDescriptor(&outputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnTensorDescriptorGuard outputGuard(&outputDescriptor);

  ret = cudnnSetTensor4dDescriptor(
      outputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      outChannels,
      outHeight,
      outWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // kernel tensor
  cudnnFilterDescriptor_t kernelDescriptor;
  ret = cudnnCreateFilterDescriptor(&kernelDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnFilterDescriptorGuard kernelGuard(&kernelDescriptor);

  ret = cudnnSetFilter4dDescriptor(
      kernelDescriptor,
      dataType,
      CUDNN_TENSOR_NCHW,
      outChannels,
      inChannels,
      kernelHeight,
      kernelWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // conv descriptor
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  ret = cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnConvolutionDescriptorGuard convGuard(&convolutionDescriptor);

  tfcc::CUDADevice* device = static_cast<tfcc::CUDADevice*>(tfcc::Device::getThreadDefault());
#ifdef TFCC_USE_TENSOR_CORE
  if (device->isTensorCoreEnabled()) {
    ret = cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);  //CUDNN_TENSOR_OP_MATH);
    if (ret != CUDNN_STATUS_SUCCESS)
      throw CUDNNRuntimeError(ret);
  }
#endif

  ret = cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingHeight,
      paddingWidth,
      strideHeight,
      strideWidth,
      1,
      1,
      CUDNN_CROSS_CORRELATION,
      dataType);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  // conv algorithm
  cudnnConvolutionFwdAlgo_t convolutionAlgorithm;

  if (device->isTensorCoreEnabled()) {
    convolutionAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  } else {
    ret = cudnnGetConvolutionForwardAlgorithm(
        session->getImpl()->cudnnHandle(),
        inputDescriptor,
        kernelDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        0,
        &convolutionAlgorithm);
    if (ret != CUDNN_STATUS_SUCCESS)
      throw CUDNNRuntimeError(ret);
  }
  // alloc workspace memory
  size_t workspaceBytes = 0;
  ret = cudnnGetConvolutionForwardWorkspaceSize(
      session->getImpl()->cudnnHandle(),
      inputDescriptor,
      kernelDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      convolutionAlgorithm,
      &workspaceBytes);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  unsigned tmpSize = static_cast<unsigned>((workspaceBytes + sizeof(T) - 1) / sizeof(T));
  tmpSize = tmpSize == 0 ? 1 : tmpSize;
  Variable<T> tmp({
      tmpSize,
  });

  // run
  T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
  ret = cudnnConvolutionForward(
      session->getImpl()->cudnnHandle(),
      &alpha,
      inputDescriptor,
      input.data(),
      kernelDescriptor,
      kernel.data(),
      convolutionDescriptor,
      convolutionAlgorithm,
      tmp.data(),
      workspaceBytes,
      &beta,
      outputDescriptor,
      output.data());
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  return output;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::conv2dBackwardData(
    const Tensor<T>& input, bool nhwc,
    const Tensor<T>& kernel,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth) {
  if (!nhwc) {
    return conv2dBackwardDataNCHW(input, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
  }
  Variable<T> realInput = nhwc2nchw(input);
  Variable<T> output = conv2dBackwardDataNCHW(realInput, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
  return nchw2nhwc(output);
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::maxPool2d(
    const Tensor<T>& input, bool nhwc,
    unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth) {
  unsigned batch = input.shape(0);
  unsigned outChannels = nhwc ? input.shape(3) : input.shape(1);
  unsigned inChannels = outChannels;
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;
  cudnnDataType_t dataType = CUDATypeTraits<T>::getCUDNNType();

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  Variable<T> output(nhwc ? Shape({batch, outHeight, outWidth, outChannels}) : Shape({batch, outChannels, outHeight, outWidth}));

  cudnnTensorDescriptor_t inputDescriptor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&inputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  ret = cudnnSetTensor4dDescriptor(
      inputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      inChannels,
      inHeight,
      inWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnTensorDescriptor_t outputDescriptor;
  ret = cudnnCreateTensorDescriptor(&outputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetTensor4dDescriptor(
      outputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      outChannels,
      outHeight,
      outWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnPoolingDescriptor_t poolingDescriptor;
  ret = cudnnCreatePoolingDescriptor(&poolingDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetPooling2dDescriptor(
      poolingDescriptor,
      CUDNN_POOLING_MAX,
      CUDNN_NOT_PROPAGATE_NAN,
      kernelHeight,
      kernelWidth,
      paddingHeight,
      paddingWidth,
      strideHeight,
      strideWidth);

  T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
  ret = cudnnPoolingForward(
      session->getImpl()->cudnnHandle(),
      poolingDescriptor,
      &alpha,
      inputDescriptor,
      input.data(),
      &beta,
      outputDescriptor,
      output.data());
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnDestroyTensorDescriptor(inputDescriptor);
  cudnnDestroyTensorDescriptor(outputDescriptor);
  cudnnDestroyPoolingDescriptor(poolingDescriptor);
  return output;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::avgPool2d(
    const Tensor<T>& input, bool nhwc,
    unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth) {
  unsigned batch = input.shape(0);
  unsigned outChannels = nhwc ? input.shape(3) : input.shape(1);
  unsigned inChannels = outChannels;
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;
  cudnnDataType_t dataType = CUDATypeTraits<T>::getCUDNNType();

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  Variable<T> output(nhwc ? Shape({batch, outHeight, outWidth, outChannels}) : Shape({batch, outChannels, outHeight, outWidth}));

  cudnnTensorDescriptor_t inputDescriptor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&inputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  ret = cudnnSetTensor4dDescriptor(
      inputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      inChannels,
      inHeight,
      inWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnTensorDescriptor_t outputDescriptor;
  ret = cudnnCreateTensorDescriptor(&outputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetTensor4dDescriptor(
      outputDescriptor,
      nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      outChannels,
      outHeight,
      outWidth);

  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnPoolingDescriptor_t poolingDescriptor;
  ret = cudnnCreatePoolingDescriptor(&poolingDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetPooling2dDescriptor(
      poolingDescriptor,
      CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN,
      kernelHeight,
      kernelWidth,
      paddingHeight,
      paddingWidth,
      strideHeight,
      strideWidth);

  T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
  ret = cudnnPoolingForward(
      session->getImpl()->cudnnHandle(),
      poolingDescriptor,
      &alpha,
      inputDescriptor,
      input.data(),
      &beta,
      outputDescriptor,
      output.data());
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnDestroyTensorDescriptor(inputDescriptor);
  cudnnDestroyTensorDescriptor(outputDescriptor);
  cudnnDestroyPoolingDescriptor(poolingDescriptor);
  return output;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::conv2dBackwardDataNCHW(
    const Tensor<T>& input,
    const Tensor<T>& kernel,
    unsigned paddingHeight, unsigned paddingWidth,
    unsigned strideHeight, unsigned strideWidth) {
  unsigned batch = input.shape(0);
  unsigned inHeight = input.shape(2);
  unsigned inWidth = input.shape(3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight = (inHeight - 1) * strideHeight + kernelHeight - 2 * paddingHeight;
  unsigned outWidth = (inWidth - 1) * strideWidth + kernelWidth - 2 * paddingWidth;
  unsigned inChannels = kernel.shape(0);
  unsigned outChannels = kernel.shape(1);
  cudnnDataType_t dataType = CUDATypeTraits<T>::getCUDNNType();

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  Variable<T> output({batch, outChannels, outHeight, outWidth});

  cudnnTensorDescriptor_t outputDescriptor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&outputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetTensor4dDescriptor(
      outputDescriptor,
      CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      outChannels,
      outHeight,
      outWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnTensorDescriptor_t inputDescriptor;
  ret = cudnnCreateTensorDescriptor(&inputDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetTensor4dDescriptor(
      inputDescriptor,
      CUDNN_TENSOR_NCHW,
      dataType,
      batch,
      inChannels,
      inHeight,
      inWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnFilterDescriptor_t kernelDescriptor;
  ret = cudnnCreateFilterDescriptor(&kernelDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  ret = cudnnSetFilter4dDescriptor(
      kernelDescriptor,
      dataType,
      CUDNN_TENSOR_NCHW,
      inChannels,
      outChannels,
      kernelHeight,
      kernelWidth);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnConvolutionDescriptor_t convolutionDescriptor;
  ret = cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  ret = cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingHeight,
      paddingWidth,
      strideHeight,
      strideWidth,
      1,
      1,
      CUDNN_CROSS_CORRELATION,
      dataType);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnConvolutionBwdDataAlgo_t convolutionAlgorithm;
  ret = cudnnGetConvolutionBackwardDataAlgorithm(
      session->getImpl()->cudnnHandle(),
      kernelDescriptor,
      inputDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
      0,
      &convolutionAlgorithm);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  size_t workspaceBytes = 0;
  ret = cudnnGetConvolutionBackwardDataWorkspaceSize(
      session->getImpl()->cudnnHandle(),
      kernelDescriptor,
      inputDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      convolutionAlgorithm,
      &workspaceBytes);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  unsigned tmpSize = static_cast<unsigned>((workspaceBytes + sizeof(T) - 1) / sizeof(T));
  tmpSize = tmpSize == 0 ? 1 : tmpSize;
  Variable<T> tmp({
      tmpSize,
  });

  T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
  ret = cudnnConvolutionBackwardData(
      session->getImpl()->cudnnHandle(),
      &alpha,
      kernelDescriptor,
      kernel.data(),
      inputDescriptor,
      input.data(),
      convolutionDescriptor,
      convolutionAlgorithm,
      tmp.data(),
      workspaceBytes,
      &beta,
      outputDescriptor,
      output.data());
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  cudnnDestroyTensorDescriptor(inputDescriptor);
  cudnnDestroyTensorDescriptor(outputDescriptor);
  cudnnDestroyFilterDescriptor(kernelDescriptor);
  cudnnDestroyConvolutionDescriptor(convolutionDescriptor);

  return output;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::nhwc2nchw(const Tensor<T>& a) {
  Variable<T> result({a.shape(0), a.shape(3), a.shape(1), a.shape(2)});
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(result.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_convolution_transpose<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.shape(0), a.shape(1) * a.shape(2), a.shape(3),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDAConvolutionInterface<T>::nchw2nhwc(const Tensor<T>& a) {
  Variable<T> result({a.shape(0), a.shape(2), a.shape(3), a.shape(1)});
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(result.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_convolution_transpose<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.shape(0), a.shape(1), a.shape(2) * a.shape(3),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

#define DEFINE_FUNC(type) template class CUDAConvolutionInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
