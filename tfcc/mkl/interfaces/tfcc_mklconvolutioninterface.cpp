// Copyright 2021 Wechat Group, Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tfcc_mklconvolutioninterface.h"

#include <omp.h>
#include <dnnl.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "utils/tfcc_releaseguard.h"

namespace tfcc {

class _MKLOpenMPThreadNumGuard {
  unsigned _threadNum;

 public:
  _MKLOpenMPThreadNumGuard() : _threadNum(omp_get_max_threads()) { omp_set_num_threads(1); }

  ~_MKLOpenMPThreadNumGuard() { omp_set_num_threads(_threadNum); }
};

template <class T>
static inline dnnl::memory::data_type _mkl_get_data_type(T v) {
  throw NotImplementedError();
}

static inline dnnl::memory::data_type _mkl_get_data_type(float v) {
  return dnnl::memory::data_type::f32;
}

static void _mkl_run_net(
    const std::vector<dnnl::primitive>& net,
    const std::vector<std::unordered_map<int, dnnl::memory>>& netArgs, MKLSession* session) {
  for (size_t i = 0; i < net.size(); ++i) {
    net[i].execute(session->getDNNLStream(), netArgs[i]);
  }
  session->getDNNLStream().wait();
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  _MKLOpenMPThreadNumGuard _ompGuard;
  unsigned batch = input.shape(0);
  unsigned outChannels = kernel.shape(0);
  unsigned inChannels = kernel.shape(1);
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  auto dataType = _mkl_get_data_type(static_cast<T>(0));

  // create dims
  dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  dnnl::memory::dims kernelDims = {outChannels, inChannels, kernelHeight, kernelWidth};
  dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  dnnl::memory::dims strides = {strideHeight, strideWidth};
  dnnl::memory::dims paddings = {paddingHeight, paddingWidth};

  // create memory
  dnnl::memory inputMemory(
      {{inputDims},
       dataType,
       (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
      session->getDNNLEngine(), const_cast<T*>(input.data()));
  dnnl::memory kernelMemory(
      {{kernelDims}, dataType, dnnl::memory::format_tag::oihw}, session->getDNNLEngine(),
      const_cast<T*>(kernel.data()));
  dnnl::memory outputMemory(
      {{outputDims},
       dataType,
       (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
      session->getDNNLEngine(), result.data());

  // create memory desc
  dnnl::memory::desc requiredInputDesc({inputDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredKernelDesc({kernelDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, dnnl::memory::format_tag::any);

  // create convolution
  dnnl::convolution_forward::desc convDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, requiredInputDesc,
      requiredKernelDesc, requiredOutputDesc, strides, paddings, paddings);

  mkl_async_wrapper(
      "conv2d",
      [](dnnl::memory inputMemory, dnnl::memory kernelMemory, dnnl::memory outputMemory,
         dnnl::convolution_forward::desc convDesc, MKLSession* session) {
        dnnl::convolution_forward::primitive_desc primitiveDesc(convDesc, session->getDNNLEngine());
        auto realInputMemory = inputMemory;
        if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
          realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
          dnnl::reorder(inputMemory, realInputMemory)
              .execute(
                  session->getDNNLEngine(),
                  {{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
        }
        auto realKernelMemory = kernelMemory;
        if (primitiveDesc.weights_desc() != kernelMemory.get_desc()) {
          realKernelMemory = dnnl::memory(primitiveDesc.weights_desc(), session->getDNNLEngine());
          dnnl::reorder(kernelMemory, realKernelMemory)
              .execute(
                  session->getDNNLEngine(),
                  {{DNNL_ARG_FROM, kernelMemory}, {DNNL_ARG_TO, realKernelMemory}});
        }
        auto realOutputMemory = outputMemory;
        if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
          realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
        }
        dnnl::convolution_forward(primitiveDesc)
            .execute(
                session->getDNNLEngine(), {{DNNL_ARG_SRC, realInputMemory},
                                           {DNNL_ARG_WEIGHTS, realKernelMemory},
                                           {DNNL_ARG_DST, realOutputMemory}});
        dnnl::reorder(realOutputMemory, outputMemory)
            .execute(
                session->getDNNLEngine(),
                {{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
      },
      inputMemory, kernelMemory, outputMemory, convDesc, session);

  return result;
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth) {
  _MKLOpenMPThreadNumGuard _ompGuard;
  unsigned batch = input.shape(0);
  unsigned outChannels = kernel.shape(0);
  unsigned inChannels = kernel.shape(1);
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight =
      (inHeight + 2 * paddingHeight - dilateHeight * (kernelHeight - 1) - 1) / strideHeight + 1;
  unsigned outWidth =
      (inWidth + 2 * paddingWidth - dilateWidth * (kernelWidth - 1) - 1) / strideWidth + 1;

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  auto dataType = _mkl_get_data_type(static_cast<T>(0));
  // create dims
  dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  dnnl::memory::dims kernelDims = {outChannels, inChannels, kernelHeight, kernelWidth};
  dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  dnnl::memory::dims strides = {strideHeight, strideWidth};
  dnnl::memory::dims paddings = {paddingHeight, paddingWidth};
  dnnl::memory::dims dilates = {dilateHeight - 1, dilateWidth - 1};
  // create memory desc
  dnnl::memory::desc requiredInputDesc({inputDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredKernelDesc({kernelDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, dnnl::memory::format_tag::any);
  // create convolution
  dnnl::convolution_forward::desc convDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, requiredInputDesc,
      requiredKernelDesc, requiredOutputDesc, strides, dilates, paddings, paddings);
  mkl_async_wrapper(
      "conv2d",
      [nhwc, inputDims, kernelDims, outputDims, convDesc, dataType](
          const T* input, const T* kernel, T* result, MKLSession* session) {
        // create memory
        dnnl::memory inputMemory(
            {{inputDims},
             dataType,
             (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
            session->getDNNLEngine(), const_cast<T*>(input));
        dnnl::memory kernelMemory(
            {{kernelDims}, dataType, dnnl::memory::format_tag::oihw}, session->getDNNLEngine(),
            const_cast<T*>(kernel));
        dnnl::memory outputMemory(
            {{outputDims},
             dataType,
             (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
            session->getDNNLEngine(), result);
        dnnl::convolution_forward::primitive_desc primitiveDesc(convDesc, session->getDNNLEngine());

        // convolution input if needed.
        auto realInputMemory = inputMemory;
        if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
          realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
          dnnl::reorder(inputMemory, realInputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
        }

        // convolution kernel if needed.
        auto realKernelMemory = kernelMemory;
        if (primitiveDesc.weights_desc() != kernelMemory.get_desc()) {
          realKernelMemory = dnnl::memory(primitiveDesc.weights_desc(), session->getDNNLEngine());
          dnnl::reorder(kernelMemory, realKernelMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, kernelMemory}, {DNNL_ARG_TO, realKernelMemory}});
        }

        auto realOutputMemory = outputMemory;
        if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
          realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
        }

        dnnl::convolution_forward(primitiveDesc)
            .execute(
                session->getDNNLStream(), {{DNNL_ARG_SRC, realInputMemory},
                                           {DNNL_ARG_WEIGHTS, realKernelMemory},
                                           {DNNL_ARG_DST, realOutputMemory}});

        if (realOutputMemory != outputMemory) {
          dnnl::reorder(realOutputMemory, outputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
        }
        session->getDNNLStream().wait();
      },
      input.data(), kernel.data(), result.data(), session);

  return result;
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group) {
  _MKLOpenMPThreadNumGuard _ompGuard;
  unsigned batch = input.shape(0);
  unsigned outChannels = kernel.shape(0);
  unsigned inChannels = kernel.shape(1) * group;
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight =
      (inHeight + 2 * paddingHeight - dilateHeight * (kernelHeight - 1) - 1) / strideHeight + 1;
  unsigned outWidth =
      (inWidth + 2 * paddingWidth - dilateWidth * (kernelWidth - 1) - 1) / strideWidth + 1;

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  auto dataType = _mkl_get_data_type(static_cast<T>(0));
  // create dims
  dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  dnnl::memory::dims kernelDims = {
      group, outChannels / group, inChannels / group, kernelHeight, kernelWidth};
  dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  dnnl::memory::dims strides = {strideHeight, strideWidth};
  dnnl::memory::dims paddings = {paddingHeight, paddingWidth};
  dnnl::memory::dims dilates = {dilateHeight - 1, dilateWidth - 1};
  // create memory desc
  dnnl::memory::desc requiredInputDesc({inputDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredKernelDesc({kernelDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, dnnl::memory::format_tag::any);
  // create convolution
  dnnl::convolution_forward::desc convDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, requiredInputDesc,
      requiredKernelDesc, requiredOutputDesc, strides, dilates, paddings, paddings);
  mkl_async_wrapper(
      "conv2d",
      [nhwc, inputDims, kernelDims, outputDims, convDesc, dataType](
          const T* input, const T* kernel, T* result, MKLSession* session) {
        // create memory
        dnnl::memory inputMemory(
            {{inputDims},
             dataType,
             (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
            session->getDNNLEngine(), const_cast<T*>(input));
        dnnl::memory kernelMemory(
            {{kernelDims}, dataType, dnnl::memory::format_tag::goihw}, session->getDNNLEngine(),
            const_cast<T*>(kernel));
        dnnl::memory outputMemory(
            {{outputDims},
             dataType,
             (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
            session->getDNNLEngine(), result);
        dnnl::convolution_forward::primitive_desc primitiveDesc(convDesc, session->getDNNLEngine());

        // convolution input if needed.
        auto realInputMemory = inputMemory;
        if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
          realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
          dnnl::reorder(inputMemory, realInputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
        }

        // convolution kernel if needed.
        auto realKernelMemory = kernelMemory;
        if (primitiveDesc.weights_desc() != kernelMemory.get_desc()) {
          realKernelMemory = dnnl::memory(primitiveDesc.weights_desc(), session->getDNNLEngine());
          dnnl::reorder(kernelMemory, realKernelMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, kernelMemory}, {DNNL_ARG_TO, realKernelMemory}});
        }

        auto realOutputMemory = outputMemory;
        if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
          realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
        }

        dnnl::convolution_forward(primitiveDesc)
            .execute(
                session->getDNNLStream(), {{DNNL_ARG_SRC, realInputMemory},
                                           {DNNL_ARG_WEIGHTS, realKernelMemory},
                                           {DNNL_ARG_DST, realOutputMemory}});

        if (realOutputMemory != outputMemory) {
          dnnl::reorder(realOutputMemory, outputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
        }
        session->getDNNLStream().wait();
      },
      input.data(), kernel.data(), result.data(), session);

  return result;
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::conv2dBackwardData(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  unsigned batch = input.shape(0);
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned outHeight = (inHeight - 1) * strideHeight + kernelHeight - 2 * paddingHeight;
  unsigned outWidth = (inWidth - 1) * strideWidth + kernelWidth - 2 * paddingWidth;
  unsigned inChannels = kernel.shape(0);
  unsigned outChannels = kernel.shape(1);

  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto dataType = _mkl_get_data_type(static_cast<T>(0));

  std::vector<dnnl::primitive> net;
  std::vector<std::unordered_map<int, dnnl::memory>> netArgs;

  // create dims
  dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  dnnl::memory::dims kernelDims = {outChannels, inChannels, kernelHeight, kernelWidth};
  dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  dnnl::memory::dims strides = {strideHeight, strideWidth};
  dnnl::memory::dims paddings = {paddingHeight, paddingWidth};

  // create memory
  dnnl::memory inputMemory(
      {{inputDims},
       dataType,
       (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
      session->getDNNLEngine(), const_cast<T*>(input.data()));
  dnnl::memory kernelMemory(
      {{kernelDims}, dataType, dnnl::memory::format_tag::iohw}, session->getDNNLEngine(),
      const_cast<T*>(kernel.data()));
  dnnl::memory outputMemory(
      {{outputDims},
       dataType,
       (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw)},
      session->getDNNLEngine(), result.data());

  // create memory desc
  dnnl::memory::desc requiredInputDesc({inputDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredKernelDesc({kernelDims}, dataType, dnnl::memory::format_tag::any);
  dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, dnnl::memory::format_tag::any);

  // create convolution
  dnnl::deconvolution_forward::desc deconvDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, requiredInputDesc,
      requiredKernelDesc, requiredOutputDesc, strides, paddings, paddings);
  dnnl::deconvolution_forward::primitive_desc primitiveDesc(deconvDesc, session->getDNNLEngine());

  // convolution input if needed.
  auto realInputMemory = inputMemory;
  if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
    realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
    net.push_back(dnnl::reorder(inputMemory, realInputMemory));
    netArgs.push_back({{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
  }

  // convolution kernel if needed.
  auto realKernelMemory = kernelMemory;
  if (primitiveDesc.weights_desc() != kernelMemory.get_desc()) {
    realKernelMemory = dnnl::memory(primitiveDesc.weights_desc(), session->getDNNLEngine());
    net.push_back(dnnl::reorder(kernelMemory, realKernelMemory));
    netArgs.push_back({{DNNL_ARG_FROM, kernelMemory}, {DNNL_ARG_TO, realKernelMemory}});
  }

  auto realOutputMemory = outputMemory;
  if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
    realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
  }

  net.push_back(dnnl::deconvolution_forward(primitiveDesc));
  netArgs.push_back(
      {{DNNL_ARG_SRC, realInputMemory},
       {DNNL_ARG_WEIGHTS, realKernelMemory},
       {DNNL_ARG_DST, realOutputMemory}});

  if (realOutputMemory != outputMemory) {
    net.push_back(dnnl::reorder(realOutputMemory, outputMemory));
    netArgs.push_back({{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
  }

  mkl_async_wrapper("conv2d_backward_data", _mkl_run_net, net, netArgs, session);

  return result;
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::maxPool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  _MKLOpenMPThreadNumGuard _ompGuard;

  unsigned batch = input.shape(0);
  unsigned outChannels = nhwc ? input.shape(3) : input.shape(1);
  unsigned inChannels = outChannels;
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;

  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto dataType = _mkl_get_data_type(static_cast<T>(0));

  std::vector<dnnl::primitive> net;
  std::vector<std::unordered_map<int, dnnl::memory>> netArgs;

  // create dims
  const dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  const dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  const dnnl::memory::dims kernels = {kernelHeight, kernelWidth};
  const dnnl::memory::dims strides = {strideHeight, strideWidth};
  const dnnl::memory::dims paddings = {paddingHeight, paddingWidth};

  const dnnl::memory::format_tag memTag =
      (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw);

  // create memory desc
  const dnnl::memory::desc requiredInputDesc({inputDims}, dataType, memTag);
  const dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, memTag);

  // create a pooling
  dnnl::pooling_forward::desc poolDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max, requiredInputDesc,
      requiredOutputDesc, strides, kernels, paddings, paddings);

  mkl_async_wrapper(
      "max_pool2d",
      [memTag, inputDims, outputDims, poolDesc, dataType](
          const T* input, T* output, MKLSession* session) {
        dnnl::pooling_forward::primitive_desc primitiveDesc(poolDesc, session->getDNNLEngine());

        // create memory
        dnnl::memory inputMemory(
            {{inputDims}, dataType, memTag}, session->getDNNLEngine(), const_cast<T*>(input));

        dnnl::memory outputMemory(
            {{outputDims}, dataType, memTag}, session->getDNNLEngine(), output);

        auto realInputMemory = inputMemory;
        if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
          realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
          dnnl::reorder(inputMemory, realInputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
        }

        auto realOutputMemory = outputMemory;
        if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
          realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
        }

        dnnl::pooling_forward(primitiveDesc)
            .execute(
                session->getDNNLStream(),
                {{DNNL_ARG_SRC, realInputMemory}, {DNNL_ARG_DST, realOutputMemory}});

        if (realOutputMemory != outputMemory) {
          dnnl::reorder(realOutputMemory, outputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
        }
      },
      input.data(), result.data(), session);

  return result;
}

template <class T>
Variable<T> MKLConvolutionInterface<T>::avgPool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  _MKLOpenMPThreadNumGuard _ompGuard;

  unsigned batch = input.shape(0);
  unsigned outChannels = nhwc ? input.shape(3) : input.shape(1);
  unsigned inChannels = outChannels;
  unsigned inHeight = nhwc ? input.shape(1) : input.shape(2);
  unsigned inWidth = nhwc ? input.shape(2) : input.shape(3);
  unsigned outHeight = (inHeight - kernelHeight + 2 * paddingHeight) / strideHeight + 1;
  unsigned outWidth = (inWidth - kernelWidth + 2 * paddingWidth) / strideWidth + 1;

  Variable<T> result;
  if (nhwc) {
    result = Variable<T>({batch, outHeight, outWidth, outChannels});
  } else {
    result = Variable<T>({batch, outChannels, outHeight, outWidth});
  }

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto dataType = _mkl_get_data_type(static_cast<T>(0));

  std::vector<dnnl::primitive> net;
  std::vector<std::unordered_map<int, dnnl::memory>> netArgs;

  // create dims
  const dnnl::memory::dims inputDims = {batch, inChannels, inHeight, inWidth};
  const dnnl::memory::dims outputDims = {batch, outChannels, outHeight, outWidth};
  const dnnl::memory::dims kernels = {kernelHeight, kernelWidth};
  const dnnl::memory::dims strides = {strideHeight, strideWidth};
  const dnnl::memory::dims paddings = {paddingHeight, paddingWidth};

  const dnnl::memory::format_tag memTag =
      (nhwc ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::nchw);

  // create memory desc
  const dnnl::memory::desc requiredInputDesc({inputDims}, dataType, memTag);
  const dnnl::memory::desc requiredOutputDesc({outputDims}, dataType, memTag);

  // create a pooling
  dnnl::pooling_forward::desc poolDesc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg, requiredInputDesc,
      requiredOutputDesc, strides, kernels, paddings, paddings);

  mkl_async_wrapper(
      "avg_pool2d",
      [memTag, inputDims, outputDims, poolDesc, dataType](
          const T* input, T* output, MKLSession* session) {
        dnnl::pooling_forward::primitive_desc primitiveDesc(poolDesc, session->getDNNLEngine());

        // create memory
        dnnl::memory inputMemory(
            {{inputDims}, dataType, memTag}, session->getDNNLEngine(), const_cast<T*>(input));

        dnnl::memory outputMemory(
            {{outputDims}, dataType, memTag}, session->getDNNLEngine(), output);

        auto realInputMemory = inputMemory;
        if (primitiveDesc.src_desc() != inputMemory.get_desc()) {
          realInputMemory = dnnl::memory(primitiveDesc.src_desc(), session->getDNNLEngine());
          dnnl::reorder(inputMemory, realInputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, inputMemory}, {DNNL_ARG_TO, realInputMemory}});
        }

        auto realOutputMemory = outputMemory;
        if (primitiveDesc.dst_desc() != outputMemory.get_desc()) {
          realOutputMemory = dnnl::memory(primitiveDesc.dst_desc(), session->getDNNLEngine());
        }

        dnnl::pooling_forward(primitiveDesc)
            .execute(
                session->getDNNLStream(),
                {{DNNL_ARG_SRC, realInputMemory}, {DNNL_ARG_DST, realOutputMemory}});

        if (realOutputMemory != outputMemory) {
          dnnl::reorder(realOutputMemory, outputMemory)
              .execute(
                  session->getDNNLStream(),
                  {{DNNL_ARG_FROM, realOutputMemory}, {DNNL_ARG_TO, outputMemory}});
        }
      },
      input.data(), result.data(), session);

  return result;
}

#define DEFINE_FUNC(type) template class MKLConvolutionInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
