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

#include "interfaces/tfcc_mklnormalizationinterface.h"

#include <cmath>

#include "dnnl.hpp"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
static void _mkl_layer_normalize(
    const T* a, const T* gamma, const T* beta, T epsilon, unsigned batchSize, unsigned chunkSize,
    T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < batchSize; ++i) {
    // mean = reduce_mean(a)
    T mean = static_cast<T>(0);
    const T* input = a + i * chunkSize;
    for (unsigned j = 0; j < chunkSize; ++j) {
      mean += input[j];
    }
    mean /= static_cast<T>(chunkSize);

    // variance = reduce_mean((inputs - mean)^2)
    T variance = static_cast<T>(0);
    for (unsigned j = 0; j < chunkSize; ++j) {
      T tmp = input[j] - mean;
      variance += tmp * tmp;
    }
    variance /= static_cast<T>(chunkSize);

    // inv = rsqrt(variance + epsilon)
    T inv = static_cast<T>(1) / std::sqrt(variance + epsilon);

    // x = inv * (*_gamma);
    // result = inputs * x + (*_beta) - mean * x
    T* output = b + i * chunkSize;
    for (unsigned j = 0; j < chunkSize; ++j) {
      T x = inv * gamma[j];
      output[j] = (input[j] - mean) * x + beta[j];
    }
  }
}

template <class T>
static inline dnnl::memory::data_type _mkl_get_data_type(T v) {
  throw NotImplementedError();
}

static inline dnnl::memory::data_type _mkl_get_data_type(float v) {
  return dnnl::memory::data_type::f32;
}

static void _mkl_run_primitive(
    const dnnl::primitive& primitive, const std::unordered_map<int, dnnl::memory>& args,
    MKLSession* session) {
  primitive.execute(session->getDNNLStream(), args);
  session->getDNNLStream().wait();
}

template <class T>
Variable<T> MKLNormalizationInterface<T>::layerNormalize(
    const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
    size_t beginNormAxis) {
  unsigned batchSize = 1;
  unsigned chunkSize = 1;
  for (size_t i = 0; i < beginNormAxis; ++i) {
    batchSize *= a.shape(i);
  }
  for (size_t i = beginNormAxis; i < a.shape().size(); ++i) {
    chunkSize *= a.shape(i);
  }

  Variable<T> result(a.shape());
  mkl_async_wrapper(
      "layer_normalize",
      [](const T* a, const T* gamma, const T* beta, T epsilon, unsigned batchSize,
         unsigned chunkSize,
         T* b) { _mkl_layer_normalize(a, gamma, beta, epsilon, batchSize, chunkSize, b); },
      a.data(), gamma.data(), beta.data(), epsilon, batchSize, chunkSize, result.data());

  return result;
}

template <class T>
Variable<T> MKLNormalizationInterface<T>::localResponseNormalization(
    const Tensor<T>& a, T alpha, T beta, T bias, unsigned size) {
  Variable<T> result(a.shape());

  auto dataType = _mkl_get_data_type(static_cast<T>(0));
  std::vector<dnnl::primitive> net;
  std::vector<std::unordered_map<int, dnnl::memory>> netArgs;

  // create dims
  dnnl::memory::dims inputDims = {a.shape(0), a.shape(1), a.shape(2), 1};
  dnnl::memory::dims outputDims = {a.shape(0), a.shape(1), a.shape(2), 1};

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());

  // create memory
  dnnl::memory inputMemory(
      {{inputDims}, dataType, dnnl::memory::format_tag::nchw}, session->getDNNLEngine(),
      const_cast<T*>(a.data()));
  dnnl::memory outputMemory(
      {{outputDims}, dataType, dnnl::memory::format_tag::nchw}, session->getDNNLEngine(),
      const_cast<T*>(result.data()));

  auto lrnDesc = dnnl::lrn_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::lrn_across_channels,
      inputMemory.get_desc(), size, alpha, beta, bias);

  dnnl::lrn_forward::primitive_desc lrnPrimitiveDesc(lrnDesc, session->getDNNLEngine());

  dnnl::memory workspaceMemory(lrnPrimitiveDesc.workspace_desc(), session->getDNNLEngine());
  // Create the primitive.
  dnnl::lrn_forward lrnPrimitive(lrnPrimitiveDesc);

  std::unordered_map<int, dnnl::memory> lrnArgs;
  lrnArgs.insert({DNNL_ARG_SRC, inputMemory});
  lrnArgs.insert({DNNL_ARG_WORKSPACE, workspaceMemory});
  lrnArgs.insert({DNNL_ARG_DST, outputMemory});

  mkl_async_wrapper(
      "local_response_normalization",
      [](const dnnl::primitive& primitive, const std::unordered_map<int, dnnl::memory>& args,
         MKLSession* session) { _mkl_run_primitive(primitive, args, session); },
      lrnPrimitive, lrnArgs, session);
  return result;
}

template <class T>
static void _mkl_batch_normalization(
    const T* a, const T* scale, const T* offset, const T* mean, const T* var, T epsilon,
    unsigned batchSize, unsigned channelSize, unsigned chunkSize, T* result) {
#pragma omp parallel for
  for (unsigned outChannel = 0; outChannel < channelSize; outChannel++) {
    for (unsigned batch = 0; batch < batchSize; batch++) {
      T inv = (static_cast<T>(1) / std::sqrt(var[outChannel] + epsilon)) * scale[outChannel];
      for (unsigned i = 0; i < chunkSize; i++) {
        unsigned index = batch * channelSize * chunkSize + outChannel * chunkSize + i;
        result[index] = a[index] * inv + offset[outChannel] - mean[outChannel] * inv;
      }
    }
  }
}

template <class T>
static void _mkl_batch_normalization_v2(
    const T* a, const T* scale, const T* offset, const T* mean, const T* var, T epsilon,
    unsigned batchSize, unsigned channelSize, T* result) {
  std::vector<T> invList(channelSize);
#pragma omp parallel for
  for (unsigned i = 0; i < channelSize; ++i) {
    invList[i] = (static_cast<T>(1) / std::sqrt(var[i] + epsilon)) * scale[i];
  }
#pragma omp parallel for
  for (unsigned i = 0; i < batchSize; ++i) {
    const T* ra = a + i * channelSize;
    T* r = result + i * channelSize;
    for (unsigned j = 0; j < channelSize; ++j) {
      r[j] = (ra[j] - mean[j]) * invList[j] + offset[j];
    }
  }
}

template <class T>
Variable<T> MKLNormalizationInterface<T>::batchNormalization(
    const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
    const Tensor<T>& mean, const Tensor<T>& var, T epsilon) {
  Variable<T> result(a.shape());
  unsigned batchSize = 1;
  for (size_t i = 0; i < axis; ++i) {
    batchSize *= a.shape(i);
  }
  unsigned channelSize = a.shape(axis);
  unsigned chunkSize = 1;
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    chunkSize *= a.shape(i);
  }
  if (chunkSize > 1) {
    mkl_async_wrapper(
        "batch_normalize",
        [](const T* a, const T* scale, const T* offset, const T* mean, const T* var, T epsilon,
           unsigned batchSize, unsigned channelSize, unsigned chunkSize, T* result) {
          _mkl_batch_normalization(
              a, scale, offset, mean, var, epsilon, batchSize, channelSize, chunkSize, result);
        },
        a.data(), scale.data(), offset.data(), mean.data(), var.data(), epsilon, batchSize,
        channelSize, chunkSize, result.data());
  } else {
    mkl_async_wrapper(
        "batch_normalize",
        [](const T* a, const T* scale, const T* offset, const T* mean, const T* var, T epsilon,
           unsigned batchSize, unsigned channelSize, T* result) {
          _mkl_batch_normalization_v2(
              a, scale, offset, mean, var, epsilon, batchSize, channelSize, result);
        },
        a.data(), scale.data(), offset.data(), mean.data(), var.data(), epsilon, batchSize,
        channelSize, result.data());
  }
  return result;
}

#define DEFINE_FUNC(type) template class MKLNormalizationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
