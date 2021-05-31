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

#include "tfcc_mklblasinterface.h"

#include <cstring>

#include <memory>

#include <omp.h>
#include <dnnl.hpp>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mkldevice.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
static inline void _call_matmul_wrapper(
    const T* a, unsigned strideA, const T* b, unsigned strideB, T* c, unsigned strideC, int m,
    int n, int k, int batchCount) {
  throw NotImplementedError();
}

static inline void _call_matmul_wrapper(
    const float* a, unsigned strideA, const float* b, unsigned strideB, float* c, unsigned strideC,
    int m, int n, int k, int batchCount) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  dnnl::memory::desc aDesc({batchCount, m, k}, dnnl::memory::data_type::f32, {strideA, k, 1});
  dnnl::memory::desc bDesc({batchCount, k, n}, dnnl::memory::data_type::f32, {strideB, n, 1});
  dnnl::memory::desc cDesc({batchCount, m, n}, dnnl::memory::data_type::f32, {strideC, n, 1});

  dnnl::memory aM(aDesc, session->getDNNLEngine(), const_cast<float*>(a));
  dnnl::memory bM(bDesc, session->getDNNLEngine(), const_cast<float*>(b));
  dnnl::memory cM(cDesc, session->getDNNLEngine(), c);

  dnnl::matmul::desc matmulDesc(aDesc, bDesc, cDesc);

  mkl_async_wrapper(
      "matmul",
      [](dnnl::matmul::desc matmulDesc, dnnl::memory a, dnnl::memory b, dnnl::memory c,
         MKLSession* session) {
        dnnl::matmul::primitive_desc primitiveDesc(matmulDesc, session->getDNNLEngine());
        dnnl::matmul(primitiveDesc)
            .execute(
                session->getDNNLStream(),
                {{DNNL_ARG_SRC, a}, {DNNL_ARG_WEIGHTS, b}, {DNNL_ARG_DST, c}});
        session->getDNNLStream().wait();
      },
      matmulDesc, aM, bM, cM, session);
}

template <class T>
static inline void _mkl_set_bias_wrapper(const T* bias, unsigned batch, unsigned chunk, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    T* r = result + i * chunk;
    for (unsigned j = 0; j < chunk; ++j) {
      r[j] = bias[j];
    }
  }
}

template <class T>
static inline void _call_matmul_with_beta_wrapper(
    const T* a, unsigned strideA, const T* b, unsigned strideB, T* c, unsigned strideC, int m,
    int n, int k, int batchCount) {
  throw NotImplementedError();
}

static inline void _call_matmul_with_beta_wrapper(
    const float* a, unsigned strideA, const float* b, unsigned strideB, float* c, unsigned strideC,
    int m, int n, int k, int batchCount) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  dnnl::memory::desc aDesc({batchCount, m, k}, dnnl::memory::data_type::f32, {strideA, k, 1});
  dnnl::memory::desc bDesc({batchCount, k, n}, dnnl::memory::data_type::f32, {strideB, n, 1});
  dnnl::memory::desc cDesc({batchCount, m, n}, dnnl::memory::data_type::f32, {strideC, n, 1});

  dnnl::memory aM(aDesc, session->getDNNLEngine(), const_cast<float*>(a));
  dnnl::memory bM(bDesc, session->getDNNLEngine(), const_cast<float*>(b));
  dnnl::memory cM(cDesc, session->getDNNLEngine(), c);

  dnnl::matmul::desc matmulDesc(aDesc, bDesc, cDesc);

  dnnl::primitive_attr attr;
  dnnl::post_ops po;
  po.append_sum(1.0f);
  attr.set_post_ops(po);

  mkl_async_wrapper(
      "matmul_with_bias",
      [](dnnl::matmul::desc matmulDesc, dnnl::primitive_attr attr, dnnl::memory a, dnnl::memory b,
         dnnl::memory c, MKLSession* session) {
        dnnl::matmul::primitive_desc primitiveDesc(matmulDesc, attr, session->getDNNLEngine());
        dnnl::matmul(primitiveDesc)
            .execute(
                session->getDNNLStream(),
                {{DNNL_ARG_SRC, a}, {DNNL_ARG_WEIGHTS, b}, {DNNL_ARG_DST, c}});
        session->getDNNLStream().wait();
      },
      matmulDesc, attr, aM, bM, cM, session);
}

template <class T>
MKLBlasInterface<T>::MKLBlasInterface(const MKLDevice& device) {}

template <class T>
MKLBlasInterface<T>::~MKLBlasInterface() {}

template <class T>
Variable<T> MKLBlasInterface<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) {
  unsigned m = a.shape(a.shape().size() - 2);
  unsigned n = b.shape(b.shape().size() - 1);
  unsigned k = a.shape(a.shape().size() - 1);
  unsigned strideA = m * k;
  unsigned strideB = n * k;
  unsigned strideC = m * n;
  if (a.shape().size() != b.shape().size()) {
    strideA = a.shape().size() == 2 ? 0 : strideA;
    strideB = b.shape().size() == 2 ? 0 : strideB;
  }
  std::vector<unsigned> resultS =
      a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
  resultS[resultS.size() - 2] = m;
  resultS[resultS.size() - 1] = n;

  Variable<T> result(std::move(resultS));
  unsigned batchCount = result.size() / (result.shape(result.shape().size() - 1) *
                                         result.shape(result.shape().size() - 2));

  _call_matmul_wrapper(
      a.data(), strideA, b.data(), strideB, result.data(), strideC, m, n, k, batchCount);

  return result;
}

template <class T>
Variable<T> MKLBlasInterface<T>::matmul(
    const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& bias) {
  unsigned m = a.shape(a.shape().size() - 2);
  unsigned n = b.shape(b.shape().size() - 1);
  unsigned k = a.shape(a.shape().size() - 1);
  unsigned strideA = m * k;
  unsigned strideB = n * k;
  unsigned strideC = m * n;
  if (a.shape().size() != b.shape().size()) {
    strideA = a.shape().size() == 2 ? 0 : strideA;
    strideB = b.shape().size() == 2 ? 0 : strideB;
  }
  std::vector<unsigned> resultS =
      a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
  resultS[resultS.size() - 2] = m;
  resultS[resultS.size() - 1] = n;

  Variable<T> result(std::move(resultS));
  unsigned batchCount = result.size() / (result.shape(result.shape().size() - 1) *
                                         result.shape(result.shape().size() - 2));

  mkl_async_wrapper(
      "matmul_set_bias",
      [](const T* bias, unsigned batch, unsigned chunk, T* result) {
        _mkl_set_bias_wrapper(bias, batch, chunk, result);
      },
      bias.data(), result.size() / bias.size(), bias.size(), result.data());

  _call_matmul_with_beta_wrapper(
      a.data(), strideA, b.data(), strideB, result.data(), strideC, m, n, k, batchCount);

  return result;
}

#define DEFINE_FUNC(type) template class MKLBlasInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
