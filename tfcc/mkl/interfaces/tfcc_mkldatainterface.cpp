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

#include "tfcc_mkldatainterface.h"
#include "tfcc_mklinterfacehelper.h"

#include <omp.h>
#include <cstring>
#include <type_traits>

#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

// mkl functions
template <class T>
static typename std::enable_if<std::is_arithmetic<T>::value, void>::type _mkl_ones(
    T* a, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    a[i] = static_cast<T>(1);
  }
}

template <class T>
static typename std::enable_if<!std::is_arithmetic<T>::value, void>::type _mkl_ones(
    T* a, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    a[i].real = static_cast<typename TypeInfo<T>::BaseType>(1.0);
    a[i].imag = static_cast<typename TypeInfo<T>::BaseType>(0.0);
  }
}

template <class T>
static void _mkl_copy(T* dst, const T* data, unsigned total) {
#pragma omp parallel
  {
    unsigned pcnt = omp_get_num_threads();
    unsigned secTotal = total / pcnt;
    unsigned threadNum = omp_get_thread_num();
    unsigned offset = secTotal * threadNum;
    if (threadNum == pcnt - 1) {
      secTotal += total - secTotal * pcnt;
    }
    memcpy(dst + offset, data + offset, secTotal * sizeof(T));
  }
}

template <class T>
static void _mkl_zeros(T* dst, unsigned total) {
#pragma omp parallel
  {
    unsigned pcnt = omp_get_num_threads();
    unsigned secTotal = total / pcnt;
    unsigned threadNum = omp_get_thread_num();
    unsigned offset = secTotal * threadNum;
    if (threadNum == pcnt - 1) {
      secTotal += total - secTotal * pcnt;
    }
    memset(dst + offset, 0, secTotal * sizeof(T));
  }
}

template <class T1, class T2>
static void _mkl_cast(const T1* a, unsigned total, T2* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    b[i] = static_cast<T2>(a[i]);
  }
}

template <class T>
static void _mkl_cast_to_boolean(const T* a, unsigned total, uint8_t* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    b[i] = a[i] ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
  }
}

// helper function
template <class T1, class T2>
static inline Variable<T2> _cast_helper(const Tensor<T1>& a, T2) {
  Variable<T2> result(a.shape());
  mkl_async_wrapper(
      "cast", [](const T1* a, unsigned total, T2* b) { _mkl_cast(a, total, b); }, a.data(),
      a.size(), result.data());
  return result;
}

// class function
template <class T>
void MKLDataInterface<T>::set(T* dst, const T* data, size_t len) {
  memcpy(dst, data, len * sizeof(T));
}

template <class T>
void MKLDataInterface<T>::set(Variable<T>& a, std::vector<T>&& data) {
  std::shared_ptr<std::vector<T>> p = std::make_shared<std::vector<T>>(std::move(data));
  if (a.size() > 512) {
    mkl_async_wrapper(
        "set",
        [](T* dst, std::shared_ptr<std::vector<T>> p, unsigned total) {
          _mkl_copy(dst, p->data(), total);
        },
        a.data(), p, a.size());
  } else {
    mkl_async_wrapper(
        "set",
        [](T* dst, std::shared_ptr<std::vector<T>> p, unsigned total) {
          memcpy(dst, p->data(), total * sizeof(T));
        },
        a.data(), p, a.size());
  }
}

template <class T>
void MKLDataInterface<T>::set(Variable<T>& a, const T* data) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "set", [](T* dst, const T* data, unsigned total) { _mkl_copy(dst, data, total); }, a.data(),
        data, a.size());
  } else {
    mkl_async_wrapper("set", memcpy, a.data(), data, a.size() * sizeof(T));
  }
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->sync();
}

template <class T>
void MKLDataInterface<T>::get(const Tensor<T>& a, T* data) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "get", [](T* dst, const T* data, unsigned total) { _mkl_copy(dst, data, total); }, data,
        a.data(), a.size());
  } else {
    mkl_async_wrapper("get", memcpy, data, a.data(), a.size() * sizeof(T));
  }

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->sync();
}

template <class T>
void MKLDataInterface<T>::zeros(Variable<T>& a) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "zeros", [](T* dst, unsigned total) { _mkl_zeros(dst, total); }, a.data(), a.size());
  } else {
    mkl_async_wrapper("zeros", memset, a.data(), 0, a.size() * sizeof(T));
  }
}

template <class T>
void MKLDataInterface<T>::ones(Variable<T>& a) {
  mkl_async_wrapper(
      "ones", [](T* a, unsigned total) { _mkl_ones(a, total); }, a.data(), a.size());
}

template <class T>
Variable<T> MKLDataInterface<T>::copy(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  if (a.size() > 512) {
    mkl_async_wrapper(
        "copy", [](T* dst, const T* data, unsigned total) { _mkl_copy(dst, data, total); },
        result.data(), a.data(), a.size());
  } else {
    mkl_async_wrapper("copy", memcpy, result.data(), a.data(), a.size() * sizeof(T));
  }
  return result;
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<float>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<double>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<int8_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<uint8_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<int16_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<uint16_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<int32_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<uint32_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<int64_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> MKLDataInterface<T>::cast(const Tensor<uint64_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<uint8_t> MKLDataInterface<T>::cast_to_boolean(const Tensor<T>& a) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "cast_to_boolean",
      [](const T* a, unsigned total, uint8_t* b) { _mkl_cast_to_boolean(a, total, b); }, a.data(),
      a.size(), result.data());
  return result;
}

// complex
template <class T>
void MKLDataInterface<Complex<T>>::set(Complex<T>* dst, const Complex<T>* data, size_t len) {
  memcpy(dst, data, len * sizeof(Complex<T>));
}

template <class T>
void MKLDataInterface<Complex<T>>::set(Variable<Complex<T>>& a, std::vector<Complex<T>>&& data) {
  std::shared_ptr<std::vector<Complex<T>>> p =
      std::make_shared<std::vector<Complex<T>>>(std::move(data));
  if (a.size() > 512) {
    mkl_async_wrapper(
        "set",
        [](Complex<T>* dst, std::shared_ptr<std::vector<Complex<T>>> p, unsigned total) {
          _mkl_copy(dst, p->data(), total);
        },
        a.data(), p, a.size());
  } else {
    mkl_async_wrapper(
        "set",
        [](Complex<T>* dst, std::shared_ptr<std::vector<Complex<T>>> p, unsigned total) {
          memcpy(dst, p->data(), total * sizeof(Complex<T>));
        },
        a.data(), p, a.size());
  }
}

template <class T>
void MKLDataInterface<Complex<T>>::set(Variable<Complex<T>>& a, const Complex<T>* data) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "set",
        [](Complex<T>* dst, const Complex<T>* data, unsigned total) {
          _mkl_copy(dst, data, total);
        },
        a.data(), data, a.size());
  } else {
    mkl_async_wrapper("set", memcpy, a.data(), data, a.size() * sizeof(Complex<T>));
  }
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->sync();
}

template <class T>
void MKLDataInterface<Complex<T>>::get(const Tensor<Complex<T>>& a, Complex<T>* data) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "get",
        [](Complex<T>* dst, const Complex<T>* data, unsigned total) {
          _mkl_copy(dst, data, total);
        },
        data, a.data(), a.size());
  } else {
    mkl_async_wrapper("get", memcpy, data, a.data(), a.size() * sizeof(Complex<T>));
  }

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->sync();
}

template <class T>
void MKLDataInterface<Complex<T>>::zeros(Variable<Complex<T>>& a) {
  if (a.size() > 512) {
    mkl_async_wrapper(
        "zeros", [](Complex<T>* dst, unsigned total) { _mkl_zeros(dst, total); }, a.data(),
        a.size());
  } else {
    mkl_async_wrapper("zeros", memset, a.data(), 0, a.size() * sizeof(Complex<T>));
  }
}

template <class T>
void MKLDataInterface<Complex<T>>::ones(Variable<Complex<T>>& a) {
  mkl_async_wrapper(
      "ones", [](Complex<T>* a, unsigned total) { _mkl_ones(a, total); }, a.data(), a.size());
}

template <class T>
Variable<Complex<T>> MKLDataInterface<Complex<T>>::copy(const Tensor<Complex<T>>& a) {
  Variable<Complex<T>> result(a.shape());
  if (a.size() > 512) {
    mkl_async_wrapper(
        "copy",
        [](Complex<T>* dst, const Complex<T>* data, unsigned total) {
          _mkl_copy(dst, data, total);
        },
        result.data(), a.data(), a.size());
  } else {
    mkl_async_wrapper("copy", memcpy, result.data(), a.data(), a.size() * sizeof(Complex<T>));
  }
  return result;
}

#define DEFINE_FUNC(type) template class MKLDataInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
