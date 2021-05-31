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

#pragma once

namespace tfcc {

template <class T>
class _MKLActivationKernel {
 public:
  static inline void sigmoid(const T* a, unsigned total, T* b);
  static inline void tanh(const T* a, unsigned total, T* b);
  static inline void relu(const T* a, unsigned total, T* b);
  static inline void leakyRelu(const T* a, unsigned total, T alpha, T* b);
  static inline void softplus(const T* a, unsigned total, T* b);
  static inline void log(const T* a, unsigned total, T* b);
  static inline void rsqrt(const T* a, unsigned total, T* b);
  static inline void softmax(const T* a, unsigned s1, unsigned s2, unsigned s3, T* b);
  static inline void softmaxV2(const T* a, unsigned s1, unsigned s2, T* b);
  static inline void sin(const T* a, unsigned total, T* b);
  static inline void cos(const T* a, unsigned total, T* b);
  static inline void pow(const T* a, unsigned total, T exponent, T* b);
  static inline void powV2(const T* a, const T* exponent, T* b, unsigned total);
  static inline void powV3(const T* exponent, unsigned total, T a, T* b);
  static inline void gelu(const T* a, unsigned total, T* b);
  static inline void geluAccurate(const T* a, unsigned total, T* b);
  static inline void erf(const T* a, unsigned total, T* b);
  static inline void asin(const T* a, unsigned total, T* b);
  static inline void asinh(const T* a, unsigned total, T* b);
  static inline void acos(const T* a, unsigned total, T* b);
  static inline void acosh(const T* a, unsigned total, T* b);
  static inline void atan(const T* a, unsigned total, T* b);
  static inline void atanh(const T* a, unsigned total, T* b);
  static inline void sign(const T* a, unsigned total, T* b);
};

template <class T>
inline void _MKLActivationKernel<T>::sigmoid(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(1 / (1 + std::exp(0 - v)));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::tanh(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::tanh(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::relu(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = std::max(v, static_cast<T>(0));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::leakyRelu(const T* a, unsigned total, T alpha, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    T v2 = alpha * v;
    b[i] = std::max(v, v2);
  }
}

template <class T>
inline void _MKLActivationKernel<T>::softplus(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::log(std::exp(v) + 1));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::log(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::log(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::rsqrt(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(1 / std::sqrt(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::softmax(
    const T* a, unsigned s1, unsigned s2, unsigned s3, T* b) {
  T maxValue = std::numeric_limits<T>::lowest();
  unsigned total = s1 * s2 * s3;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) maxValue = std::max(maxValue, a[i]);
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    v = static_cast<T>(exp(v - maxValue));
    b[i] = v;
  }

  unsigned batch = s1 * s3;
#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    unsigned x = i / s3;
    unsigned z = i % s3;
    T sumValue = static_cast<T>(0);
    for (unsigned y = 0; y < s2; ++y) sumValue += b[z + y * s3 + x * s3 * s2];
    for (unsigned y = 0; y < s2; ++y) b[z + y * s3 + x * s3 * s2] /= sumValue;
  }
}

template <class T>
inline void _MKLActivationKernel<T>::softmaxV2(const T* a, unsigned s1, unsigned s2, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < s1; ++i) {
    T maxValue = std::numeric_limits<T>::lowest();
    const T* ra = a + i * s2;
    T* rb = b + i * s2;
    for (unsigned y = 0; y < s2; ++y) maxValue = std::max(maxValue, ra[y]);
    for (unsigned y = 0; y < s2; ++y) {
      T v = ra[y];
      v = static_cast<T>(exp(v - maxValue));
      rb[y] = v;
    }

    T sumValue = static_cast<T>(0);
    for (unsigned y = 0; y < s2; ++y) sumValue += rb[y];
    for (unsigned y = 0; y < s2; ++y) rb[y] /= sumValue;
  }
}

template <class T>
inline void _MKLActivationKernel<T>::sin(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::sin(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::cos(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::cos(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::pow(const T* a, unsigned total, T exponent, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::pow(v, exponent));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::powV2(const T* a, const T* exponent, T* b, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    T e = exponent[i];
    b[i] = static_cast<T>(std::pow(v, e));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::powV3(const T* exponent, unsigned total, T a, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T e = exponent[i];
    b[i] = static_cast<T>(std::pow(a, e));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::gelu(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    T tmp = static_cast<T>(0.7978845608028654) * (v + 0.044715 * v * v * v);
    b[i] = v * static_cast<T>(0.5) * (static_cast<T>(1.0) + std::tanh(tmp));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::geluAccurate(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = 0.5 * v * (1 + std::erf(v / static_cast<T>(1.4142135623730951)));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::erf(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::erf(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::asin(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::asin(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::asinh(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::asinh(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::acos(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::acos(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::acosh(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::acosh(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::atan(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::atan(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::atanh(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = static_cast<T>(std::atanh(v));
  }
}

template <class T>
inline void _MKLActivationKernel<T>::sign(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    b[i] = v < 0 ? static_cast<T>(-1) : (v > 0 ? static_cast<T>(1) : static_cast<T>(0));
  }
}

}  // namespace tfcc
