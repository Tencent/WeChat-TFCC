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
class MKLColMajorStrideOutput {
  T* _data;
  unsigned _row;
  unsigned _col;
  unsigned _stride;

 public:
  MKLColMajorStrideOutput() : _data(nullptr), _row(0), _col(0), _stride(0) {}

  MKLColMajorStrideOutput(T* data, unsigned row, unsigned col, unsigned stride)
      : _data(data), _row(row), _col(col), _stride(stride) {}

  unsigned row() const { return _row; }

  unsigned col() const { return _col; }

  unsigned stride() const { return _stride; }

  unsigned rowStride() const { return 1; }

  unsigned colStride() const { return _stride; }

  const T* data() const { return _data; }

  const T* data(unsigned r, unsigned c) const { return _data + r * rowStride() + c * colStride(); }

  const T& operator()(unsigned r, unsigned c) const { return *data(r, c); }

  T* data() { return _data; }

  T* data(unsigned r, unsigned c) { return _data + r * rowStride() + c * colStride(); }

  T& operator()(unsigned r, unsigned c) { return *data(r, c); }

  MKLColMajorStrideOutput subOutput(unsigned r, unsigned c, unsigned rs, unsigned cs) {
    return MKLColMajorStrideOutput(data(r, c), rs, cs, _stride);
  }
};

}  // namespace tfcc
