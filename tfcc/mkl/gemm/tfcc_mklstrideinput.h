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

template <class T, bool WidthMajor>
class MKLStrideInput {
  const T* _data;
  unsigned _width;
  unsigned _depth;
  unsigned _stride;

 public:
  MKLStrideInput() : _data(nullptr), _width(0), _depth(0), _stride(0) {}

  MKLStrideInput(const T* data, unsigned width, unsigned depth, unsigned stride)
      : _data(data), _width(width), _depth(depth), _stride(stride) {}

  unsigned width() const { return _width; }

  unsigned depth() const { return _depth; }

  unsigned stride() const { return _stride; }

  unsigned widthStride() const { return WidthMajor ? _stride : 1; }

  unsigned depthStride() const { return WidthMajor ? 1 : _stride; }

  const T* data() const { return _data; }

  const T* data(unsigned w, unsigned d) const {
    return _data + w * widthStride() + d * depthStride();
  }

  const T& operator()(unsigned w, unsigned d) const { return *data(w, d); }

  MKLStrideInput subInput(unsigned w, unsigned d, unsigned ws, unsigned ds) {
    return MKLStrideInput(data(w, d), ws, ds, _stride);
  }
};

}  // namespace tfcc
