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

#include <map>
#include <vector>

#include "tfcc_mklinputpacker.h"

namespace tfcc {

template <class T, bool WidthMajor, unsigned KernelWidth, unsigned KernelDepth, unsigned KernelCell>
class MKLInputPackerManager {
  MKLInputPacker<T, WidthMajor, KernelWidth, KernelDepth, KernelCell> _packer;
  std::map<const T*, size_t> _packCache;
  std::vector<T>& _cacheBuffer;

 public:
  MKLInputPackerManager(unsigned cacheWidth, unsigned cacheDepth, std::vector<T>& cacheBuffer)
      : _packer(cacheWidth, cacheDepth), _cacheBuffer(cacheBuffer) {}

  T* pack(MKLStrideInput<T, WidthMajor> src) {
    auto it = _packCache.find(src.data());
    if (it == _packCache.end()) {
      size_t pos = _cacheBuffer.size();
      unsigned expectLength =
          roundUp(src.width(), KernelWidth * KernelCell) * roundUp(src.depth(), KernelDepth);
      _cacheBuffer.resize(_cacheBuffer.size() + expectLength);
      T* result = _cacheBuffer.data() + pos;
      _packer.process(result, src);
      _packCache[src.data()] = pos;
      return result;
    }
    return _cacheBuffer.data() + it->second;
  }
};

}  // namespace tfcc
