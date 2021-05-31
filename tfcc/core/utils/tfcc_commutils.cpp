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

#include "tfcc_commutils.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stack>
#include <streambuf>
#include <utility>

#include "zlib.h"

#include "dataloaders/tfcc_dataloader.h"
#include "exceptions/tfcc_dataformaterror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "framework/tfcc_configure.h"
#include "framework/tfcc_constant.h"
#include "framework/tfcc_constantmanager.h"
#include "framework/tfcc_device.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

static inline std::string _inflate(const char* data, size_t len, size_t targetLen) {
  size_t bufferSize = 1024 * 1024 * 1024;
  bufferSize = std::min(bufferSize, targetLen);
  std::unique_ptr<Bytef[]> buffer(new Bytef[bufferSize]);
  std::string result;
  result.reserve(targetLen);

  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  int ret = inflateInit2(&zs, -8);
  if (ret != Z_OK) {
    inflateEnd(&zs);
    throw DataFormatError("zip uncompress failed");
  }

  zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data));
  while (ret != Z_STREAM_END) {
    zs.avail_in = std::min(bufferSize, len - (zs.next_in - reinterpret_cast<const Bytef*>(data)));
    zs.avail_out = bufferSize;
    zs.next_out = buffer.get();
    ret = inflate(&zs, Z_NO_FLUSH);
    if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
      inflateEnd(&zs);
      throw DataFormatError("zip uncompress failed");
    }
    result.append(reinterpret_cast<char*>(buffer.get()), bufferSize - zs.avail_out);
  }
  inflateEnd(&zs);

  return result;
}

static inline std::string _translate_data_order(
    const std::string& data, const Shape& shape, size_t typeSize) {
  if (data.size() % typeSize != 0 || data.size() < typeSize * shape.area()) {
    throw DataFormatError("invalid data size");
  }

  std::string result(data.size(), '\0');
  unsigned lastOffset = 1;
  std::vector<unsigned> newOffsets(shape.size(), 0u);
  for (size_t i = 0; i < shape.size(); ++i) {
    newOffsets[shape.size() - i - 1] = lastOffset;
    lastOffset *= shape[shape.size() - i - 1];
  }

  lastOffset = 1;
  std::vector<unsigned> oldOffsets(shape.size(), 0u);
  for (size_t i = 0; i < shape.size(); ++i) {
    oldOffsets[i] = lastOffset;
    lastOffset *= shape[i];
  }

  for (size_t i = 0; i < data.size() / typeSize; ++i) {
    unsigned newPos = 0;
    for (size_t j = 0; j < shape.size(); ++j) {
      unsigned n = (i / oldOffsets[j]) % shape[j];
      newPos += n * newOffsets[j];
    }
    result.replace(newPos * typeSize, typeSize, data.data() + i * typeSize, typeSize);
  }

  return result;
}

std::map<std::string, std::string> unzip(const std::string& data) {
  std::map<std::string, std::string> result;
  size_t pos = 0;
  while (pos + 4 < data.size()) {
    uint32_t sign = *reinterpret_cast<const uint32_t*>(data.data() + pos);
    if (sign != 0x04034b50) {
      break;
    }
    if (pos + 30 >= data.size()) {
      throw DataFormatError("invalid zip format");
    }
    uint16_t compressType = *reinterpret_cast<const uint16_t*>(data.data() + pos + 8);
    uint64_t compressedDataSize = *reinterpret_cast<const uint32_t*>(data.data() + pos + 18);
    uint64_t uncompressedDataSize = *reinterpret_cast<const uint32_t*>(data.data() + pos + 22);
    uint32_t nameSize = *reinterpret_cast<const uint16_t*>(data.data() + pos + 26);
    uint32_t extraDataSize = *reinterpret_cast<const uint16_t*>(data.data() + pos + 28);
    if (pos + 30 + nameSize + extraDataSize >= data.size()) {
      throw DataFormatError("invalid zip format");
    }
    // extract zip64 info
    uint32_t currentExtraDataPos = 0;
    while (extraDataSize >= currentExtraDataPos + 4) {
      uint16_t currentExtraHeaderType = *reinterpret_cast<const uint16_t*>(
          data.data() + pos + 30 + nameSize + currentExtraDataPos);
      uint32_t currentExtraDataSize = *reinterpret_cast<const uint16_t*>(
          data.data() + pos + 30 + nameSize + currentExtraDataPos + 2);
      if (currentExtraHeaderType != 1) {
        currentExtraDataPos += 4 + currentExtraDataSize;
        continue;
      }
      currentExtraDataSize =
          std::min(currentExtraDataSize, extraDataSize - currentExtraDataPos - 4);
      if (currentExtraDataSize < 16) {
        throw DataFormatError("invalid zip64 extra header format");
      }
      uncompressedDataSize = *reinterpret_cast<const uint64_t*>(
          data.data() + pos + 30 + nameSize + currentExtraDataPos + 4);
      compressedDataSize = *reinterpret_cast<const uint64_t*>(
          data.data() + pos + 30 + nameSize + currentExtraDataPos + 12);
      currentExtraDataPos += 4 + currentExtraDataSize;
    }

    if (compressType != 0 && compressType != 8) {
      throw DataFormatError("invalid zip compress type: " + std::to_string(compressType));
    }
    if (pos + 30 + nameSize + extraDataSize + compressedDataSize >= data.size()) {
      throw DataFormatError("invalid zip format");
    }
    std::string name(data.data() + pos + 30, data.data() + pos + 30 + nameSize);
    if (compressType == 0) {
      result[name] = std::string(
          data.data() + pos + 30 + nameSize + extraDataSize,
          data.data() + pos + 30 + nameSize + extraDataSize + compressedDataSize);
    } else {
      result[name] = _inflate(
          data.data() + pos + 30 + nameSize + extraDataSize, compressedDataSize,
          uncompressedDataSize);
    }
    pos = pos + 30 + nameSize + extraDataSize + compressedDataSize;
  }

  return result;
}

std::map<std::string, std::string> unzip_from_path(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw InvalidArgumentError("open file: \"" + path + "\" failed");
  }
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return unzip(data);
}

static inline std::string _parse_dict(const std::string& str, const std::string& key) {
  static const std::map<char, char> pairMap = {
      {'"', '"'}, {'\'', '\''}, {'[', ']'}, {'{', '}'}, {'(', ')'},
  };
  size_t pos = str.find("'" + key + "'");
  if (pos == std::string::npos) {
    return "";
  }
  pos = str.find(":", pos);
  if (pos == std::string::npos) {
    return "";
  }
  size_t endPos = pos + 1;
  std::stack<char> pairStack;
  std::string value;
  while (true) {
    if (endPos >= str.size()) {
      return "";
    }
    if (pairStack.empty() && (str[endPos] == ',' || str[endPos] == '}')) {
      break;
    }
    if (!pairStack.empty() && str[endPos] == pairMap.find(pairStack.top())->second) {
      pairStack.pop();
    } else if (pairMap.find(str[endPos]) != pairMap.end()) {
      pairStack.push(str[endPos]);
    }
    value += str[endPos];
    ++endPos;
  }
  size_t start = value.find_first_not_of(" \r\n\r\f\b\t");
  size_t end = value.find_last_not_of(" \r\n\r\f\b\t");
  if (start == std::string::npos || end == std::string::npos || end < start) {
    return "";
  }
  return value.substr(start, end + 1 - start);
}

std::tuple<Shape, std::string, std::string> parse_npy(const std::string& data) {
  // 1. check header
  if (data.substr(0, 6) != "\x93NUMPY") {
    throw DataFormatError("invalid npy header");
  }
  // 2. parse head length
  char version = data[6];
  size_t pos = 8;
  size_t headLength = 0;
  if (version == 0x01) {
    headLength = *reinterpret_cast<const uint16_t*>(data.data() + pos);
    pos += 2;
  } else if (version == 0x02) {
    headLength = *reinterpret_cast<const uint32_t*>(data.data() + pos);
    pos += 4;
  } else {
    throw DataFormatError("unknow npy version");
  }
  // 3. parse descr
  std::string header = data.substr(pos, headLength);
  std::string descr = _parse_dict(header, "descr");
  if (descr.size() < 5) {
    throw DataFormatError("invalid npy descr");
  }
  if (descr[1] != '<' && descr[1] != '|') {
    throw DataFormatError("invalid npy endian");
  }
  for (size_t i = 3; i + 1 < descr.size(); ++i) {
    if (descr[i] < '0' || descr[i] > '9') {
      throw DataFormatError("invalid npy type size");
    }
  }
  // 4. parse shape
  std::string strShape = _parse_dict(header, "shape");
  if ((strShape[0] != '(' || strShape[strShape.size() - 1] != ')') &&
      (strShape[0] != '[' || strShape[strShape.size() - 1] != ']')) {
    throw DataFormatError("invalid npy shape");
  }
  std::vector<unsigned> s;
  std::string strNum;
  for (size_t i = 1; i < strShape.size(); ++i) {
    if (strShape[i] >= '0' && strShape[i] <= '9') {
      strNum += strShape[i];
    } else if (strNum != "") {
      s.push_back(std::stoull(strNum));
      strNum = "";
    }
  }
  if (s.size() == 0) {
    s.push_back(1u);
  }
  Shape shape(std::move(s));
  // 5. parse data order
  std::string fortranOrder = _parse_dict(header, "fortran_order");
  if (fortranOrder != "True" && fortranOrder != "False") {
    throw DataFormatError("invalid npy data order");
  }
  // 6. translate data order if needed
  size_t typeSize = std::stoull(descr.substr(3, descr.size() - 4));
  std::string body = data.substr(pos + headLength);
  if (fortranOrder == "True") {
    body = _translate_data_order(body, shape, typeSize);
  }
  return std::make_tuple(std::move(shape), descr.substr(2, descr.size() - 3), std::move(body));
}

template <class T>
std::vector<T> host_transpose(std::vector<T> data, Shape shape, const std::vector<size_t>& perm) {
  if (shape.size() != perm.size()) {
    throw InvalidArgumentError("perm isn't match shape");
  }
  std::vector<T> result;
  result.resize(data.size());

  std::vector<unsigned> newS;
  newS.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    newS.emplace_back(shape[perm[i]]);
  }

  unsigned lastOffset = 1;
  std::vector<unsigned> newOffsets(shape.size(), 0u);
  for (size_t i = 0; i < shape.size(); ++i) {
    newOffsets[perm[perm.size() - i - 1]] = lastOffset;
    lastOffset *= newS[newS.size() - i - 1];
  }
  lastOffset = 1;
  std::vector<unsigned> oldOffsets(shape.size(), 0u);
  for (size_t i = 0; i < shape.size(); ++i) {
    oldOffsets[shape.size() - i - 1] = lastOffset;
    lastOffset *= shape[shape.size() - i - 1];
  }

  for (size_t i = 0; i < data.size(); ++i) {
    unsigned newPos = 0;
    for (size_t j = 0; j < shape.size(); ++j) {
      unsigned n = (i / oldOffsets[j]) % shape[j];
      newPos += n * newOffsets[j];
    }
    result[newPos] = data[i];
  }

  return result;
}

static void preload_one_constant(const std::string& name, size_t pos, const std::string& type) {
  size_t newPos = name.find('/', pos);
  if (newPos != std::string::npos) {
    auto scopeG = Scope::scope(name.substr(pos, newPos - pos));
    preload_one_constant(name, newPos + 1, type);
    return;
  }

  if (type == "f4") {
    Constant<float>::getConstant(name.substr(pos));
  } else if (type == "f8") {
    Constant<double>::getConstant(name.substr(pos));
  } else if (type == "i1") {
    Constant<int8_t>::getConstant(name.substr(pos));
  } else if (type == "i2") {
    Constant<int16_t>::getConstant(name.substr(pos));
  } else if (type == "i4") {
    Constant<int32_t>::getConstant(name.substr(pos));
  } else if (type == "i8") {
    Constant<int64_t>::getConstant(name.substr(pos));
  } else if (type == "u1") {
    Constant<uint8_t>::getConstant(name.substr(pos));
  } else if (type == "u2") {
    Constant<uint16_t>::getConstant(name.substr(pos));
  } else if (type == "u4") {
    Constant<uint32_t>::getConstant(name.substr(pos));
  } else if (type == "u8") {
    Constant<uint64_t>::getConstant(name.substr(pos));
  } else {
    throw RuntimeError("DataLoader return a unknow type. Constant name: " + name);
  }
}

void preload_constants() {
  auto names = DataLoader::getGlobalDefault()->getAllNames();
  for (auto& name : names) {
    preload_one_constant(std::get<0>(name), 0, std::get<1>(name));
  }
}

void release_constants(const Scope* scope) {
  auto devices = Device::getAllDevices();
  for (Device* device : devices) {
    device->getConstantManager(float()).removeConstants(scope);
    device->getConstantManager(double()).removeConstants(scope);
    device->getConstantManager(int8_t()).removeConstants(scope);
    device->getConstantManager(uint8_t()).removeConstants(scope);
    device->getConstantManager(int16_t()).removeConstants(scope);
    device->getConstantManager(uint16_t()).removeConstants(scope);
    device->getConstantManager(int32_t()).removeConstants(scope);
    device->getConstantManager(uint32_t()).removeConstants(scope);
    device->getConstantManager(int64_t()).removeConstants(scope);
    device->getConstantManager(uint64_t()).removeConstants(scope);

    Configure<float>::removeConfigures(scope);
    Configure<double>::removeConfigures(scope);
    Configure<int8_t>::removeConfigures(scope);
    Configure<uint8_t>::removeConfigures(scope);
    Configure<int16_t>::removeConfigures(scope);
    Configure<uint16_t>::removeConfigures(scope);
    Configure<int32_t>::removeConfigures(scope);
    Configure<uint32_t>::removeConfigures(scope);
    Configure<int64_t>::removeConfigures(scope);
    Configure<uint64_t>::removeConfigures(scope);
  }
}

void release_cache() {
  Session::getThreadDefault()->releaseCache();
  Device::getThreadDefault()->releaseCache();
}

template <class T1, class T2>
static std::vector<T1> _transfer_string_data(const std::string& data) {
  const T2* p = reinterpret_cast<const T2*>(data.data());
  size_t count = data.size() / sizeof(T2);
  std::vector<T1> realData;
  realData.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    realData.push_back(static_cast<T1>(p[i]));
  }
  return realData;
}

template <class T>
std::vector<T> transfer_string_data(const std::string& type, const std::string& data) {
  std::vector<T> realData;
  if (type == "f4") {
    realData = _transfer_string_data<T, float>(data);
  } else if (type == "f8") {
    realData = _transfer_string_data<T, double>(data);
  } else if (type == "i1") {
    realData = _transfer_string_data<T, int8_t>(data);
  } else if (type == "i2") {
    realData = _transfer_string_data<T, int16_t>(data);
  } else if (type == "i4") {
    realData = _transfer_string_data<T, int32_t>(data);
  } else if (type == "i8") {
    realData = _transfer_string_data<T, int64_t>(data);
  } else if (type == "u1") {
    realData = _transfer_string_data<T, uint8_t>(data);
  } else if (type == "u2") {
    realData = _transfer_string_data<T, uint16_t>(data);
  } else if (type == "u4") {
    realData = _transfer_string_data<T, uint32_t>(data);
  } else if (type == "u8") {
    realData = _transfer_string_data<T, uint64_t>(data);
  }
  return realData;
}

#define DEFINE_FUNC(type)                                                    \
  template std::vector<type> host_transpose(                                 \
      std::vector<type> data, Shape shape, const std::vector<size_t>& perm); \
  template std::vector<type> transfer_string_data(const std::string&, const std::string&);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
