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
#include <string>
#include <tuple>

#include "framework/tfcc_shape.h"

namespace tfcc {

class Scope;

std::map<std::string, std::string> unzip(const std::string& data);
std::map<std::string, std::string> unzip_from_path(const std::string& path);

/**
 * Parse npy data.
 * @param data npy data.
 * @return tuple(shape, type, data).
 */
std::tuple<Shape, std::string, std::string> parse_npy(const std::string& data);

/**
 * This function is too slow to call frequently.
 * @param data The data will be transpose
 * @param shape The data's shape
 * @param perm A permutation of the dimensions of data.
 */
template <class T>
std::vector<T> host_transpose(std::vector<T> data, Shape shape, const std::vector<size_t>& perm);

template <class T>
inline T roundUp(T value, T modulus) {
  return (value + modulus - 1) / modulus * modulus;
}

template <class T>
inline T roundDown(T value, T modulus) {
  return value / modulus * modulus;
}

/**
 * Preload all constant variables to speed up the first time inference.
 * You must initialize Device, Session and DataLoader before call this function.
 */
void preload_constants();

/**
 * Release all constants/configure belong to the scope.
 * @param scope The constants belongs.
 */
void release_constants(const Scope* scope);

/**
 * Release all memory cache.
 */
void release_cache();

/**
 * Transfer string data to vector
 */
template <class T>
std::vector<T> transfer_string_data(const std::string& type, const std::string& data);

}  // namespace tfcc
