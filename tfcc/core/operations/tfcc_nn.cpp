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

#include "tfcc_nn.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_basicinterface.h"
#include "interfaces/tfcc_convolutioninterface.h"
#include "interfaces/tfcc_datainterface.h"
#include "interfaces/tfcc_gatherinterface.h"
#include "interfaces/tfcc_normalizationinterface.h"
#include "interfaces/tfcc_scatterinterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace nn {

static inline unsigned _calculate_padding_size(
    unsigned input, unsigned output, unsigned stride, unsigned kernel) {
  long padding = (static_cast<long>(output) - 1) * static_cast<long>(stride) +
                 static_cast<long>(kernel) - static_cast<long>(input);
  padding = std::max(0l, padding);
  return static_cast<unsigned>(padding);
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.size() == 0 || kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }
  if (input.shape(nhwc ? 3 : 1) != kernel.shape(1)) {
    throw InvalidArgumentError("input channels error");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2d(
      input, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.size() == 0 || kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }
  if (input.shape(nhwc ? 3 : 1) != kernel.shape(1)) {
    throw InvalidArgumentError("input channels error");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2d(
      input, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
      dilateWidth);
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.size() == 0 || kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }
  if (input.shape(nhwc ? 3 : 1) != kernel.shape(1) * group) {
    throw InvalidArgumentError("input channels error");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2d(
      input, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
      dilateWidth, group);
}

template <class T>
Variable<T> conv2d_same(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned strideHeight,
    unsigned strideWidth) {
  if (input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  unsigned inputHeight = input.shape(nhwc ? 1 : 2);
  unsigned inputWidth = input.shape(nhwc ? 2 : 3);
  unsigned outputHeight = (inputHeight + strideHeight - 1) / strideHeight;
  unsigned outputWidth = (inputWidth + strideWidth - 1) / strideWidth;
  unsigned totalPaddingHeight =
      _calculate_padding_size(inputHeight, outputHeight, strideHeight, kernelHeight);
  unsigned totalPaddingWidth =
      _calculate_padding_size(inputWidth, outputWidth, strideWidth, kernelWidth);
  unsigned paddingHeight = static_cast<unsigned>(totalPaddingHeight) / 2;
  unsigned paddingWidth = static_cast<unsigned>(totalPaddingWidth) / 2;
  Variable<T> tmp;
  View<T> realInput(input);
  if (totalPaddingHeight % 2 == 1) {
    if (nhwc) {
      tmp = Variable<T>(
          {realInput.shape(0), realInput.shape(1) + totalPaddingHeight, realInput.shape(2),
           realInput.shape(3)});
    } else {
      tmp = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2) + totalPaddingHeight,
           realInput.shape(3)});
    }
    interface.getDataInterface().zeros(tmp);
    interface.getBasicInterface().assignTo(realInput, nhwc ? 1 : 2, paddingHeight, tmp);
    realInput = View<T>(tmp);
    paddingHeight = 0;
  }

  Variable<T> tmp2;
  if (totalPaddingWidth % 2 == 1) {
    if (nhwc) {
      tmp2 = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2) + totalPaddingWidth,
           realInput.shape(3)});
    } else {
      tmp2 = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2),
           realInput.shape(3) + totalPaddingWidth});
    }
    interface.getDataInterface().zeros(tmp2);
    interface.getBasicInterface().assignTo(realInput, nhwc ? 2 : 3, paddingWidth, tmp2);
    realInput = View<T>(tmp2);
    paddingWidth = 0;
  }

  Variable<T> result = interface.getConvolutionInterface().conv2d(
      realInput, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
  return result;
}

template <class T>
Variable<T> conv2d_valid(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned strideHeight,
    unsigned strideWidth) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.size() == 0 || kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2d(
      input, nhwc, kernel, 0, 0, strideHeight, strideWidth);
}

template <class T>
Variable<T> conv1d(
    const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned padding, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 3) {
    throw InvalidArgumentError("invalid kernel");
  }

  View<T> realInput;
  View<T> realKernel(kernel, {kernel.shape(0), kernel.shape(1), 1, kernel.shape(2)});
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  return conv2d(realInput, nwc, realKernel, 0, padding, 1, stride);
}

template <class T>
Variable<T> conv1d_same(
    const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 3) {
    throw InvalidArgumentError("invalid kernel");
  }

  View<T> realInput;
  View<T> realKernel(kernel, {kernel.shape(0), kernel.shape(1), 1, kernel.shape(2)});
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = conv2d_same(realInput, nwc, realKernel, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

/**
 * kernel: [out_channels, in_channels, width], padding = 'valid'
 */
template <class T>
Variable<T> conv1d_valid(
    const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 3) {
    throw InvalidArgumentError("invalid kernel");
  }

  View<T> realInput;
  View<T> realKernel(kernel, {kernel.shape(0), kernel.shape(1), 1, kernel.shape(2)});
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = conv2d_valid(realInput, nwc, realKernel, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

template <class T>
Variable<T> conv2d_backward_data(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  if (input.shape().size() != 4 || input.size() == 0) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 4 || kernel.size() == 0) {
    throw InvalidArgumentError("invalid kernel");
  }
  unsigned inHeight = input.shape(nhwc ? 1 : 2);
  unsigned inWidth = input.shape(nhwc ? 2 : 3);
  unsigned inChannel = input.shape(nhwc ? 3 : 1);
  unsigned kernelInChannels = kernel.shape(0);
  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);
  if (inChannel != kernelInChannels) {
    throw InvalidArgumentError("input channel not match with kernel input channel");
  }
  if ((inHeight - 1) * strideHeight + kernelHeight <= 2 * paddingHeight) {
    throw InvalidArgumentError("invalid input and kernel");
  }
  if ((inWidth - 1) * strideWidth + kernelWidth <= 2 * paddingWidth) {
    throw InvalidArgumentError("invalid input and kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2dBackwardData(
      input, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
}

template <class T>
Variable<T> conv2d_transpose(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned outputHeight,
    unsigned outputWidth, unsigned strideHeight, unsigned strideWidth) {
  if (input.shape().size() != 4 || input.size() == 0) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernel.shape().size() != 4) {
    throw InvalidArgumentError("invalid kernel");
  }

  unsigned kernelHeight = kernel.shape(2);
  unsigned kernelWidth = kernel.shape(3);

  unsigned paddingHeight = (kernelHeight - strideHeight) / 2;
  unsigned paddingWidth = (kernelWidth - strideWidth) / 2;

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().conv2dBackwardData(
      input, nhwc, kernel, paddingHeight, paddingWidth, strideHeight, strideWidth);
}

template <class T>
Variable<T> embedding_lookup(const Tensor<T>& params, const std::vector<unsigned> ids) {
  auto s = params.shape().toVector();
  s[0] = static_cast<unsigned>(ids.size());
  Variable<T> result(s);

  Interface<T>& interface = Operation<T>::getCurrentInterface();

  for (size_t i = 0; i < ids.size(); ++i) {
    unsigned id = ids[i];
    if (id >= params.shape(0)) {
      throw InvalidArgumentError("invalid id");
    }
    View<T> view(params, params.shape(), id, id + 1);

    interface.getBasicInterface().assignTo(view, 0, i, result);
  }

  return result;
}

template <class T>
Variable<T> embedding_lookup(
    const Tensor<T>& params, const std::vector<unsigned> ids, Shape idShape) {
  if (idShape.area() != ids.size()) {
    throw InvalidArgumentError("ids.size() and idShape don't match");
  }

  Variable<T> result = embedding_lookup(params, ids);
  auto s = result.shape().toVector();
  s.erase(s.begin());
  auto idShapeVector = idShape.toVector();
  s.insert(s.begin(), idShapeVector.begin(), idShapeVector.end());
  result.reshape(s);
  return result;
}

template <class T, class IDX>
static inline Variable<T> _gather_inner(const Tensor<T>& params, const Tensor<IDX>& indices) {
  if (params.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (indices.size() == 0) {
    std::vector<unsigned> shape;
    for (size_t i = 0; i < indices.shape().size(); ++i) {
      shape.push_back(indices.shape(i));
    }
    for (size_t i = 1; i < params.shape().size(); ++i) {
      shape.push_back(params.shape(i));
    }

    Variable<T> result(shape);
    return result;
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getGatherInterface().gather(params, indices);
}

template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<uint32_t>& indices) {
  return _gather_inner(params, indices);
}

template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<int32_t>& indices) {
  return _gather_inner(params, indices);
}

template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<uint64_t>& indices) {
  return _gather_inner(params, indices);
}

template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<int64_t>& indices) {
  return _gather_inner(params, indices);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<uint32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  if (indices.size() == 0 || updates.size() == 0 || shape.area() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getScatterInterface().scatterND(indices, updates, shape);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<int32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  if (indices.size() == 0 || updates.size() == 0 || shape.area() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getScatterInterface().scatterND(indices, updates, shape);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<uint64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  if (indices.size() == 0 || updates.size() == 0 || shape.area() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getScatterInterface().scatterND(indices, updates, shape);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<int64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  if (indices.size() == 0 || updates.size() == 0 || shape.area() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getScatterInterface().scatterND(indices, updates, shape);
}

template <class T, class IDX>
static inline Variable<T> _scatter_nd_inner(
    const Tensor<T>& data, const Tensor<IDX>& indices, const Tensor<T>& updates) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  if (indices.size() == 0) {
    return interface.getDataInterface().copy(data);
  }
  if (data.size() == 0 || updates.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  return interface.getScatterInterface().scatterND(data, indices, updates);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<uint32_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_inner(data, indices, updates);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<int32_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_inner(data, indices, updates);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<uint64_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_inner(data, indices, updates);
}

template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<int64_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_inner(data, indices, updates);
}

template <class T>
Variable<T> one_hot(const std::vector<unsigned> ids, unsigned depth, T onValue, T offValue) {
  std::vector<T> data(ids.size() * depth, offValue);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] >= depth) {
      throw InvalidArgumentError("invalid id");
    }
    data[i * depth + ids[i]] = onValue;
  }

  Variable<T> result({static_cast<unsigned>(ids.size()), depth});

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(result, data.data());
  return result;
}

template <class T>
Variable<T> one_hot(
    const std::vector<unsigned> ids, Shape idShape, unsigned depth, T onValue, T offValue) {
  if (idShape.area() != ids.size()) {
    throw InvalidArgumentError("ids.size() and idShape don't match");
  }

  Variable<T> result = one_hot(ids, depth, onValue, offValue);
  auto s = idShape.toVector();
  s.push_back(depth);
  result.reshape(s);
  return result;
}

template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y) {
  if (condition.shape() != x.shape() || x.shape() != y.shape()) {
    throw InvalidArgumentError("condition x and y has different shape");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().where(condition, x, y);
}

template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y) {
  if (condition.shape() != y.shape()) {
    throw InvalidArgumentError("condition x and y has different shape");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().where(condition, x, y);
}

template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y) {
  if (condition.shape() != x.shape()) {
    throw InvalidArgumentError("condition x and y has different shape");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().where(condition, x, y);
}

template <class T>
Variable<T> max_pool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernelHeight == 0 || kernelWidth == 0) {
    throw InvalidArgumentError("invalid kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().maxPool2d(
      input, nhwc, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight,
      strideWidth);
}

template <class T>
Variable<T> max_pool2d_same(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth) {
  if (input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernelHeight == 0 || kernelWidth == 0) {
    throw InvalidArgumentError("invalid kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  unsigned inputHeight = input.shape(nhwc ? 1 : 2);
  unsigned inputWidth = input.shape(nhwc ? 2 : 3);
  unsigned outputHeight = (inputHeight + strideHeight - 1) / strideHeight;
  unsigned outputWidth = (inputWidth + strideWidth - 1) / strideWidth;
  unsigned totalPaddingHeight =
      _calculate_padding_size(inputHeight, outputHeight, strideHeight, kernelHeight);
  unsigned totalPaddingWidth =
      _calculate_padding_size(inputWidth, outputWidth, strideWidth, kernelWidth);
  unsigned paddingHeight = totalPaddingHeight / 2;
  unsigned paddingWidth = totalPaddingWidth / 2;

  Variable<T> tmp;
  View<T> realInput(input);
  if (totalPaddingHeight % 2 == 1) {
    if (nhwc) {
      tmp = Variable<T>(
          {realInput.shape(0), realInput.shape(1) + totalPaddingHeight, realInput.shape(2),
           realInput.shape(3)});
    } else {
      tmp = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2) + totalPaddingHeight,
           realInput.shape(3)});
    }
    interface.getDataInterface().zeros(tmp);
    interface.getBasicInterface().assignTo(realInput, nhwc ? 1 : 2, paddingHeight, tmp);
    realInput = View<T>(tmp);
    paddingHeight = 0;
  }

  Variable<T> tmp2;
  if (totalPaddingWidth % 2 == 1) {
    if (nhwc) {
      tmp2 = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2) + totalPaddingWidth,
           realInput.shape(3)});
    } else {
      tmp2 = Variable<T>(
          {realInput.shape(0), realInput.shape(1), realInput.shape(2),
           realInput.shape(3) + totalPaddingWidth});
    }
    interface.getDataInterface().zeros(tmp2);
    interface.getBasicInterface().assignTo(realInput, nhwc ? 2 : 3, paddingWidth, tmp2);
    realInput = View<T>(tmp2);
    paddingWidth = 0;
  }

  Variable<T> result = interface.getConvolutionInterface().maxPool2d(
      realInput, nhwc, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight,
      strideWidth);
  return result;
}

template <class T>
Variable<T> max_pool2d_valid(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth) {
  if (input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernelHeight == 0 || kernelWidth == 0) {
    throw InvalidArgumentError("invalid kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().maxPool2d(
      input, nhwc, kernelHeight, kernelWidth, 0, 0, strideHeight, strideWidth);
}

template <class T>
Variable<T> max_pool1d(
    const Tensor<T>& input, bool nwc, unsigned kernel, unsigned padding, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }

  View<T> realInput;
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = max_pool2d(realInput, nwc, 1, kernel, 0, padding, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

template <class T>
Variable<T> max_pool1d_same(const Tensor<T>& input, bool nwc, unsigned kernel, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }

  View<T> realInput;
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = max_pool2d_same(realInput, nwc, 1, kernel, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

/**
 * kernel: [out_channels, in_channels, width], padding = 'valid'
 */
template <class T>
Variable<T> max_pool1d_valid(const Tensor<T>& input, bool nwc, unsigned kernel, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }

  View<T> realInput;
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = max_pool2d_valid(realInput, nwc, 1, kernel, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

template <class T>
Variable<T> avg_pool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  if (input.size() == 0 || input.shape().size() != 4) {
    throw InvalidArgumentError("invalid input");
  }
  if (kernelHeight == 0 || kernelWidth == 0) {
    throw InvalidArgumentError("invalid kernel");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getConvolutionInterface().avgPool2d(
      input, nhwc, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight,
      strideWidth);
}

template <class T>
Variable<T> avg_pool1d(
    const Tensor<T>& input, bool nwc, unsigned kernel, unsigned padding, unsigned stride) {
  if (input.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }

  View<T> realInput;
  if (nwc) {
    realInput = View<T>(input, {input.shape(0), 1, input.shape(1), input.shape(2)});
  } else {
    realInput = View<T>(input, {input.shape(0), input.shape(1), 1, input.shape(2)});
  }

  auto result = avg_pool2d(realInput, nwc, 1, kernel, 0, padding, 1, stride);
  if (nwc) {
    result.reshape({result.shape(0), result.shape(2), result.shape(3)});
  } else {
    result.reshape({result.shape(0), result.shape(1), result.shape(3)});
  }
  return result;
}

template <class T>
Variable<T> local_response_normalization(
    const Tensor<T>& a, size_t axis, T alpha, T beta, T bias, unsigned size) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  unsigned n = 1;
  for (size_t i = 0; i < axis; ++i) {
    n *= a.shape(i);
  }
  unsigned d = 1;
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    d *= a.shape(i);
  }
  View<T> aView(a, {n, a.shape(axis), d});
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> result = interface.getNormalizationInterface().localResponseNormalization(
      aView, alpha, beta, bias, size);
  result.reshape(a.shape());
  return result;
}

template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
    size_t beginNormAxis) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> result =
      interface.getNormalizationInterface().layerNormalize(a, gamma, beta, epsilon, beginNormAxis);
  return result;
}

template <class T>
Variable<T> batch_normalization(
    const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
    const Tensor<T>& mean, const Tensor<T>& var, T epsilon) {
  if (scale.shape().size() > 1 || offset.shape().size() > 1 || mean.shape().size() > 1 ||
      var.shape().size() > 1) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getNormalizationInterface().batchNormalization(
      a, axis, scale, offset, mean, var, epsilon);
}

#define DEFINE_FUNC(type)                                                                          \
  template Variable<type> conv2d(                                                                  \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned paddingHeight,    \
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);                         \
  template Variable<type> conv2d(                                                                  \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned paddingHeight,    \
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,   \
      unsigned dilateWidth);                                                                       \
  template Variable<type> conv2d(                                                                  \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned paddingHeight,    \
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,   \
      unsigned dilateWidth, unsigned group);                                                       \
  template Variable<type> conv2d_same(                                                             \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned strideHeight,     \
      unsigned strideWidth);                                                                       \
  template Variable<type> conv2d_valid(                                                            \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned strideHeight,     \
      unsigned strideWidth);                                                                       \
  template Variable<type> conv1d(                                                                  \
      const Tensor<type>& input, bool nwc, const Tensor<type>& kernel, unsigned padding,           \
      unsigned stride);                                                                            \
  template Variable<type> conv1d_same(                                                             \
      const Tensor<type>& input, bool nwc, const Tensor<type>& kernel, unsigned stride);           \
  template Variable<type> conv1d_valid(                                                            \
      const Tensor<type>& input, bool nwc, const Tensor<type>& kernel, unsigned stride);           \
  template Variable<type> conv2d_backward_data(                                                    \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned paddingHeight,    \
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);                         \
  template Variable<type> conv2d_transpose(                                                        \
      const Tensor<type>& input, bool nhwc, const Tensor<type>& kernel, unsigned outputHeight,     \
      unsigned outputWidth, unsigned strideHeight, unsigned strideWidth);                          \
  template Variable<type> embedding_lookup(                                                        \
      const Tensor<type>& params, const std::vector<unsigned> ids);                                \
  template Variable<type> embedding_lookup(                                                        \
      const Tensor<type>& params, const std::vector<unsigned> ids, Shape idShape);                 \
  template Variable<type> gather(const Tensor<type>& params, const Tensor<uint32_t>& indices);     \
  template Variable<type> gather(const Tensor<type>& params, const Tensor<int32_t>& indices);      \
  template Variable<type> gather(const Tensor<type>& params, const Tensor<uint64_t>& indices);     \
  template Variable<type> gather(const Tensor<type>& params, const Tensor<int64_t>& indices);      \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<uint32_t>& indices, const Tensor<type>& updates, const Shape& shape);           \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<int32_t>& indices, const Tensor<type>& updates, const Shape& shape);            \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<uint64_t>& indices, const Tensor<type>& updates, const Shape& shape);           \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<int64_t>& indices, const Tensor<type>& updates, const Shape& shape);            \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<type>& data, const Tensor<uint32_t>& indices, const Tensor<type>& updates);     \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<type>& data, const Tensor<int32_t>& indices, const Tensor<type>& updates);      \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<type>& data, const Tensor<uint64_t>& indices, const Tensor<type>& updates);     \
  template Variable<type> scatter_nd(                                                              \
      const Tensor<type>& data, const Tensor<int64_t>& indices, const Tensor<type>& updates);      \
  template Variable<type> one_hot(                                                                 \
      const std::vector<unsigned> ids, unsigned depth, type onValue, type offValue);               \
  template Variable<type> one_hot(                                                                 \
      const std::vector<unsigned> ids, Shape idShape, unsigned depth, type onValue,                \
      type offValue);                                                                              \
  template Variable<type> where(                                                                   \
      const Tensor<uint8_t>& condition, const Tensor<type>& x, const Tensor<type>& y);             \
  template Variable<type> where(const Tensor<uint8_t>& condition, type x, const Tensor<type>& y);  \
  template Variable<type> where(const Tensor<uint8_t>& condition, const Tensor<type>& x, type y);  \
  template Variable<type> max_pool2d(                                                              \
      const Tensor<type>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,           \
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth); \
  template Variable<type> max_pool2d_same(                                                         \
      const Tensor<type>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,           \
      unsigned strideHeight, unsigned strideWidth);                                                \
  template Variable<type> max_pool2d_valid(                                                        \
      const Tensor<type>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,           \
      unsigned strideHeight, unsigned strideWidth);                                                \
  template Variable<type> max_pool1d(                                                              \
      const Tensor<type>& input, bool nhwc, unsigned kernel, unsigned padding, unsigned stride);   \
  template Variable<type> max_pool1d_same(                                                         \
      const Tensor<type>& input, bool nhwc, unsigned kernel, unsigned stride);                     \
  template Variable<type> max_pool1d_valid(                                                        \
      const Tensor<type>& input, bool nhwc, unsigned kernel, unsigned stride);                     \
  template Variable<type> avg_pool2d(                                                              \
      const Tensor<type>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,           \
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth); \
  template Variable<type> avg_pool1d(                                                              \
      const Tensor<type>& input, bool nwc, unsigned kernel, unsigned padding, unsigned stride);    \
  template Variable<type> local_response_normalization(                                            \
      const Tensor<type>& a, size_t axis, type alpha, type beta, type bias, unsigned size);        \
  template Variable<type> layer_normalization(                                                     \
      const Tensor<type>& a, const Tensor<type>& gamma, const Tensor<type>& beta, type epsilon,    \
      size_t beginNormAxis);                                                                       \
  template Variable<type> batch_normalization(                                                     \
      const Tensor<type>& a, size_t axis, const Tensor<type>& scale, const Tensor<type>& offset,   \
      const Tensor<type>& mean, const Tensor<type>& var, type epsilon);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace nn
}  // namespace tfcc
