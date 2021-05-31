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

#include <vector>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace nn {

/**
 * Calculate 2D convolution.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, height, width].
 * @param paddingHeight Padding count.
 * @param paddingWidth Padding count.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

/**
 * 2D convolution with dilation
 * if dilateHeight == dilateWidth == 1, means no dilation in the corresponding dimension
 */
template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth);

/**
 * 2D convolution with dilation with group
 */
template <class T>
Variable<T> conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group);

/**
 * Calculate 2D convolution with same padding.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, height, width].
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> conv2d_same(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned strideHeight,
    unsigned strideWidth);

/**
 * Calculate 2D convolution with valid padding.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, height, width].
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> conv2d_valid(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned strideHeight,
    unsigned strideWidth);

/**
 * Calculate 1D convolution.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, width].
 * @param padding padding count.
 * @param stride Stride count.
 */
template <class T>
Variable<T> conv1d(
    const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned padding, unsigned stride);

/**
 * Calculate 1D convolution with same padding.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, width].
 * @param stride Stride count.
 */
template <class T>
Variable<T> conv1d_same(const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned stride);

/**
 * Calculate 1D convolution with same padding.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel A convolution kernel with shape [out_channels, in_channels, width].
 * @param stride Stride count.
 */
template <class T>
Variable<T> conv1d_valid(
    const Tensor<T>& input, bool nwc, const Tensor<T>& kernel, unsigned stride);

/**
 * Calculate 2D convolution backward data.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernel A convolution kernel with shape [in_channels, out_channels, height,
 * width](in_channels means the channels of param input).
 * @param paddingHeight Padding count.
 * @param paddingWidth Padding count.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> conv2d_backward_data(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

/**
 * Calculate 2D convolution transpose.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernel A convolution kernel with shape [in_channels, out_channels, height,
 * width](in_channels means the channels of param input).
 * @param outputHeight Output height.
 * @param outputWidth Output width.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> conv2d_transpose(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned outputHeight,
    unsigned outputWidth, unsigned strideHeight, unsigned strideWidth);

/**
 * Looks up ids in a list of embedding tensors.
 * @param params A single tensor representing the complete embedding tensor.
 * @param ids A vector containing the ids to be looked up in params.
 * @return A Variable has shape ids.size() + params.shape()[1:]
 */
template <class T>
Variable<T> embedding_lookup(const Tensor<T>& params, const std::vector<unsigned> ids);

/**
 * Looks up ids in a list of embedding tensors.
 * @param params A single tensor representing the complete embedding tensor.
 * @param ids A vector containing the ids to be looked up in params.
 * @param idShape The id's shape.
 * @return A Variable has shape idShape + params.shape()[1:]
 */
template <class T>
Variable<T> embedding_lookup(
    const Tensor<T>& params, const std::vector<unsigned> ids, Shape idShape);

/**
 * Gather slices from params axis 0 according to indices.
 * @param params The Tensor from which to gather values.
 * @param indices The index Tensor.
 * @return A Tensor.
 */
template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<uint32_t>& indices);

/**
 * @overload
 */
template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<int32_t>& indices);

/**
 * @overload
 */
template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<uint64_t>& indices);

/**
 * @overload
 */
template <class T>
Variable<T> gather(const Tensor<T>& params, const Tensor<int64_t>& indices);

/**
 * Scatter updates into a new tensor according to indices.
 * @param indices A Tensor. Index tensor.
 * @param updates A Tensor. Updates to scatter into output.
 * @param shape A vector. Must have the same type as indices. 1-D. The shape of the resulting
 * tensor.
 * @return A Tensor.
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<uint32_t>& indices, const Tensor<T>& updates, const Shape& shape);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<int32_t>& indices, const Tensor<T>& updates, const Shape& shape);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<uint64_t>& indices, const Tensor<T>& updates, const Shape& shape);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<int64_t>& indices, const Tensor<T>& updates, const Shape& shape);

/**
 * Scatter updates into a new tensor according to indices.
 * @param data A Tensor. Base tensor.
 * @param indices A Tensor. Index tensor.
 * @param updates A Tensor. Updates to scatter into output.
 * @return A Tensor.
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<uint32_t>& indices, const Tensor<T>& updates);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<int32_t>& indices, const Tensor<T>& updates);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<uint64_t>& indices, const Tensor<T>& updates);

/**
 * @overload
 */
template <class T>
Variable<T> scatter_nd(
    const Tensor<T>& data, const Tensor<int64_t>& indices, const Tensor<T>& updates);

/**
 * Create a one-hot variable.
 * @param ids A vector containing the ids to be looked up in params.
 * @param depth A scalar defining the depth of the one hot dimension.
 * @param onValue A scalar defining the value to fill in output when indices[j] = i.
 * @param offValue A scalar defining the value to fill in output when indices[j] != i.
 * @return A Variable has shape [ids.size(), depth]
 */
template <class T>
Variable<T> one_hot(const std::vector<unsigned> ids, unsigned depth, T onValue, T offValue);

/**
 * Looks up ids in a list of embedding tensors.
 * @param ids A vector containing the ids to be looked up in params.
 * @param idShape The id's shape.
 * @param depth A scalar defining the depth of the one hot dimension.
 * @param onValue A scalar defining the value to fill in output when indices[j] = i.
 * @param offValue A scalar defining the value to fill in output when indices[j] != i.
 * @return A Variable has shape idShape + [1]
 */
template <class T>
Variable<T> one_hot(
    const std::vector<unsigned> ids, Shape idShape, unsigned depth, T onValue, T offValue);

/**
 * Return the elements where condition is True (multiplexing x and y). Where behaves like
 * numpy.where .
 * @param condition A uint8_t tensor.
 * @param x A tensor.
 * @param y A tensor.
 * @return A tensor.
 */
template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y);

/**
 * Return the elements where condition is True (multiplexing x and y). Where behaves like
 * numpy.where .
 * @param condition A uint8_t tensor.
 * @param x A value.
 * @param y A tensor.
 * @return A tensor.
 */
template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y);

/**
 * Return the elements where condition is True (multiplexing x and y). Where behaves like
 * numpy.where .
 * @param condition A uint8_t tensor.
 * @param x A value.
 * @param y A tensor.
 * @return A tensor.
 */
template <class T>
Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y);

/**
 * Calculate 2D max pooling.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernelHeight Height of pooling kernel.
 * @param kernelWidth Width of pooling kernel.
 * @param paddingHeight Padding count.
 * @param paddingWidth Padding count.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> max_pool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

/**
 * Calculate 2D max pooling with same padding.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernelHeight Height of pooling kernel.
 * @param kernelWidth Width of pooling kernel.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> max_pool2d_same(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth);

/**
 * Calculate 2D max pooling with valid padding.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernelHeight Height of pooling kernel.
 * @param kernelWidth Width of pooling kernel.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> max_pool2d_valid(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth);

/**
 * Calculate 1D max pooling.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel Size of pooling kernel.
 * @param padding Padding count.
 * @param stride Stride count.
 */
template <class T>
Variable<T> max_pool1d(
    const Tensor<T>& input, bool nwc, unsigned kernel, unsigned padding, unsigned stride);

/**
 * Calculate 1D max pooling with same padding.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel Size of pooling kernel.
 * @param stride Stride count.
 */
template <class T>
Variable<T> max_pool1d_same(const Tensor<T>& input, bool nwc, unsigned kernel, unsigned stride);

/**
 * Calculate 1D max pooling with valid padding.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel Size of pooling kernel.
 * @param stride Stride count.
 */
template <class T>
Variable<T> max_pool1d_valid(const Tensor<T>& input, bool nwc, unsigned kernel, unsigned stride);

/**
 * Calculate 2D avg pooling.
 * @param input A tensor.
 * @param nhwc The input format.
 * @param kernelHeight Height of pooling kernel.
 * @param kernelWidth Width of pooling kernel.
 * @param paddingHeight Padding count.
 * @param paddingWidth Padding count.
 * @param strideHeight Stride count.
 * @param strideWidth Stride count.
 */
template <class T>
Variable<T> avg_pool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

/**
 * Calculate 1D avg pooling.
 * @param input A tensor.
 * @param nwc The input format.
 * @param kernel Size of pooling kernel.
 * @param padding Padding count.
 * @param stride Stride count.
 */
template <class T>
Variable<T> avg_pool1d(
    const Tensor<T>& input, bool nwc, unsigned kernel, unsigned padding, unsigned stride);

/**
 * Local Response Normalization proposed in the AlexNet paper.
 * It normalizes over local input regions.
 * The local region is defined across the channels.
 * For an element X[n1, ..., nj, c, d1, ..., dk] in a tensor of shape (N1, ..., Nj x C x D1 x D2,
 * ..., Dk), its region is {X[n1, ..., nj, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i
 * <= min(C - 1, c + ceil((size - 1) / 2))}. square_sum[n1, ..., nj, c, d1, ..., dk] = sum(X[n1,
 * ..., nj, i, d1, ..., dk] ^ 2), where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c +
 * ceil((size - 1) / 2)). Y[n1, ..., nj, c, d1, ..., dk] = X[n1, ..., nj, c, d1, ..., dk] / (bias +
 * alpha / size * square_sum[n1, ..., nj, c, d1, ..., dk] ) ^ beta
 * @param a A tensor.
 * @param axis The dimension of channel.
 * @param alpha Scaling parameter.
 * @param beta The exponent.
 * @param bias The offset.
 * @param size The number of channels to sum over.
 * @return A tensor.
 */
template <class T>
Variable<T> local_response_normalization(
    const Tensor<T>& a, size_t axis, T alpha, T beta, T bias, unsigned size);

/**
 * Layer normalization
 */
template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
    size_t beginNormAxis);

template <class T>
Variable<T> batch_normalization(
    const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
    const Tensor<T>& mean, const Tensor<T>& var, T epsilon);

}  // namespace nn
}  // namespace tfcc
