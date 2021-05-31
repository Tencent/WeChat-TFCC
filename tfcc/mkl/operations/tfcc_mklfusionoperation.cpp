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

#include "tfcc_mklfusionoperation.h"

#include <omp.h>
#include <iostream>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_shape.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "utils/tfcc_debugutils.h"

#include "fusionop/tfcc_mklfusionopdynamicshape.h"
#include "fusionop/tfcc_mklfusionopfixedshape.h"

namespace tfcc {
namespace fusionop {

static tfcc::Shape _get_broadcast_shape(const tfcc::Shape& s1, const tfcc::Shape& s2) {
  if (s1.size() == 0 || s2.size() == 0) {
    throw InvalidArgumentError("invalid tensor shape");
  }

  std::vector<unsigned> resultS;
  resultS = s1.size() > s2.size() ? s1.toVector() : s2.toVector();
  for (size_t i = 0; i < s1.size() && i < s2.size(); ++i) {
    unsigned l1 = s1[s1.size() - 1 - i];
    unsigned l2 = s2[s2.size() - 1 - i];
    if (l1 != l2 && l1 != 1 && l2 != 1) {
      throw InvalidArgumentError("broadcast failed");
    }
    resultS[resultS.size() - 1 - i] = std::max(l1, l2);
  }
  return tfcc::Shape(resultS);
}

static tfcc::Shape _get_broadcast_shape(const std::vector<tfcc::Shape>& shapes) {
  assert(!shapes.empty());
  tfcc::Shape resultShape = shapes[0];
  for (size_t i = 1; i < shapes.size(); ++i) {
    resultShape = _get_broadcast_shape(resultShape, shapes[i]);
  }
  return resultShape;
}

FusionHandler getFixedShapeFusionHandler(
    const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
    const std::vector<std::vector<bool>>& broadcastMarks) {
  std::vector<std::vector<bool>> realBroadcastMarks;
  std::vector<unsigned> realResultShape;
  size_t index = 0;
  for (; index < resultShape.size() - 1; ++index) {
    if (resultShape[index] > 1) {
      break;
    }
  }
  if (index > 0) {
    realResultShape.insert(realResultShape.begin(), resultShape.begin() + index, resultShape.end());
    for (auto& v : broadcastMarks) {
      realBroadcastMarks.push_back(std::vector<bool>(v.begin() + index, v.end()));
    }
  } else {
    realResultShape = resultShape;
    realBroadcastMarks = broadcastMarks;
  }

  if (realResultShape.size() > 1) {
    for (auto& v : realBroadcastMarks) {
      v.erase(v.begin());
    }
    realResultShape.erase(realResultShape.begin());
  }

  thread_local std::map<
      std::tuple<std::vector<OperationType>, std::vector<std::vector<bool>>>,
      std::unique_ptr<FusionOpFixedShape>>
      fusionOpFixedShapeHandlers;
  auto key = std::make_tuple(opTypes, realBroadcastMarks);
  auto it = fusionOpFixedShapeHandlers.find(key);
  if (it == fusionOpFixedShapeHandlers.end()) {
    auto h = new FusionOpFixedShape(opTypes, realResultShape, realBroadcastMarks);
    fusionOpFixedShapeHandlers.insert(std::make_pair(key, std::unique_ptr<FusionOpFixedShape>(h)));
    return FusionHandler(h, nullptr);
  } else {
    return FusionHandler(it->second.get(), nullptr);
  }
}

static void fixedShapeFusionMP(
    FusionOpFixedShape* op, const std::vector<const tfcc::Tensor<float>*> inputs,
    const std::vector<bool>& stepByBroadcastMarks, float* resultData, unsigned resultSize,
    std::vector<unsigned> resultShape) {
  std::vector<unsigned> skips(inputs.size(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!stepByBroadcastMarks[i]) {
      size_t begin = inputs[i]->shape().size() > resultShape.size()
                         ? inputs[i]->shape().size() - resultShape.size()
                         : 0;
      skips[i] = inputs[i]->size() / inputs[i]->shape(begin);
    }
  }
  unsigned batchSize = resultShape[0];
  unsigned len = resultSize / batchSize;
#pragma omp parallel for
  for (unsigned b = 0; b < batchSize; ++b) {
    unsigned step = b * len;
    std::vector<float*> inp(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      inp[i] = const_cast<float*>(inputs[i]->data() + b * skips[i]);
    }
    op->process(inp.data(), resultData + step);
  }
}

tfcc::Variable<float> fixedShapeFusion(
    const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
    const std::vector<std::vector<bool>>& broadcastMarks,
    const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler) {
  if (!handler.fixedHandler) {
    throw InvalidArgumentError("invalid handler");
  }
  tfcc::Variable<float> result(resultShape);
  size_t index = 0;
  for (; index < resultShape.size() - 1; ++index) {
    if (resultShape[index] > 1) {
      break;
    }
  }
  std::vector<unsigned> realResultShape;
  if (index == 0) {
    realResultShape = resultShape;
  } else {
    realResultShape.insert(realResultShape.begin(), resultShape.begin() + index, resultShape.end());
  }
  if (realResultShape.size() == 1) {
    mkl_async_wrapper(
        "fusionop_fixed_shape_process",
        [](FusionOpFixedShape* op, const std::vector<const tfcc::Tensor<float>*> inputs,
           float* resultData) {
          std::vector<float*> inp(inputs.size());
          for (size_t i = 0; i < inputs.size(); ++i) {
            inp[i] = const_cast<float*>(inputs[i]->data());
          }
          op->process(inp.data(), resultData);
        },
        handler.fixedHandler, inputs, result.data());
  } else {
    std::vector<bool> stepByBroadcastMarks;
    for (auto& v : broadcastMarks) {
      stepByBroadcastMarks.push_back(v[index]);
    }
    mkl_async_wrapper(
        "fusionop_fixed_shape_process", &fixedShapeFusionMP, handler.fixedHandler, inputs,
        stepByBroadcastMarks, result.data(), result.size(), realResultShape);
  }
  return result;
}

FusionHandler getDynamicShapeFusionHandler(const std::vector<OperationType>& opTypes) {
  thread_local std::map<
      std::vector<OperationType>,
      std::map<std::vector<std::vector<bool>>, std::unique_ptr<FusionOpDynamicShape>>>
      fusionOpDynamicShapeHandlers;
  auto it = fusionOpDynamicShapeHandlers.find(opTypes);
  if (it == fusionOpDynamicShapeHandlers.end()) {
    auto h = fusionOpDynamicShapeHandlers.insert(std::make_pair(
        opTypes,
        std::map<std::vector<std::vector<bool>>, std::unique_ptr<FusionOpDynamicShape>>()));
    return FusionHandler(nullptr, &h.first->second);
  } else {
    return FusionHandler(nullptr, &it->second);
  }
}

static void dynamicShapeFusionMP(
    FusionOpDynamicShape* op, const std::vector<const tfcc::Tensor<float>*>& inputs,
    const std::vector<bool>& stepByBroadcastMarks, float* resultData, unsigned resultSize,
    std::vector<unsigned> resultShape) {
  unsigned batchSize = resultShape[0];
  unsigned threadSize = omp_get_max_threads();
  unsigned everyLen = resultSize / batchSize;
  std::vector<unsigned> skips(inputs.size(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!stepByBroadcastMarks[i]) {
      size_t begin = inputs[i]->shape().size() > resultShape.size()
                         ? inputs[i]->shape().size() - resultShape.size()
                         : 0;
      skips[i] = inputs[i]->size() / inputs[i]->shape(begin);
    }
  }
  if (batchSize > threadSize) {
    unsigned every = (batchSize + threadSize - 1) / threadSize;
    unsigned len = everyLen * every;
    resultShape[0] = every;
    std::vector<std::vector<unsigned>> resultShapes(threadSize, resultShape);
    if (batchSize % threadSize != 0) {
      resultShapes[threadSize - 1][0] = every * threadSize - batchSize;
    }
    unsigned step = 0;
    std::vector<std::vector<const float*>> inps(
        threadSize, std::vector<const float*>(inputs.size()));
    for (unsigned b = 0; b < threadSize; ++b) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        inps[b][i] = inputs[i]->data() + step * skips[i];
      }
      step += resultShapes[b][0];
    }

#pragma omp parallel for
    for (unsigned b = 0; b < threadSize; ++b) {
      op->process(inps[b].data(), resultData + b * len, resultShapes[b]);
    }
  } else {
    resultShape[0] = 1;
#pragma omp parallel for
    for (unsigned b = 0; b < batchSize; ++b) {
      unsigned step = b * everyLen;
      std::vector<const float*> inp(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        inp[i] = inputs[i]->data() + b * skips[i];
      }
      op->process(inp.data(), resultData + step, resultShape);
    }
  }
}

tfcc::Variable<float> dynamicShapeFusion(
    const std::vector<OperationType>& opTypes,
    const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler) {
  auto ops = handler.dynamicHandlers;
  if (!ops) {
    throw InvalidArgumentError("invalid handler");
  }

  std::vector<tfcc::Shape> shapes;
  for (auto& i : inputs) {
    shapes.push_back(i->shape());
  }

  std::vector<unsigned> resultShape = _get_broadcast_shape(shapes).toVector();
  std::vector<std::vector<bool>> broadcastMarks(inputs.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    assert(resultShape.size() >= shapes[i].size());
    size_t len = resultShape.size() - shapes[i].size();
    if (len > 0) {
      broadcastMarks[i].insert(broadcastMarks[i].begin(), len, true);
    }

    for (size_t j = 0; j < shapes[i].size(); ++j) {
      if (shapes[i][j] <= 1) {
        broadcastMarks[i].push_back(true);
      } else {
        broadcastMarks[i].push_back(false);
      }
    }
  }

  // auto a = getFixedShapeFusionHandler(opTypes, resultShape, broadcastMarks);
  // return fixedShapeFusion(opTypes, resultShape, broadcastMarks, inputs, a);

  tfcc::Variable<float> result(resultShape);
  size_t index = 0;
  for (; index < resultShape.size() - 1; ++index) {
    if (resultShape[index] > 1) {
      break;
    }
  }
  if (index > 0) {
    resultShape.erase(resultShape.begin(), resultShape.begin() + index);
    for (auto& v : broadcastMarks) {
      v.erase(v.begin(), v.begin() + index);
    }
  }
  FusionOpDynamicShape* op = nullptr;
  auto it = ops->find(broadcastMarks);
  if (it == ops->end()) {
    op = ops->insert(std::make_pair(
                         broadcastMarks, std::unique_ptr<FusionOpDynamicShape>(
                                             new FusionOpDynamicShape(opTypes, broadcastMarks))))
             .first->second.get();
  } else {
    op = it->second.get();
  }

  if (resultShape.size() == 1) {
    mkl_async_wrapper(
        "fusionop_dynamic_shape_process",
        [](FusionOpDynamicShape* op, const std::vector<const tfcc::Tensor<float>*>& inputs,
           float* resultData, std::vector<unsigned> resultShape) {
          std::vector<float*> inp(inputs.size());
          for (size_t i = 0; i < inputs.size(); ++i) {
            inp[i] = const_cast<float*>(inputs[i]->data());
          }
          op->process(inp.data(), resultData, resultShape);
        },
        op, inputs, result.data(), resultShape);
  } else {
    std::vector<bool> stepByBroadcastMarks;
    for (auto& v : broadcastMarks) {
      stepByBroadcastMarks.push_back(v[0]);
    }
    mkl_async_wrapper(
        "fusionop_dynamic_shape_process", &dynamicShapeFusionMP, op, inputs, stepByBroadcastMarks,
        result.data(), result.size(), resultShape);
  }
  return result;
}

}  // namespace fusionop
}  // namespace tfcc