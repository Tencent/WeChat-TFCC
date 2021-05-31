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

#include "tfcc_cudainterface.h"

#include <memory>

#include "framework/tfcc_types.h"
#include "interfaces/tfcc_cudaactivationinterface.h"
#include "interfaces/tfcc_cudaarithmeticinterface.h"
#include "interfaces/tfcc_cudabasicinterface.h"
#include "interfaces/tfcc_cudabatcharithmeticinterface.h"
#include "interfaces/tfcc_cudablasinterface.h"
#include "interfaces/tfcc_cudacellinterface.h"
#include "interfaces/tfcc_cudacomparisoninterface.h"
#include "interfaces/tfcc_cudaconvolutioninterface.h"
#include "interfaces/tfcc_cudadatainterface.h"
#include "interfaces/tfcc_cudagatherinterface.h"
#include "interfaces/tfcc_cudaminmaxinterface.h"
#include "interfaces/tfcc_cudanormalizationinterface.h"
#include "interfaces/tfcc_cudareduceinterface.h"
#include "interfaces/tfcc_cudasegmentinterface.h"
#include "interfaces/tfcc_cudatransformationinterface.h"

namespace tfcc {

template <class T>
Interface<T> get_cuda_interface(T, const CUDADeviceProperty& property) {
  Interface<T> interface;
  interface.setBasicInterface(
      std::unique_ptr<BasicInterface<T>>(new CUDABasicInterface<T>(property)));
  interface.setDataInterface(std::unique_ptr<DataInterface<T>>(new CUDADataInterface<T>(property)));
  interface.setBlasInterface(std::unique_ptr<BlasInterface<T>>(new CUDABlasInterface<T>(property)));
  interface.setActivationInterface(
      std::unique_ptr<ActivationInterface<T>>(new CUDAActivationInterface<T>(property)));
  interface.setConvolutionInterface(
      std::unique_ptr<ConvolutionInterface<T>>(new CUDAConvolutionInterface<T>(property)));
  interface.setReduceInterface(
      std::unique_ptr<ReduceInterface<T>>(new CUDAReduceInterface<T>(property)));
  interface.setTransformationInterface(
      std::unique_ptr<TransformationInterface<T>>(new CUDATransformationInterface<T>(property)));
  interface.setArithmeticInterface(
      std::unique_ptr<ArithmeticInterface<T>>(new CUDAArithmeticInterface<T>(property)));
  interface.setSegmentInterface(
      std::unique_ptr<SegmentInterface<T>>(new CUDASegmentInterface<T>(property)));
  interface.setCellInterface(std::unique_ptr<CellInterface<T>>(new CUDACellInterface<T>(property)));
  interface.setBatchArithmeticInterface(
      std::unique_ptr<BatchArithmeticInterface<T>>(new CUDABatchArithmeticInterface<T>(property)));
  interface.setNormalizationInterface(
      std::unique_ptr<NormalizationInterface<T>>(new CUDANormalizationInterface<T>(property)));
  interface.setGatherInterface(
      std::unique_ptr<GatherInterface<T>>(new CUDAGatherInterface<T>(property)));
  interface.setComparisonInterface(
      std::unique_ptr<ComparisonInterface<T>>(new CUDAComparisonInterface<T>(property)));
  interface.setMinMaxInterface(
      std::unique_ptr<MinMaxInterface<T>>(new CUDAMinMaxInterface<T>(property)));

  return interface;
}

#define DEFINE_FUNC(type) \
  template Interface<type> get_cuda_interface(type, const CUDADeviceProperty& property);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
