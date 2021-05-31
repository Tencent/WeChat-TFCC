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

#include "tfcc_mklinterface.h"

#include <type_traits>

#include "tfcc_mklactivationinterface.h"
#include "tfcc_mklarithmeticinterface.h"
#include "tfcc_mklbasicinterface.h"
#include "tfcc_mklbatcharithmeticinterface.h"
#include "tfcc_mklblasinterface.h"
#include "tfcc_mklcellinterface.h"
#include "tfcc_mklcomparisoninterface.h"
#include "tfcc_mklconvolutioninterface.h"
#include "tfcc_mkldatainterface.h"
#include "tfcc_mklgatherinterface.h"
#include "tfcc_mklminmaxinterface.h"
#include "tfcc_mklnormalizationinterface.h"
#include "tfcc_mklquantizationinterface.h"
#include "tfcc_mklreduceinterface.h"
#include "tfcc_mklscatterinterface.h"
#include "tfcc_mklsegmentinterface.h"
#include "tfcc_mklsignalinterface.h"
#include "tfcc_mkltransformationinterface.h"

#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
static inline void makeMKLInterface(T&) {}

template <class T>
static inline typename std::enable_if<TypeInfo<T>::quantizationType, void>::type makeMKLInterface(
    Interface<T>& interface) {
  interface.setQuantizationInterface(
      std::unique_ptr<QuantizationInterface<T>>(new MKLQuantizationInterface<T>));
}

template <class T>
Interface<T> get_mkl_interface(T, const MKLDevice& device) {
  Interface<T> interface;
  interface.setBasicInterface(std::unique_ptr<BasicInterface<T>>(new MKLBasicInterface<T>()));
  interface.setDataInterface(std::unique_ptr<DataInterface<T>>(new MKLDataInterface<T>()));
  interface.setBlasInterface(std::unique_ptr<BlasInterface<T>>(new MKLBlasInterface<T>(device)));
  interface.setActivationInterface(
      std::unique_ptr<ActivationInterface<T>>(new MKLActivationInterface<T>()));
  interface.setConvolutionInterface(
      std::unique_ptr<ConvolutionInterface<T>>(new MKLConvolutionInterface<T>()));
  interface.setReduceInterface(std::unique_ptr<ReduceInterface<T>>(new MKLReduceInterface<T>()));
  interface.setSegmentInterface(std::unique_ptr<SegmentInterface<T>>(new MKLSegmentInterface<T>()));
  interface.setTransformationInterface(
      std::unique_ptr<TransformationInterface<T>>(new MKLTransformationInterface<T>()));
  interface.setArithmeticInterface(
      std::unique_ptr<ArithmeticInterface<T>>(new MKLArithmeticInterface<T>()));
  interface.setCellInterface(std::unique_ptr<CellInterface<T>>(new MKLCellInterface<T>()));
  interface.setBatchArithmeticInterface(
      std::unique_ptr<BatchArithmeticInterface<T>>(new MKLBatchArithmeticInterface<T>()));
  interface.setNormalizationInterface(
      std::unique_ptr<NormalizationInterface<T>>(new MKLNormalizationInterface<T>()));
  interface.setGatherInterface(std::unique_ptr<GatherInterface<T>>(new MKLGatherInterface<T>()));
  interface.setComparisonInterface(
      std::unique_ptr<ComparisonInterface<T>>(new MKLComparisonInterface<T>()));
  interface.setMinMaxInterface(std::unique_ptr<MinMaxInterface<T>>(new MKLMinMaxInterface<T>()));
  interface.setSignalInterface(std::unique_ptr<SignalInterface<T>>(new MKLSignalInterface<T>()));
  interface.setScatterInterface(std::unique_ptr<ScatterInterface<T>>(new MKLScatterInterface<T>()));

  // quantization interface
  makeMKLInterface(interface);

  return interface;
}

template <class T>
Interface<Complex<T>> get_mkl_complex_interface(Complex<T>, const MKLDevice& device) {
  Interface<Complex<T>> interface;
  interface.setDataInterface(
      std::unique_ptr<DataInterface<Complex<T>>>(new MKLDataInterface<Complex<T>>()));
  interface.setArithmeticInterface(
      std::unique_ptr<ArithmeticInterface<Complex<T>>>(new MKLArithmeticInterface<Complex<T>>()));

  // quantization interface
  makeMKLInterface(interface);

  return interface;
}

#define DEFINE_FUNC(type) template Interface<type> get_mkl_interface(type, const MKLDevice& device);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

template Interface<Complex<float>> get_mkl_complex_interface(
    Complex<float>, const MKLDevice& device);
template Interface<Complex<double>> get_mkl_complex_interface(
    Complex<double>, const MKLDevice& device);

}  // namespace tfcc
