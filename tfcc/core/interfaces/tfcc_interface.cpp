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

#include "tfcc_interface.h"

#include "framework/tfcc_types.h"
#include "interfaces/tfcc_activationinterface.h"
#include "interfaces/tfcc_arithmeticinterface.h"
#include "interfaces/tfcc_basicinterface.h"
#include "interfaces/tfcc_batcharithmeticinterface.h"
#include "interfaces/tfcc_blasinterface.h"
#include "interfaces/tfcc_cellinterface.h"
#include "interfaces/tfcc_comparisoninterface.h"
#include "interfaces/tfcc_convolutioninterface.h"
#include "interfaces/tfcc_datainterface.h"
#include "interfaces/tfcc_gatherinterface.h"
#include "interfaces/tfcc_minmaxinterface.h"
#include "interfaces/tfcc_normalizationinterface.h"
#include "interfaces/tfcc_quantizationinterface.h"
#include "interfaces/tfcc_reduceinterface.h"
#include "interfaces/tfcc_scatterinterface.h"
#include "interfaces/tfcc_segmentinterface.h"
#include "interfaces/tfcc_signalinterface.h"
#include "interfaces/tfcc_transformationinterface.h"

namespace tfcc {

template <class T>
Interface<T>::Interface() {
  _basicInterface = std::unique_ptr<BasicInterface<T>>(new BasicInterface<T>);
  _dataInterface = std::unique_ptr<DataInterface<T>>(new DataInterface<T>);
  _blasInterface = std::unique_ptr<BlasInterface<T>>(new BlasInterface<T>);
  _activationInterface = std::unique_ptr<ActivationInterface<T>>(new ActivationInterface<T>);
  _convolutionInterface = std::unique_ptr<ConvolutionInterface<T>>(new ConvolutionInterface<T>);
  _reduceInterface = std::unique_ptr<ReduceInterface<T>>(new ReduceInterface<T>);
  _segmentInterface = std::unique_ptr<SegmentInterface<T>>(new SegmentInterface<T>);
  _quantizationInterface = std::unique_ptr<QuantizationInterface<T>>(new QuantizationInterface<T>);
  _transformationInterface =
      std::unique_ptr<TransformationInterface<T>>(new TransformationInterface<T>);
  _arithmeticInterface = std::unique_ptr<ArithmeticInterface<T>>(new ArithmeticInterface<T>);
  _cellInterface = std::unique_ptr<CellInterface<T>>(new CellInterface<T>);
  _batchArithmeticInterface =
      std::unique_ptr<BatchArithmeticInterface<T>>(new BatchArithmeticInterface<T>());
  _normalizationInterface =
      std::unique_ptr<NormalizationInterface<T>>(new NormalizationInterface<T>());
  _gatherInterface = std::unique_ptr<GatherInterface<T>>(new GatherInterface<T>());
  _comparisonInterface = std::unique_ptr<ComparisonInterface<T>>(new ComparisonInterface<T>());
  _minMaxInterface = std::unique_ptr<MinMaxInterface<T>>(new MinMaxInterface<T>());
  _signalInterface = std::unique_ptr<SignalInterface<T>>(new SignalInterface<T>());
  _scatterInterface = std::unique_ptr<ScatterInterface<T>>(new ScatterInterface<T>());
}

template <class T>
Interface<T>::Interface(Interface&&) = default;

template <class T>
Interface<T>& Interface<T>::operator=(Interface&&) = default;

template <class T>
Interface<T>::~Interface() {}

template <class T>
void Interface<T>::setBasicInterface(std::unique_ptr<BasicInterface<T>> basicInterface) {
  _basicInterface = std::move(basicInterface);
}

template <class T>
void Interface<T>::setDataInterface(std::unique_ptr<DataInterface<T>> dataInterface) {
  _dataInterface = std::move(dataInterface);
}

template <class T>
void Interface<T>::setBlasInterface(std::unique_ptr<BlasInterface<T>> blasInterface) {
  _blasInterface = std::move(blasInterface);
}

template <class T>
void Interface<T>::setActivationInterface(
    std::unique_ptr<ActivationInterface<T>> activationInterface) {
  _activationInterface = std::move(activationInterface);
}

template <class T>
void Interface<T>::setConvolutionInterface(
    std::unique_ptr<ConvolutionInterface<T>> convolutionInterface) {
  _convolutionInterface = std::move(convolutionInterface);
}

template <class T>
void Interface<T>::setReduceInterface(std::unique_ptr<ReduceInterface<T>> reduceInterface) {
  _reduceInterface = std::move(reduceInterface);
}

template <class T>
void Interface<T>::setSegmentInterface(std::unique_ptr<SegmentInterface<T>> segmentInterface) {
  _segmentInterface = std::move(segmentInterface);
}

template <class T>
void Interface<T>::setQuantizationInterface(
    std::unique_ptr<QuantizationInterface<T>> quantizationInterface) {
  _quantizationInterface = std::move(quantizationInterface);
}

template <class T>
void Interface<T>::setTransformationInterface(
    std::unique_ptr<TransformationInterface<T>> transformationInterface) {
  _transformationInterface = std::move(transformationInterface);
}

template <class T>
void Interface<T>::setArithmeticInterface(
    std::unique_ptr<ArithmeticInterface<T>> arithmeticInterface) {
  _arithmeticInterface = std::move(arithmeticInterface);
}

template <class T>
void Interface<T>::setCellInterface(std::unique_ptr<CellInterface<T>> cellInterface) {
  _cellInterface = std::move(cellInterface);
}

template <class T>
void Interface<T>::setBatchArithmeticInterface(
    std::unique_ptr<BatchArithmeticInterface<T>> batchArithmeticInterface) {
  _batchArithmeticInterface = std::move(batchArithmeticInterface);
}

template <class T>
void Interface<T>::setNormalizationInterface(
    std::unique_ptr<NormalizationInterface<T>> normalizationInterface) {
  _normalizationInterface = std::move(normalizationInterface);
}

template <class T>
void Interface<T>::setGatherInterface(std::unique_ptr<GatherInterface<T>> gatherInterface) {
  _gatherInterface = std::move(gatherInterface);
}

template <class T>
void Interface<T>::setComparisonInterface(
    std::unique_ptr<ComparisonInterface<T>> comparisonInterface) {
  _comparisonInterface = std::move(comparisonInterface);
}

template <class T>
void Interface<T>::setMinMaxInterface(std::unique_ptr<MinMaxInterface<T>> minMaxInterface) {
  _minMaxInterface = std::move(minMaxInterface);
}

template <class T>
void Interface<T>::setSignalInterface(std::unique_ptr<SignalInterface<T>> signalInterface) {
  _signalInterface = std::move(signalInterface);
}

template <class T>
void Interface<T>::setScatterInterface(std::unique_ptr<ScatterInterface<T>> scatterInterface) {
  _scatterInterface = std::move(scatterInterface);
}

#define DEFINE_FUNC(type) template class Interface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
