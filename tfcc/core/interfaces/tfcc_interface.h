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

#include <memory>
#include <utility>

namespace tfcc {

template <class T>
class BasicInterface;

template <class T>
class DataInterface;

template <class T>
class BlasInterface;

template <class T>
class ActivationInterface;

template <class T>
class ConvolutionInterface;

template <class T>
class ReduceInterface;

template <class T>
class SegmentInterface;

template <class T>
class QuantizationInterface;

template <class T>
class TransformationInterface;

template <class T>
class ArithmeticInterface;

template <class T>
class CellInterface;

template <class T>
class BatchArithmeticInterface;

template <class T>
class NormalizationInterface;

template <class T>
class GatherInterface;

template <class T>
class ComparisonInterface;

template <class T>
class MinMaxInterface;

template <class T>
class SignalInterface;

template <class T>
class ScatterInterface;

template <class T>
class Interface {
  std::unique_ptr<BasicInterface<T>> _basicInterface;
  std::unique_ptr<DataInterface<T>> _dataInterface;
  std::unique_ptr<BlasInterface<T>> _blasInterface;
  std::unique_ptr<ActivationInterface<T>> _activationInterface;
  std::unique_ptr<ConvolutionInterface<T>> _convolutionInterface;
  std::unique_ptr<ReduceInterface<T>> _reduceInterface;
  std::unique_ptr<SegmentInterface<T>> _segmentInterface;
  std::unique_ptr<QuantizationInterface<T>> _quantizationInterface;
  std::unique_ptr<TransformationInterface<T>> _transformationInterface;
  std::unique_ptr<ArithmeticInterface<T>> _arithmeticInterface;
  std::unique_ptr<CellInterface<T>> _cellInterface;
  std::unique_ptr<BatchArithmeticInterface<T>> _batchArithmeticInterface;
  std::unique_ptr<NormalizationInterface<T>> _normalizationInterface;
  std::unique_ptr<GatherInterface<T>> _gatherInterface;
  std::unique_ptr<ComparisonInterface<T>> _comparisonInterface;
  std::unique_ptr<MinMaxInterface<T>> _minMaxInterface;
  std::unique_ptr<SignalInterface<T>> _signalInterface;
  std::unique_ptr<ScatterInterface<T>> _scatterInterface;

 public:
  Interface();
  Interface(const Interface&) = delete;
  Interface(Interface&&);
  ~Interface();

  Interface& operator=(const Interface&) = delete;
  Interface& operator=(Interface&&);

  BasicInterface<T>& getBasicInterface() { return *_basicInterface; }
  void setBasicInterface(std::unique_ptr<BasicInterface<T>> basicInterface);

  DataInterface<T>& getDataInterface() { return *_dataInterface; }
  void setDataInterface(std::unique_ptr<DataInterface<T>> dataInterface);

  BlasInterface<T>& getBlasInterface() { return *_blasInterface; }
  void setBlasInterface(std::unique_ptr<BlasInterface<T>> blasInterface);

  ActivationInterface<T>& getActivationInterface() { return *_activationInterface; }
  void setActivationInterface(std::unique_ptr<ActivationInterface<T>> activationInterface);

  ConvolutionInterface<T>& getConvolutionInterface() { return *_convolutionInterface; }
  void setConvolutionInterface(std::unique_ptr<ConvolutionInterface<T>> convolutionInterface);

  ReduceInterface<T>& getReduceInterface() { return *_reduceInterface; }
  void setReduceInterface(std::unique_ptr<ReduceInterface<T>> reduceInterface);

  SegmentInterface<T>& getSegmentInterface() { return *_segmentInterface; }
  void setSegmentInterface(std::unique_ptr<SegmentInterface<T>> segmentInterface);

  QuantizationInterface<T>& getQuantizationInterface() { return *_quantizationInterface; }
  void setQuantizationInterface(std::unique_ptr<QuantizationInterface<T>> quantizationInterface);

  TransformationInterface<T>& getTransformationInterface() { return *_transformationInterface; }
  void setTransformationInterface(
      std::unique_ptr<TransformationInterface<T>> transformationInterface);

  ArithmeticInterface<T>& getArithmeticInterface() { return *_arithmeticInterface; }
  void setArithmeticInterface(std::unique_ptr<ArithmeticInterface<T>> arithmeticInterface);

  CellInterface<T>& getCellInterface() { return *_cellInterface; }
  void setCellInterface(std::unique_ptr<CellInterface<T>> cellInterface);

  BatchArithmeticInterface<T>& getBatchArithmeticInterface() { return *_batchArithmeticInterface; }
  void setBatchArithmeticInterface(
      std::unique_ptr<BatchArithmeticInterface<T>> batchArithmeticInterface);

  NormalizationInterface<T>& getNormalizationInterface() { return *_normalizationInterface; }
  void setNormalizationInterface(std::unique_ptr<NormalizationInterface<T>> normalizationInterface);

  GatherInterface<T>& getGatherInterface() { return *_gatherInterface; }
  void setGatherInterface(std::unique_ptr<GatherInterface<T>> gatherInterface);

  ComparisonInterface<T>& getComparisonInterface() { return *_comparisonInterface; }
  void setComparisonInterface(std::unique_ptr<ComparisonInterface<T>> comparisonInterface);

  MinMaxInterface<T>& getMinMaxInterface() { return *_minMaxInterface; }
  void setMinMaxInterface(std::unique_ptr<MinMaxInterface<T>> minMaxInterface);

  SignalInterface<T>& getSignalInterface() { return *_signalInterface; }
  void setSignalInterface(std::unique_ptr<SignalInterface<T>> signalInterface);

  ScatterInterface<T>& getScatterInterface() { return *_scatterInterface; }
  void setScatterInterface(std::unique_ptr<ScatterInterface<T>> scatterInterface);
};

}  // namespace tfcc
