/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef OPERATOR_INCLUDE_ADD_H_
#define OPERATOR_INCLUDE_ADD_H_

#include <memory>
#include <vector>
#include "operator/include/operator.h"

namespace easydl {

template <typename T>
class CPUAddOp : public CPUOperator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 public:
  virtual void operator()(const vector<TensorPtr>&);
  virtual bool check(const vector<TensorPtr>&);
  virtual void reshape(const vector<TensorPtr>&);
};

template <typename T>
class GPUAddOp : public GPUOperator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 public:
  virtual void operator()(const vector<TensorPtr>&);
  virtual bool check(const vector<TensorPtr>&);
  virtual void reshape(const vector<TensorPtr>&);
};

}  // namespace easydl

#endif  // OPERATOR_INCLUDE_ADD_H_
