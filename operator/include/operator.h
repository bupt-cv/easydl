/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef OPERATOR_INCLUDE_OPERATOR_H_
#define OPERATOR_INCLUDE_OPERATOR_H_

#include <memory>
#include <vector>
#include "tensor/include/tensor.h"

namespace easydl {

using std::vector;

template <typename T>
class Operator {
 public: 
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
  virtual void operator()(vector<TensorPtr>&) {} 
 protected:
  virtual bool check(const vector<TensorPtr>&) {}
  virtual void reshape(vector<TensorPtr>&) {}
};

template <typename T>
class GPUOperator : public Operator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 protected:
  virtual bool check(const vector<TensorPtr>& ts) {
    bool flag = true;
    for (size_t i = 0; i < ts.size(); ++i) {
      flag &= (ts[i]->type() == "GPUTensor");
    }
    return flag;
  }
};

template <typename T>
class CPUOperator : public Operator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 protected:
  virtual bool check(const vector<TensorPtr>& ts) {
    bool flag = true;
    for (size_t i = 0; i < ts.size(); ++i) {
      flag &= (ts[i]->type() == "CPUTensor");
    }      
    return flag; 
  }
};

}  // namespace easydl

#endif  // OPERATOR_INCLUDE_OPERATOR_H_
