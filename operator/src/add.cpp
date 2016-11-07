/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <iostream>
#include "operator/include/add.h"
#include "util/include/math_functions.h"
#include "util/include/common.h"
#include "tensor/include/tensor.h"

namespace easydl {

template <typename T>
bool CPUAddOp<T>::check(const vector<TensorPtr>& ts) {
  bool flag = CPUOperator<T>::check(ts);
  flag &= (ts.size() >= 3);
  return flag; 
}

template <typename T>
void CPUAddOp<T>::reshape(vector<TensorPtr>& ts) {
  for (size_t i = 1; i < ts.size() - 1; ++i) {
    CHECK_EQ(ts[i]->size(), ts[0]->size());
  }
  ts.back()->reshape(ts[0]->shape(), false);
}

template <typename T>
void CPUAddOp<T>::operator()(vector<TensorPtr>& ts) {
  T* out = ts.back()->data_mutable();
  memset(out, 0, sizeof(T) * ts.back()->size());
  for (size_t i = 0; i < ts.size() - 1; ++i) {
    const T* in = ts[i]->data();
    for (size_t j = 0; j < ts[0]->size(); ++j) {
      out[j] += in[j];
    }
  }
}

template <typename T>
bool GPUAddOp<T>::check(const vector<TensorPtr>& ts) {
  bool flag = GPUOperator<T>::check(ts);
  flag &= (ts.size() >= 3);
  return flag; 
}

template <typename T>
void GPUAddOp<T>::reshape(vector<TensorPtr>& ts) {
  for (size_t i = 1; i < ts.size() - 1; ++i) {
    CHECK_EQ(ts[i]->size(), ts[0]->size());
  }
  ts.back()->reshape(ts[0]->shape(), false);
}

template <typename T>
void GPUAddOp<T>::operator()(vector<TensorPtr>& ts) {
  T* out = ts.back()->data_mutable();
  gpuset(out, 0, ts.back()->size());
  for (size_t i = 0; i < ts.size() - 1; ++i) {
    gpuadd(ts[i]->size(), ts[i]->data(), out, out);
  }
}

INSTANTIATE_CLASS(GPUAddOp);
INSTANTIATE_CLASS(CPUAddOp);

}  // namespace easydl
