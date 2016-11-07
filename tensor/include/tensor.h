/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef TENSOR_INCLUDE_TENSOR_H_
#define TENSOR_INCLUDE_TENSOR_H_

#include <vector>
#include <string>
#include <cstring>

#include "util/include/cuda_common.h"

namespace easydl {

template <typename T>
class Tensor {
 public:
  typedef T Datatype;
  Tensor(): inflate_ratio_(2), capacity_(0), data_(NULL) {}
  virtual ~Tensor() {}

  virtual inline std::string type() const { return "Tensor"; }
  // reshape will call different version of reserve
  void reshape(const std::vector<size_t>& shape, bool check = true);
  // reserve guarantees capacity >= shape,
  // while doesn't keep the data
  virtual void reserve(const size_t s) = 0;

  void set_inflate_ratio(size_t ratio) { inflate_ratio_ = ratio; }
  // tuple for shape
  const std::vector<size_t>& shape() const { return shape_; }
  // available array range
  size_t capacity() const { return capacity_; }
  // meaningful array range, equals to cummul of shape
  size_t size() const;

  // read only data
  const T* data() const { return data_; }
  // writable data
  T* data_mutable() { return data_; }

  // copy from src to data
  virtual void fill(const T* src) = 0;
  // copy from data to dst
  virtual void fill_to(T* dst) const = 0;

 protected:
  size_t inflate_ratio_;
  size_t capacity_;
  std::vector<size_t> shape_;

  T* data_;
};

template <typename T>
class CPUTensor : public Tensor<T> {
 public:
  CPUTensor() {}
  explicit CPUTensor(const std::vector<size_t>& shape);
  virtual ~CPUTensor();

  virtual inline std::string type() const { return "CPUTensor"; }
  virtual void reserve(const size_t s);
  virtual void fill(const T* src);
  virtual void fill_to(T* dst) const;
};

template <typename T>
class GPUTensor : public Tensor<T> {
 public:
  GPUTensor() {}
  explicit GPUTensor(const std::vector<size_t>& shape);
  virtual ~GPUTensor();

  virtual inline std::string type() const { return "GPUTensor"; }
  virtual void reserve(const size_t s);
  virtual void fill(const T* src);
  virtual void fill_to(T* dst) const;
};

}  // namespace easydl

#endif  // TENSOR_INCLUDE_TENSOR_H_
