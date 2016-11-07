/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include "util/include/util.h"
#include "tensor/include/tensor.h"
#include "util/include/common.h"

namespace easydl {

// Implementation of Tensor
// when check is true, simply do elem count check
// when check is false, call reserve()
template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& shape, bool check) {
  size_t shape_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_size *= shape[i];
  }
  if (check) {
    if (shape_size != this->size()) {
      LOG(FATAL) << "shape size mismatch. current shape: "
          << shape_string(shape_) << "input shape: " << shape_string(shape);
    }
  } else {
    reserve(shape_size);
  }
  shape_ = shape;  // deep copy
}

template <typename T>
size_t Tensor<T>::size() const {
  if (shape_.empty()) return 0;
  size_t shape_size = 1;
  for (size_t i = 0; i < shape_.size(); ++i) {
    shape_size *= shape_[i];
  }
  return shape_size;
}

// Implementation of CPUTensor
template <typename T>
CPUTensor<T>::CPUTensor(const std::vector<size_t>& shape) {
  this->shape_ = shape;  // deep copy
  this->capacity_ = 0;
  this->data_ = NULL;
  size_t s = this->size() * this->inflate_ratio_;
  if (s > 0) {
    this->data_ = reinterpret_cast<T*>(std::malloc(s * sizeof(T)));
    CHECK(this->data_) << "Allocation failed, size: " << s;
    this->capacity_ = s;
  }
}

template <typename T>
CPUTensor<T>::~CPUTensor() {
  if (this->data_ != NULL) {
    std::free(this->data_);
    this->data_ = NULL;
  }
}

template <typename T>
void CPUTensor<T>::reserve(const size_t s) {
  if (s > this->capacity_) {
    std::free(this->data_);
    size_t new_s = s * this->inflate_ratio_;
    this->data_ = reinterpret_cast<T*>(std::malloc(new_s * sizeof(T)));
    this->capacity_ = new_s;
  }
}

template <typename T>
void CPUTensor<T>::fill(const T* src) {
  memcpy(this->data_, src, this->size() * sizeof(T));
}

template <typename T>
void CPUTensor<T>::fill_to(T* dst) const {
  memcpy(dst, this->data_, this->size() * sizeof(T));
}

// Implementation of GPUTensor
template <typename T>
GPUTensor<T>::GPUTensor(const std::vector<size_t>& shape) {
  this->shape_ = shape;
  this->capacity_ = 0;
  this->data_ = NULL;
  size_t s = this->size() * this->inflate_ratio_;
  if (s > 0) {
    CUDA_CHECK(cudaMalloc(&this->data_, s * sizeof(T)));
    this->capacity_ = s;
  }
}

template <typename T>
GPUTensor<T>::~GPUTensor() {
  if (this->data_ != NULL) {
    CUDA_CHECK(cudaFree(this->data_));
  }
}

template <typename T>
void GPUTensor<T>::reserve(size_t s) {
  if (s > this->capacity_) {
    CUDA_CHECK(cudaFree(this->data_));
    size_t new_s = s * this->inflate_ratio_;
    CUDA_CHECK(cudaMalloc(&this->data_, new_s * sizeof(T)));
    this->capacity_ = new_s;
  }
}
template <typename T>
void GPUTensor<T>::fill(const T* src) {
  CUDA_CHECK(cudaMemcpy(this->data_, src, this->size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

template <typename T>
void GPUTensor<T>::fill_to(T* dst) const {
  CUDA_CHECK(cudaMemcpy(dst, this->data_, this->size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

INSTANTIATE_CLASS(Tensor);
INSTANTIATE_CLASS(GPUTensor);
INSTANTIATE_CLASS(CPUTensor);

}  // namespace easydl
