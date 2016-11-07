/* Copyright (c) 2015, Alibaba Inc. All Rights Reserved
 *
 *  Author: zhenqi.xzq
 */

#ifndef UTIL_INCLUDE_UTIL_H_
#define UTIL_INCLUDE_UTIL_H_

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include "boost/python.hpp"
// #include "numpy/arrayobject.h"

#include "tensor/include/tensor.h"

// namespace bp = boost::python;

namespace easydl {

using std::vector;
// return numpy shape as vector
// inline std::vector<size_t> get_array_shape(PyArrayObject* data) {
//   return std::vector<size_t>(PyArray_DIMS(data),
//       PyArray_DIMS(data) + PyArray_NDIM(data));
// }
// 
// // get data_type of numpy object
// inline std::string get_type(bp::object bparr) {
//   PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(bparr.ptr());
//   PyArray_Descr* desc = PyArray_DESCR(arr);
//   return std::string(1, desc->type);
// }

// format shape string
inline std::string shape_string(const std::vector<size_t>& shape) {
  std::ostringstream stream;
  for (size_t i = 0; i < shape.size(); ++i) {
    stream << shape[i] << " ";
  }
  return stream.str();
}

// stat file in bytes
inline size_t get_file_size(const std::string& path) {
  std::ifstream in(path.c_str(), std::ios::binary);
  in.seekg(0, in.end);
  size_t out = in.tellg();
  in.close();
  return out;
}

// save gpu/cpu matrix to file
template <typename T>
void dump_matrix(std::string filename, const Tensor<T>* m) {
  std::ofstream of(filename);
  std::vector<size_t> shape = m->shape();
  int row = shape[0];
  int col = m->size() / row;
  std::vector<T> temp(m->size());
  m->fill_to(temp.data());

  for (size_t i = 0; i < shape.size(); ++i) {
    of << shape[i] << " ";
  }
  of << std::endl << row << " " << col << std::endl;

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      of << temp[i * col + j] << " ";
    }
    of << std::endl;
  }
}

template void dump_matrix<int>(std::string, const Tensor<int>*);
template void dump_matrix<float>(std::string, const Tensor<float>*);

// fill numpy array into matrix
// template<typename T>
// void nparray_to_matrix(bp::object arr, Tensor<T>* matrix) {
//   CHECK_EQ(get_type(arr), typeid(T).name());
//   PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(arr.ptr());
//   std::vector<size_t> shape = get_array_shape(data_arr);
//   matrix->reshape(shape, false);
//   matrix->fill(reinterpret_cast<T*>(PyArray_DATA(data_arr)));
//   return;
// }
// 
// template void nparray_to_matrix<int>(bp::object arr, Tensor<int>* matrix);
// template void nparray_to_matrix<float>(bp::object arr, Tensor<float>* matrix);

class Index {
 public:
  Index(const vector<size_t>& stride): stride_(stride) {
    stride_.resize(4, 1);
  }
  // TODO: rewrite to get a more general index
  size_t operator()(int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0) {
    return ((i0 * stride_[1] + i1) * stride_[2] + i2) * stride_[3] + i3;
  }
 private:
  vector<size_t> stride_;
};

}  // namespace easydl

#endif  // UTIL_INCLUDE_UTIL_H_
