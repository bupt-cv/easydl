/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifdef PYWRAPPER_INCLUDE__EASYDL_H_
#define PYWRAPPER_INCLUDE__EASYDL_H_

#include "boost/python.hpp"

#include "tensor/include/tensor.h"
#include "operator/include/operator.h"

namespace bp = boost::python;

typedef T float;

namespace xdl {

BOOST_PYTHON_MODULE(easydl) {
  class_<CPUTensor<T>>("CPUTensor", init<std::vector<size_t>())
      .def("type", &CPUTensor<T>::type)
      .def("reserve", &CPUTensor<T>::reserve);
}

}  // namespace xdl

#endif  // PYWRAPPER_INCLUDE__EASYDL_H_
