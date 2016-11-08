/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <memory>
#include "Python.h"
#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"

#include "tensor/include/tensor.h"
#include "operator/include/operator.h"

namespace bp = boost::python;

#define T float

namespace easydl {
using std::shared_ptr;

shared_ptr<Tensor<T>> TensorInit(bool gpu) {
  if (gpu) {
    return shared_ptr<Tensor<T>>(new GPUTensor<T>());
  } else {
    return shared_ptr<Tensor<T>>(new CPUTensor<T>());
  }
}

shared_ptr<Tensor<T>> TensorInitShape(bool gpu, bp::list s) {
  std::vector<int> shape((bp::stl_input_iterator<int>(s)),
                            bp::stl_input_iterator<int>());
  if (gpu) {
    return shared_ptr<Tensor<T>>(new GPUTensor<T>(shape));
  } else {
    return shared_ptr<Tensor<T>>(new CPUTensor<T>(shape));
  }
}

template<typename containedType>
struct custom_vector_to_list{
  static PyObject* convert(const std::vector<containedType>& v){
    bp::list ret;
    BOOST_FOREACH(const containedType& e, v) ret.append(e);
    return bp::incref(ret.ptr());
  }
};

template <typename containedType>
struct custom_vector_from_seq{
  custom_vector_from_seq(){
    bp::converter::registry::push_back(&convertible,&construct,
                    bp::type_id<std::vector<containedType> >());
  }
  static void* convertible(PyObject* obj_ptr){
    // the second condition is important, for some reason otherwise there
    // were attempted conversions of Body to list which failed afterwards.
    if(!PySequence_Check(obj_ptr) ||
       !PyObject_HasAttrString(obj_ptr,"__len__")) return 0;
    return obj_ptr;
  }
  static void construct(PyObject* obj_ptr,
        bp::converter::rvalue_from_python_stage1_data* data){
    void* storage=((bp::converter::rvalue_from_python_storage<
                    std::vector<containedType> >*)(data))->storage.bytes;
    new (storage) std::vector<containedType>();
    std::vector<containedType>* v=(std::vector<containedType>*)(storage);
    int l=PySequence_Size(obj_ptr); if(l<0) abort();
    /*std::cerr<<"l="<<l<<"; "<<typeid(containedType).name()<<std::endl;*/
    v->reserve(l);
    for(int i=0; i<l; i++) {
      v->push_back(bp::extract<containedType>(PySequence_GetItem(obj_ptr,i)));
    }
    data->convertible=storage;
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Tensor_reshape_overloads,
                                       reshape, 1, 2);

BOOST_PYTHON_MODULE(easydl) {
  custom_vector_from_seq<int>();
  bp::to_python_converter<std::vector<int>, custom_vector_to_list<int>>();

  bp::class_<Tensor<T>, shared_ptr<Tensor<T>>>("Tensor", bp::no_init)
      .def("__init__", bp::make_constructor(TensorInit))
      .def("__init__", bp::make_constructor(TensorInitShape))
      .def("reshape", &Tensor<T>::reshape, Tensor_reshape_overloads())
      .def("reserve", &Tensor<T>::reserve)
      .def("shape", &Tensor<T>::shape, bp::return_value_policy<bp::copy_const_reference>())
      .add_property("size", &Tensor<T>::size)
      .def("type", &Tensor<T>::type);
}

}  // namespace xdl
