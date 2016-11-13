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


/* type conversion between std::vector and list */
template<typename containedType>
struct custom_vector_to_list {
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
    v->reserve(l);
    for(int i=0; i<l; i++) {
      v->push_back(bp::extract<containedType>(PySequence_GetItem(obj_ptr,i)));
    }
    data->convertible=storage;
  }
};

/* type conversion between std::map and dict */
template <typename keyType, typename valueType>
struct custom_map_to_dict {
  static PyObject* convert(const std::map<keyType, valueType>& v) {
    bp::dict ret;     
    for (auto it = v.begin(); it != v.end(); ++it) {
      ret.setdefault(it->first, it->second);
    }
    return bp::incref(ret.ptr());
  }
};

template <typename keyType, typename valueType>
struct custom_map_from_dict {
  custom_map_from_dict() {
    bp::converter::registry::push_back(&convertible, &construct, 
                                       bp::type_id<std::map<keyType, valueType>>());
  }
  static void* convertible(PyObject* obj_ptr) {
    if (!PyDict_Check(obj_ptr)) 
      return 0;
    else
      return obj_ptr;
  }
  static void construct(PyObject* obj_ptr, 
                        bp::converter::rvalue_from_python_stage1_data* data) {
    void * storage = ((bp::converter::rvalue_from_python_storage<
                       std::map<keyType, valueType>>*)(data))->storage.bytes;
    new (storage) std::map<keyType, valueType>();
    std::map<keyType, valueType>* v = (std::map<keyType, valueType>*)(storage);
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj_ptr, &pos, &key, &value)) {
      (*v)[bp::extract<keyType>(key)] = bp::extract<valueType>(value);
    } 
    data->convertible = storage;
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Tensor_reshape_overloads,
                                       reshape, 1, 2);

BOOST_PYTHON_MODULE(easydl) {
  custom_vector_from_seq<int>();
  bp::to_python_converter<std::vector<int>, custom_vector_to_list<int>>();
  custom_vector_from_seq<std::string>();
  bp::to_python_converter<std::vector<std::string>, custom_vector_to_list<std::string>>();
  custom_map_from_dict<std::string, std::string>();
  bp::to_python_converter<std::map<std::string, std::string>,
      custom_map_to_dict<std::string, std::string>>();
  custom_vector_from_seq<Operator<T>::TensorPtr>();
  bp::to_python_converter<std::vector<Operator<T>::TensorPtr>, custom_vector_to_list<Operator<T>::TensorPtr>>();
  bp::register_ptr_to_python<std::shared_ptr<std::string>>();

  bp::class_<Tensor<T>, shared_ptr<Tensor<T>>>("Tensor", bp::no_init)
      .def("__init__", bp::make_constructor(TensorInit))
      .def("__init__", bp::make_constructor(TensorInitShape))
      .def("reshape", &Tensor<T>::reshape, Tensor_reshape_overloads())
      .def("reserve", &Tensor<T>::reserve)
      .def("shape", &Tensor<T>::shape, bp::return_value_policy<bp::copy_const_reference>())
      .add_property("size", &Tensor<T>::size)
      .def("type", &Tensor<T>::type);

  bp::class_<OperatorParameter>("OperatorParameter", bp::init<>())
      .def(bp::init<OperatorParameter::ParamType>())
      .def("get", &OperatorParameter::get)
      .def("set", &OperatorParameter::set);

  bp::class_<Operator<T>, std::shared_ptr<Operator<T>>>("Operator", bp::no_init)
      .def("__call__", &Operator<T>::operator())
      .def("check", &Operator<T>::check)
      .def("reshape", &Operator<T>::reshape);
  
  bp::class_<OperatorRegistry<T>>("OperatorRegistry", bp::no_init)
     .def("CreatorOperator", &OperatorRegistry<T>::CreateOperator)
     .staticmethod("CreatorOperator")
     .def("OperatorTypeList", &OperatorRegistry<T>::OperatorTypeList)
     .staticmethod("OperatorTypeList");
}

}  // namespace xdl
