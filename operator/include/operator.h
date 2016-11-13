/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef OPERATOR_INCLUDE_OPERATOR_H_
#define OPERATOR_INCLUDE_OPERATOR_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "tensor/include/tensor.h"

namespace easydl {

using std::vector;

template <typename T>
class Operator {
 public:
  virtual inline std::string type() const { return "Operator"; }
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
  virtual void operator()(const vector<TensorPtr>&) {}
  virtual bool check(const vector<TensorPtr>&) {}
  virtual void reshape(const vector<TensorPtr>&) {}
};

template <typename T>
class GPUOperator : public Operator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 public:
  virtual inline std::string type() const { return "GPUOperator"; }
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
  virtual inline std::string type() const { return "CPUOperator"; }
 public:
  virtual bool check(const vector<TensorPtr>& ts) {
    bool flag = true;
    for (size_t i = 0; i < ts.size(); ++i) {
      flag &= (ts[i]->type() == "CPUTensor");
    }
    return flag;
  }
};

class OperatorParameter {
 public:
  typedef std::map<std::string, std::string> ParamType;
  OperatorParameter() {}
  OperatorParameter(const ParamType& p): params_(p) {}
  std::string get(const std::string& item) const {
    std::string ret;
    if (params_.find(item) != params_.end()) {
      return std::string(params_.at(item));
    } else
      return std::string();
  }
  void set(const std::string& item, const std::string& value) {
    params_[item] = value;
  }
 private:
  ParamType params_;
};

template <typename T>
class OperatorRegistry {
 public:
  typedef std::shared_ptr<Operator<T> > (*Creator)(const OperatorParameter&);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Operator type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a OperatorParameter.
  static std::shared_ptr<Operator<T> > CreateOperator(const std::string& type,
      const OperatorParameter& param) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << OperatorTypeListString() << ")";
    return registry[type](param);
  }

  static vector<std::string> OperatorTypeList() {
    CreatorRegistry& registry = Registry();
    vector<std::string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Operator registry should never be instantiated -
  // everything is done with its
  // static variables.
  OperatorRegistry() {}

  static std::string OperatorTypeListString() {
    vector<std::string> layer_types = OperatorTypeList();
    std::string layer_types_str;
    for (vector<std::string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};

template <typename T>
class OperatorRegisterer {
 public:
  OperatorRegisterer(const std::string& type,
                  std::shared_ptr<Operator<T> > (*creator)(
                      const OperatorParameter&)) {
    OperatorRegistry<T>::AddCreator(type, creator);
  }
};

#define REGISTER_OPERATOR_CREATOR(type, creator)                                  \
  static OperatorRegisterer<float> g_creator_f_##type(#type, creator<float>);  \
  static OperatorRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_OPERATOR_CLASS(type)                          \
  template <typename T>                                        \
  std::shared_ptr<Operator<T> > Creator_##type(                \
      const OperatorParameter& param) {                        \
    return std::shared_ptr<Operator<T> >(new type<T>(param));  \
  }                                                            \
  REGISTER_OPERATOR_CREATOR(type, Creator_##type)

}  // namespace easydl

#endif  // OPERATOR_INCLUDE_OPERATOR_H_
