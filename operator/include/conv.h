/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef OPERATOR_INCLUDE_CONV_H_
#define OPERATOR_INCLUDE_CONV_H_

#include <memory>
#include <vector>
#include "operator/include/operator.h"

namespace easydl {

struct ConvOpParam {
  ConvOpParam(int output_channel,
              int kernel_size = 3,
              int stride = 1,
              int padding = 1)
      : kernel_size_(kernel_size), stride_(stride), padding_(padding),
      output_channel_(output_channel) {}
  int kernel_size_, stride_, padding_, output_channel_;
};

// TODO(xuzhenqi): remove ConvOpParameter
template <typename T>
class CPUConvOp : public CPUOperator<T> {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;
 public: 
  virtual inline std::string type() const { return "CPUConvOp"; }
  // TODO(xuzhenqi): may change to use [], take "" as defaults
  explicit CPUConvOp(const OperatorParameter& param): param_(-1) {
    std::string value;
    value = param.get("output_channel");
    CHECK(!value.empty());
    param_.output_channel_ = std::stoi(value);
    value = param.get("kernel_size");
    if (!value.empty()) {
      param_.kernel_size_ = std::stoi(value);
    }
    value = param.get("stride");
    if (!value.empty()) {
      param_.stride_ = std::stoi(value);
    }
    value = param.get("padding");
    if (!value.empty()) {
      param_.padding_ = std::stoi(value);
    }
  }
  virtual void operator()(const vector<TensorPtr>&);
  virtual bool check(const vector<TensorPtr>&);
  virtual void reshape(const vector<TensorPtr>&);
 private:
  ConvOpParam param_;
};

}  // namespace easydl

#endif  // OPERATOR_INCLUDE_CONV_H_
