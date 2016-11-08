/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <vector>
#include "operator/include/conv.h"
#include "util/include/util.h"
#include "util/include/common.h"

namespace easydl {

template <typename T>
bool CPUConvOp<T>::check(const vector<TensorPtr>& ts) {
  bool flag = CPUOperator<T>::check(ts);
  flag &= (ts.size() == 3);
  return flag;
}

template <typename T>
void CPUConvOp<T>::reshape(const vector<TensorPtr>& ts) {
  vector<int> in_shape = ts[0]->shape();
  CHECK_EQ(in_shape.size(), 4);
  vector<int> weight_shape = {in_shape[1], int(param_.output_channel_),
    int(param_.kernel_size_), int(param_.kernel_size_)};
  ts[1]->reshape(weight_shape);
  int out_height = (in_shape[2] + 2 * param_.padding_ + 1 -
                   param_.kernel_size_) / param_.stride_;
  int out_width = (in_shape[3] + 2 * param_.padding_ + 1 -
                   param_.kernel_size_) / param_.stride_;
  vector<int> out_shape = {in_shape[0], int(param_.output_channel_),
    int(out_height), int(out_width)};
  ts[2]->reshape(out_shape);
}


// TODO(xuzhenqi): This implementation is totally inefficient, and can be used
// as test utility later.
template <typename T>
void CPUConvOp<T>::operator()(const vector<TensorPtr>& ts) {
  const T* in = ts[0]->data();
  const T* weight = ts[1]->data();
  T* out = ts[2]->data_mutable();
  memset(out, 0, sizeof(T) * ts[2]->size());
  Index in_index(ts[0]->shape());
  Index weight_index(ts[1]->shape());
  Index out_index(ts[2]->shape());
  const vector<int>& in_shape = ts[0]->shape();
  const vector<int>& out_shape = ts[2]->shape();
  for (int i = 0; i < in_shape[0]; ++i) {
    for (int j = 0; j < in_shape[1]; ++j) {
      for (int k = 0; k < int(param_.output_channel_); ++k) {
        for (int l = 0; l < out_shape[2]; ++l) {
          for (int o = 0; o < out_shape[3]; ++o) {
            for (int m = 0; m < int(param_.kernel_size_); ++m) {
              for (int n = 0; n < int(param_.kernel_size_); ++n) {
                int in_height = l + m - param_.padding_;
                int in_width = o + n - param_.padding_;
                if (in_height >= 0 && in_height < static_cast<int>(in_shape[2])
                    && in_width >= 0 && in_width
                    < static_cast<int>(in_shape[3])) {
                  out[out_index(i, k, l, o)] =
                      in[in_index(i, j, l + m, o + n)] *
                      weight[weight_index(j, k, m, n)];
                }
              }
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(CPUConvOp);

}  // namespace easydl
