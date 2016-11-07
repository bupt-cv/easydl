/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include "util/include/math_functions.h"

#include <random>

namespace easydl {

template <typename T>
void gaussian(const int n, T mean, T std, T* out) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> d(mean, std);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

template void gaussian<float>(const int, float, float, float*);
template void gaussian<double>(const int, double, double, double*);

}  // namespace easydl
