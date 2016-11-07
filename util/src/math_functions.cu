/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include "util/include/math_functions.h"
#include "util/include/cuda_common.h"

namespace easydl {

template <typename T>
void gpuset(T* dst, char ch, size_t size) {
  CUDA_CHECK(cudaMemset(dst, ch, size * sizeof(T))); 
}

template void gpuset<float>(float*, char, size_t);
template void gpuset<double>(double*, char, size_t);

template <typename T>
__global__ void add_kernel(const int n, const T* d0, const T* d1, T* out){
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = d0[i] + d1[i];
  }
}

template <typename T>
void gpuadd(const int n, const T* d0, const T* d1, T* out) {
  // NOLINE_NEXT_LINE(whitespace/operators)
  add_kernel<T><<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, d0, d1, out);
}

template void gpuadd<float>(const int, const float*, const float*, float*);
template void gpuadd<double>(const int, const double*, const double*, double*);

}
