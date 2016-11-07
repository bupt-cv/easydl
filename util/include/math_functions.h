/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef UTIL_INCLUDE_MATH_FUNCTIONS_H_
#define UTIL_INCLUDE_MATH_FUNCTIONS_H_

#include <cstddef>

namespace easydl {

template <typename T>
void gpuset(T* dst, char ch, size_t size);

template <typename T>
void gpuadd(const int n, const T* d0, const T* d1, T* out);

template <typename T>
void gaussian(const int n, T mean, T std, T* out);

}  // namespace easydl

#endif  // UTIL_INCLUDE_MATH_FUNCTIONS_H_
