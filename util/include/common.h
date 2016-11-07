/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef UTIL_INCLUDE_COMMON_H_
#define UTIL_INCLUDE_COMMON_H_

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#endif  // UTIL_INCLUDE_COMMON_H_
