/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>
