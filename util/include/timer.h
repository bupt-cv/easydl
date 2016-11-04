/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#ifndef UTIL_INCLUDE_TIMER_H_
#define UTIL_INCLUDE_TIMER_H_

#include <chrono>  // NOLINT(build/c++11)

namespace easydl {

class Timer {
 public:
  Timer() { Start(); }

  void Start() {
    time_ = std::chrono::system_clock::now();
  }

  double CountMs() {
    end_ = std::chrono::system_clock::now();
    auto duration = end_ - time_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        duration).count();
  }

 private:
  std::chrono::system_clock::time_point time_, end_;
};

}  // namespace easydl

#endif  // UTIL_INCLUDE_TIMER_H_
