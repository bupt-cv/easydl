/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <memory>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "util/include/common.h"
#include "operator/include/operator.h"
#include "operator/include/add.h"

namespace easydl {

TEST(TestConstruct, Construct) {
  std::vector<std::string> types = OperatorRegistry<float>::OperatorTypeList();
  LOG(INFO) << "Registered operators:";
  for (size_t i = 0; i < types.size(); ++i) {
    LOG(INFO) << types[i];
  }
}

}  // namespace easydl

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  ::google::InstallFailureSignalHandler();
  ::google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
