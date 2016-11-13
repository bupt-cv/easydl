/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "util/include/math_functions.h"
#include "operator/include/operator.h"
#include "operator/include/add.h"

namespace easydl {

template <typename T>
class TestAddOp : public ::testing::Test {
  typedef std::shared_ptr<Tensor<T>> TensorPtr;

 protected:
  TestAddOp(): cop_(OperatorParameter()), gop_(OperatorParameter()) {}
  virtual void SetUp() {
    vector<int> shape = {2, 3, 4};
    cin0_.reset(new CPUTensor<T>(shape));
    cin1_.reset(new CPUTensor<T>(shape));
    cin2_.reset(new CPUTensor<T>(shape));
    cout_.reset(new CPUTensor<T>(shape));
    gin0_.reset(new GPUTensor<T>(shape));
    gin1_.reset(new GPUTensor<T>(shape));
    gin2_.reset(new GPUTensor<T>(shape));
    gout_.reset(new GPUTensor<T>(shape));

    gaussian(static_cast<int>(cin0_->size()), static_cast<T>(2),
             static_cast<T>(3), cin0_->data_mutable());
    gaussian(static_cast<int>(cin1_->size()), static_cast<T>(2),
             static_cast<T>(3), cin1_->data_mutable());
    gaussian(static_cast<int>(cin2_->size()), static_cast<T>(2),
             static_cast<T>(3), cin2_->data_mutable());
    gin0_->fill(cin0_->data());
    gin1_->fill(cin1_->data());
    gin2_->fill(cin2_->data());

    cvtp_ = {cin0_, cin1_, cin2_, cout_};
    gvtp_ = {gin0_, gin1_, gin2_, gout_};
  }

  TensorPtr cin0_, cin1_, cin2_, cout_;
  TensorPtr gin0_, gin1_, gin2_, gout_;
  vector<TensorPtr> cvtp_, gvtp_;
  CPUAddOp<T> cop_;
  GPUAddOp<T> gop_;
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_CASE(TestAddOp, TestTypes);

TYPED_TEST(TestAddOp, check) {
  EXPECT_TRUE(this->cop_.check(this->cvtp_));
  EXPECT_TRUE(this->gop_.check(this->gvtp_));
  EXPECT_FALSE(this->cop_.check(this->gvtp_));
  EXPECT_FALSE(this->gop_.check(this->cvtp_));
  this->cvtp_.resize(2);
  EXPECT_FALSE(this->cop_.check(this->cvtp_));
  this->gvtp_.resize(1);
  EXPECT_FALSE(this->gop_.check(this->gvtp_));
}

TYPED_TEST(TestAddOp, reshape) {
  this->cop_.reshape(this->cvtp_);
  this->gop_.reshape(this->gvtp_);
  EXPECT_EQ(this->cout_->shape().size(), 3);
  EXPECT_EQ(this->cout_->shape()[0], 2);
  EXPECT_EQ(this->cout_->shape()[1], 3);
  EXPECT_EQ(this->cout_->shape()[2], 4);
  EXPECT_EQ(this->gout_->shape().size(), 3);
  EXPECT_EQ(this->gout_->shape()[0], 2);
  EXPECT_EQ(this->gout_->shape()[1], 3);
  EXPECT_EQ(this->gout_->shape()[2], 4);
}

TYPED_TEST(TestAddOp, equal) {
  this->cop_.reshape(this->cvtp_);
  this->cop_(this->cvtp_);
  this->cop_(this->cvtp_);
  this->cop_(this->cvtp_);
  this->gop_.reshape(this->gvtp_);
  this->gop_(this->gvtp_);
  CPUTensor<TypeParam> temp(this->gout_->shape());
  this->gout_->fill_to(temp.data_mutable());
  for (size_t i = 0; i < temp.size(); ++i) {
    EXPECT_EQ(this->cout_->data()[i], temp.data()[i]);
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
