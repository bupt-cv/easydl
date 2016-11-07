/* Copyright(c). All Rights Reserved
 * Author: Xu Zhenqi
 * Email: xuzhenqi1993@gmail.com
 */

#include <vector>

#include "tensor/include/tensor.h"
#include "glog/logging.h"

using std::vector;
using namespace easydl;  // NOLINT(build/namespaces)

template <typename T>
class TestTensor {
 public:
  void test_construction() {
    vector<size_t> shape;
    shape.push_back(5);
    shape.push_back(20);
    matrix_ = new T(shape);
    CHECK_EQ(matrix_->shape().size(), 2);
    CHECK_EQ(matrix_->shape()[0], 5);
    CHECK_EQ(matrix_->shape()[1], 20);
    CHECK_EQ(matrix_->size(), 100);
    CHECK_EQ(matrix_->capacity(), 200);
    delete matrix_;
    matrix_ = NULL;
  }

  void test_reshape() {
    vector<size_t> shape;
    shape.push_back(5);
    shape.push_back(20);
    matrix_ = new T(shape);
    shape[0] = 10;
    shape[1] = 10;
    matrix_->reshape(shape);
    CHECK_EQ(matrix_->shape()[0], 10);
    CHECK_EQ(matrix_->shape()[1], 10);
    CHECK_EQ(matrix_->size(), 100);
    CHECK_EQ(matrix_->capacity(), 200);
    delete matrix_;
    matrix_ = NULL;
  }

  void test_reshape_small_no_check() {
    vector<size_t> shape;
    shape.push_back(5);
    shape.push_back(20);
    matrix_ = new T(shape);
    shape[0] = 2;
    shape[1] = 6;
    matrix_->reshape(shape, false);
    CHECK_EQ(matrix_->shape()[0], 2);
    CHECK_EQ(matrix_->shape()[1], 6);
    CHECK_EQ(matrix_->size(), 12);
    CHECK_EQ(matrix_->capacity(), 200);
    delete matrix_;
    matrix_ = NULL;
  }

  void test_reshape_big_no_check() {
    vector<size_t> shape = {10, 20};
    matrix_ = new T(shape);
    shape[0] = 10;
    shape[1] = 200;
    matrix_->reshape(shape, false);
    CHECK_EQ(matrix_->shape()[0], 10);
    CHECK_EQ(matrix_->shape()[1], 200);
    CHECK_EQ(matrix_->size(), 2000);
    CHECK_EQ(matrix_->capacity(), 4000);
    delete matrix_;
    matrix_ = NULL;
  }

  void test_data_fill() {
    typedef typename T::Datatype Datatype;
    matrix_ = new T({10, 10});
    vector<Datatype> data(matrix_->size()), data2(matrix_->size());
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i;
    }
    matrix_->fill(&(data[0]));
    matrix_->fill_to(&(data2[0]));
    for (size_t i = 0; i < data.size(); ++i) {
      CHECK_EQ(data[i], data2[i]);
    }
    delete matrix_;
    matrix_ = NULL;
  }

  void run() {
    LOG(INFO) << "Testing construction";
    test_construction();
    LOG(INFO) << "Testing reshape";
    test_reshape();
    LOG(INFO) << "Testing reshape small no check";
    test_reshape_small_no_check();
    LOG(INFO) << "Testing reshape big no check";
    test_reshape_big_no_check();
    LOG(INFO) << "Testing data fill";
    test_data_fill();
  }

 private:
  T* matrix_;
};

template class TestTensor<CPUTensor<float> >;
template class TestTensor<GPUTensor<float> >;

int main() {
  TestTensor<CPUTensor<float> > tmc;
  LOG(INFO) << "Testing TestTensor<CPUTensor>";
  tmc.run();
  TestTensor<GPUTensor<float> > tmg;
  LOG(INFO) << "Testing TestTensor<GPUTensor>";
  tmg.run();
  LOG(INFO) << "Testing finished";
  return 0;
}
