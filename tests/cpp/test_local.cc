#include "hspmd/core/ndarray.h"
#include "hspmd/execution/dar_executor.h"
#include "hspmd/execution/dbr_executor.h"
#include "hspmd/autograd/optim/Optimizer.h"
#include "hspmd/autograd/ops/Variable.h"
#include "hspmd/autograd/ops/MatMul.h"
#include "hspmd/autograd/ops/BinaryCrossEntropy.h"
#include "hspmd/autograd/ops/Sigmoid.h"
#include "hspmd/autograd/ops/Relu.h"
#include "hspmd/autograd/ops/Dropout.h"
#include <cmath>

using namespace hspmd;
using namespace hspmd::autograd;
using namespace hspmd::execution;

void TestDARLocalMLP(const Device& device, DataType dtype = kFloat32,
                     const HTShape& dims = {256, 64, 16, 1}) {
  HT_LOG_INFO << "Testing local MLP in define-and-run mode";
  HT_ASSERT(dims.back() == 1) << "Label size should be 1";
  auto x = PlaceholderOp(dtype, {-1, dims[0]})->output(0);
  auto y = PlaceholderOp(dtype, {-1, 1})->output(0);
  auto act = x;
  for (size_t i = 1; i < dims.size(); i++) {
    auto var =
      VariableOp({dims[i - 1], dims[i]}, HeUniformInitializer(), dtype, true)
        ->output(0);
    act = MatMulOp(act, var)->output(0);
    if (i + 1 < dims.size())
      act = ReluOp(act)->output(0);
  }
  auto prob = SigmoidOp(act)->output(0);
  auto loss = BinaryCrossEntropyOp(prob, y, "mean")->output(0);
  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  DARExecutor exec(device);

  NDArray features = NDArray::randn({1024, dims[0]}, device, dtype);
  NDArray labels = NDArray::zeros({1024, 1}, device, dtype);
  SynchronizeAllStreams();
  FeedDict feed_dict = {{x->id(), features}, {y->id(), labels}};

  // warmup
  for (int i = 0; i < 10; i++)
    exec.Run({train_op}, feed_dict);

  TIK(train);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob, loss, train_op}, feed_dict);
  TOK(train);
  HT_LOG_INFO << "Train 1000 iter cost " << COST_MSEC(train) << " ms";

  TIK(eval);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob}, feed_dict);
  TOK(eval);
  HT_LOG_INFO << "Infer 1000 iter cost " << COST_MSEC(eval) << " ms";
}

void TestDABLocalMLP(const Device& device, DataType dtype = kFloat32,
                     const HTShape& dims = {256, 64, 16, 1}) {
  HT_LOG_INFO << "Testing local MLP in define-by-run mode";
  HT_ASSERT(dims.back() == 1) << "Label size should be 1";
  TensorList vars;
  for (size_t i = 1; i < dims.size(); i++) {
    auto var =
      VariableOp({dims[i - 1], dims[i]}, HeUniformInitializer(), dtype, true)
        ->output(0);
    vars.push_back(var);
  }
  SGDOptimizer optimizer(vars, 0.1);

  NDArray features = NDArray::randn({1024, dims[0]}, device, dtype);
  NDArray labels = NDArray::zeros({1024, 1}, device, dtype);
  SynchronizeAllStreams();

  auto fn = [&](bool training) {
    auto x = VariableOp(features, false)->output(0);
    auto y = VariableOp(labels, false)->output(0);
    auto act = x;
    for (size_t i = 1; i < dims.size(); i++) {
      ;
      act = MatMulOp(act, vars[i - 1])->output(0);
      if (i + 1 < dims.size())
        act = ReluOp(act)->output(0);
    }
    auto prob = SigmoidOp(act)->output(0);
    if (training) {
      auto loss = BinaryCrossEntropyOp(prob, y, "mean")->output(0);
      optimizer.ZeroGrad();
      loss->Backward();
      optimizer.Step();
    } else {
      prob->GetOrCompute();
    }
  };

  // warmup
  for (int i = 0; i < 10; i++)
    fn(true);

  TIK(train);
  for (int i = 0; i < 1000; i++)
    fn(true);
  TOK(train);
  HT_LOG_INFO << "Train 1000 iter cost " << COST_MSEC(train) << " ms";

  // warmup
  for (int i = 0; i < 10; i++)
    fn(false);

  TIK(eval);
  for (int i = 0; i < 1000; i++)
    fn(false);
  TOK(eval);
  HT_LOG_INFO << "Infer 1000 iter cost " << COST_MSEC(eval) << " ms";
}

int main(int argc, char** argv) {
  TestDARLocalMLP(Device(kCUDA, 0));
  TestDABLocalMLP(Device(kCUDA, 0));
  return 0;
}