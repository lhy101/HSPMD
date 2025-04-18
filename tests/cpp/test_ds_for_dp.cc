#include "hspmd/core/ndarray.h"
#include "hspmd/execution/dar_executor.h"
#include "hspmd/execution/dbr_executor.h"
#include "hspmd/impl/communication/comm_group.h"
#include "hspmd/autograd/optim/Optimizer.h"
#include "hspmd/autograd/ops/Variable.h"
#include "hspmd/autograd/ops/MatMul.h"
#include "hspmd/autograd/ops/BinaryCrossEntropy.h"
#include "hspmd/autograd/ops/Sigmoid.h"
#include "hspmd/autograd/ops/Relu.h"
#include "hspmd/autograd/ops/Sum.h"
#include "hspmd/autograd/ops/Communicate.h"
#include "hspmd/autograd/distributed_states.h"
#include <cmath>

using namespace hspmd;
using namespace hspmd::autograd;
using namespace hspmd::execution;
using namespace hspmd::impl;
using namespace hspmd::impl::comm;

// distributed tensor版本的数据并行
void TestDARDistributedTensor(DataType dtype = kFloat32) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  HT_ASSERT(all_devices.num_devices() >= 4) << "device num must >= 4 !";
  DeviceGroup all_device_group({all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)});
  
  int n = 2;
  int dim = 4; 

  DistributedStates ds0(4, {{-1, 4}}, {-1});
  DistributedStates ds1(4, {{0, 4}}, {0});

  auto tensor1 = PlaceholderOp(dtype, {n, dim}, OpMeta().set_device_group(all_device_group).set_name("tensor1"))->output(0);
  tensor1->set_distributed_states(ds1); // tensor1: ds1: [(2,4), (2,4), (2,4), (2,4)]
  // const StreamIndex NDArray::DEFAULT_STREAM = kComputingStream;
  // auto w = VariableOp({dim*n, dim}, HeUniformInitializer(), dtype, true)->output(0);
  // auto w_data = (NDArray::rand({dim, dim}, Device(kCPU), kFloat32, 0.0, 1.0, 2023) - 0.5) * (2.0 / std::sqrt(dim));
  // auto w_data = (NDArray::rand({dim, dim}, Device(kCPU), kFloat32, 0.0, 1.0, 2023) - 0.5);  
  auto w_data = NDArray::rand({dim, dim}, Device(kCPU), kFloat32, 0.0, 1.0, 2023, kBlockingStream);

  auto w = VariableOp(w_data, true, OpMeta().set_device_group(all_device_group).set_name("w"))->output(0);
  w->set_distributed_states(ds0); // duplicate: w=(4,4)


  auto result = MatMulOp(tensor1, w)->output(0);

  auto y = PlaceholderOp(dtype, {n, dim}, OpMeta().set_device_group(all_device_group).set_name("y"))->output(0);  
  y->set_distributed_states(ds1);
  auto loss = BinaryCrossEntropyOp(result, y, "mean")->output(0);

  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  DARExecutor exec(local_device, all_devices, {result, train_op});

  NDArray data = NDArray::randn({n, dim}, local_device, dtype, 0.0, 1.0, (666 + all_device_group.get_index(local_device)));
  NDArray labels = NDArray::zeros({n, dim}, local_device, dtype);

  SynchronizeAllStreams();
  // HT_LOG_INFO << local_device << ": w_data = " << w_data;
  FeedDict feed_dict = {{tensor1->id(), data}, {y->id(), labels}};

  TensorList weights;
  OpList weights_op;
  auto topo = TopoSort(train_op);
  for (auto op : topo) {
    if (is_variable_op(op)) {
      weights.push_back(op->output(0));
      weights_op.push_back(op);
    }
  }
  weights.push_back(train_op);
  for (size_t i = 0; i < weights.size()-1; i++) {
    HT_LOG_INFO << local_device << ": init weight: " << weights[i] << ", value: " << reinterpret_cast<VariableOp&>(weights_op[i])->data();
  }

  auto results = exec.Run(weights, feed_dict);
  for (size_t i = 0; i < weights.size()-1; i++) {
    if (results[i] != Tensor())
      HT_LOG_INFO << local_device << ": updated weight: " << weights[i] << ", value: " << results[i];
  }    
}

int main(int argc, char** argv) {
  SetUpDeviceMappingAndAssignLocalDeviceOnce();
  TestDARDistributedTensor();
  return 0;
}

