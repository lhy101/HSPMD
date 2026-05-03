#include "hspmd/_binding/module.h"
#include "hspmd/_binding/constants.h"
#include "hspmd/_binding/utils/pybind_common.h"
#include "hspmd/_binding/core/device.h"
#include "hspmd/_binding/core/dtype.h"
#include "hspmd/_binding/core/stream.h"
#include "hspmd/_binding/core/ndarray.h"
#include "hspmd/_binding/core/symbol.h"
#include "hspmd/_binding/graph/operator.h"
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/_binding/graph/distributed_states.h"
#include "hspmd/_binding/graph/graph.h"
#include "hspmd/_binding/graph/autocast.h"
#include "hspmd/_binding/graph/recompute.h"
#include "hspmd/_binding/graph/cpu_offload.h"
#include "hspmd/_binding/graph/gradscaler.h"
#include "hspmd/_binding/graph/sgdoptimizer.h"
#include "hspmd/_binding/graph/subgraph.h"
#include "hspmd/_binding/graph/adamoptimizer.h"
#include "hspmd/_binding/graph/dataloader.h"
#include "hspmd/_binding/graph/init/initializer.h"
#include "hspmd/_binding/distributed/comm_group.h"
#include "hspmd/_binding/graph/profiler.h"

PYBIND11_MODULE(HT_CORE_PY_MODULE, m) {
  hspmd::AddPyDeviceTypeToModule(m);
  hspmd::AddPyDeviceGroupTypeToModule(m);
  hspmd::AddPyDataTypeTypeToModule(m);
  hspmd::AddPyStreamTypeToModule(m);
  hspmd::AddPyNDArrayTypeToModule(m);
  hspmd::AddPyIntSymbolTypeToModule(m);
  hspmd::AddPyCommGroupTypeToModule(m);
  hspmd::graph::AddPyOperatorTypeToModule(m);
  hspmd::graph::AddPyTensorTypeToModule(m);
  hspmd::graph::AddPyDistributedStatesTypeToModule(m);
  hspmd::graph::AddPyDistributedStatesUnionTypeToModule(m);
  hspmd::graph::AddPyGraphTypeToModule(m);
  hspmd::graph::AddPyAutoCastTypeToModule(m);
  hspmd::graph::AddPyGradScalerTypeToModule(m);
  hspmd::graph::AddPySGDOptimizerTypeToModule(m);
  hspmd::graph::AddPySubGraphTypeToModule(m);
  hspmd::graph::AddPyAdamOptimizerTypeToModule(m);
  hspmd::graph::AddPyDataloaderTypeToModule(m);
  hspmd::graph::AddPyInitializerTypeToModule(m);
  auto internal_sub_module = m.def_submodule("_internal_context");
  hspmd::graph::AddOpContextManagingFunctionsToModule(internal_sub_module);
  hspmd::graph::AddGraphContextManagingFunctionsToModule(internal_sub_module);
  hspmd::graph::AddAutoCastContextManagingFunctionsToModule(internal_sub_module);
  hspmd::graph::AddSubGraphContextManagingFunctionsToModule(internal_sub_module);
  hspmd::graph::AddRecomputeContextManagingFunctionsToModule(internal_sub_module);
  hspmd::graph::AddCPUOffloadContextManagingFunctionsToModule(internal_sub_module);
  hspmd::impl::AddPyProfileTypeToModule(m);
  hspmd::impl::AddProfileContextManagingFunctionsToModule(internal_sub_module);
}
