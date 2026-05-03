#include "hspmd/graph/ops/Bool.h"
#include "hspmd/graph/headers.h"
#include "hspmd/graph/ops/kernel_links.h"

namespace hspmd {
namespace graph {

void BoolOpImpl::DoCompute(Operator&op,
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hspmd::impl::Bool,
                                  inputs.at(0), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList BoolOpImpl::DoGradient(Operator& op,const TensorList& grad_outputs) const {
  return {Tensor()};
}

Tensor MakeBoolOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<BoolOpImpl>(),
          {std::move(input)},
          std::move(op_meta))->output(0);  
}

} // namespace graph
} // namespace hspmd
