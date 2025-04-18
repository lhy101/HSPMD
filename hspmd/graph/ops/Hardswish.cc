#include "hspmd/graph/ops/Hardswish.h"
#include "hspmd/graph/headers.h"
#include "hspmd/graph/ops/kernel_links.h"

namespace hspmd {
namespace graph {

void HardswishOpImpl::DoCompute(Operator& op, 
                          const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::hardswish(inputs.at(0),
                     op->instantiation_ctx().stream_index, 
                     outputs.at(0));
}

TensorList HardswishOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeHardswishGradientOp(op->input(0), grad_outputs.at(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void HardswishGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hspmd::impl::HardswishGradient, inputs.at(0),
                               inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeHardswishOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<HardswishOpImpl>(),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeHardswishGradientOp(Tensor input, Tensor grad_output,
                               OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<HardswishGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hspmd
