#include "hspmd/graph/ops/Floor.h"
#include "hspmd/graph/ops/zeros_like.h"
#include "hspmd/graph/headers.h"
#include "hspmd/graph/ops/kernel_links.h"

namespace hspmd {
namespace graph {

void FloorOpImpl::DoCompute(Operator& op, 
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  NDArray::floor(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList FloorOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList FloorOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  return {op->requires_grad(0) ? MakeZerosLikeOp(grad_outputs.at(0), g_op_meta)
                               : Tensor()};
}

Tensor MakeFloorOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<FloorOpImpl>(false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeFloorInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT16;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<FloorOpImpl>(true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hspmd
