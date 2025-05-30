#include "hspmd/graph/ops/Maskedfill.h"
#include "hspmd/graph/ops/zeros_like.h"
#include "hspmd/graph/headers.h"
#include "hspmd/graph/ops/kernel_links.h"

namespace hspmd {
namespace graph {

void MaskedfillOpImpl::DoCompute(Operator& op, 
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hspmd::impl::Maskedfill,
                                  inputs.at(0), inputs.at(1), val(),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

TensorList MaskedfillOpImpl::DoGradient(Operator& op, 
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeMaskedfillOp(grad_outputs.at(0), op->input(1), 0.0,
                                                           g_op_meta.set_name(op->grad_name(0)))
                                        : Tensor();

  return {grad_input, Tensor()};
}

HTShapeList MaskedfillOpImpl::DoInferShape(Operator& op, 
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(0).size() == input_shapes.at(1).size())
          << input_shapes.at(0) << " " << input_shapes.at(1);
  return {input_shapes.at(0)};
}

Tensor MakeMaskedfillOp(Tensor input, Tensor mask, double val,
                        OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<MaskedfillOpImpl>(val),
          {std::move(input), std::move(mask)},
          std::move(op_meta))->output(0);  
}

} // namespace graph
} // namespace hspmd
