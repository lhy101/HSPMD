#include "hspmd/graph/ops/RangeMask.h"
#include "hspmd/graph/headers.h"
#include "hspmd/graph/ops/kernel_links.h"

namespace hspmd {
namespace graph {

void RangeMaskOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), 
                                  hspmd::impl::RangeMask, inputs.at(0), min(), max(),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

TensorList RangeMaskOpImpl::DoGradient(Operator& op, 
                                       const TensorList& grad_outputs) const {
  return {Tensor()};
}

HTShapeList RangeMaskOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

Tensor MakeRangeMaskOp(Tensor input, int64_t min, int64_t max, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<RangeMaskOpImpl>(min, max),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hspmd
