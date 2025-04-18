#pragma once

#include "hspmd/graph/operator.h"
#include "hspmd/graph/utils/tensor_utils.h"
#include "hspmd/graph/ops/Unary.h"

namespace hspmd {
namespace graph {

class ExpOpImpl;
class ExpOp;
class ExpGradientOpImpl;
class ExpGradientOp;

class ExpOpImpl final : public UnaryOpImpl {
 private:
  friend class ExpOp;
  struct constructor_access_key {};

 public:
  ExpOpImpl(bool inplace)
  : UnaryOpImpl(quote(ExpOp), inplace) {
  }

 protected:
  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeExpOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeExpInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hspmd
