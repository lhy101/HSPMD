#pragma once

#include "hspmd/graph/operator.h"
#include "hspmd/graph/utils/tensor_utils.h"
#include "hspmd/graph/ops/Unary.h"

namespace hspmd {
namespace graph {

class CeilOpImpl;
class CeilOp;

class CeilOpImpl final : public UnaryOpImpl {
 private:
  friend class CeilOp;
  struct constructor_access_key {};

 public:
  CeilOpImpl(bool inplace)
  : UnaryOpImpl(quote(CeilOp), inplace) {
  }

 protected:
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs);
  }
};

Tensor MakeCeilOp(Tensor input, OpMeta op_meta = OpMeta());

Tensor MakeCeilInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hspmd
