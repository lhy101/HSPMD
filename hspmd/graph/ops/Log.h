#pragma once

#include "hspmd/graph/operator.h"
#include "hspmd/graph/utils/tensor_utils.h"
#include "hspmd/graph/ops/Unary.h"

namespace hspmd {
namespace graph {

class LogOpImpl;
class LogOp;
class LogGradientOpImpl;
class LogGradientOp;

class LogOpImpl final : public UnaryOpImpl {
 private:
  friend class LogOp;
  struct constructor_access_key {};

 public:
  LogOpImpl(bool inplace)
  : UnaryOpImpl(quote(LogOp), inplace) {
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

Tensor MakeLogOp(Tensor input, OpMeta op_meta = OpMeta());
Tensor MakeLogInplaceOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hspmd
