#pragma once

#include "hspmd/graph/operator.h"
#include "hspmd/graph/ops/scalars_like.h"

namespace hspmd {
namespace graph {

class ZerosLikeOpImpl final : public ScalarsLikeOpImpl {
 public:
  ZerosLikeOpImpl()
  : ScalarsLikeOpImpl(quote(ZerosLikeOp), 0) {}

 public:
  bool operator==(const OpInterface& rhs) const {
    return ScalarsLikeOpImpl::operator==(rhs);
  }
};

Tensor MakeZerosLikeOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hspmd
