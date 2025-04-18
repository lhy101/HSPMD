#pragma once

#include "hspmd/graph/operator.h"
#include "hspmd/graph/ops/scalars_like.h"

namespace hspmd {
namespace graph {

class OnesLikeOpImpl final : public ScalarsLikeOpImpl {
 public:
  OnesLikeOpImpl()
  : ScalarsLikeOpImpl(quote(OnesLikeOp), 1) {}

 public:
  bool operator==(const OpInterface& rhs) const {
    return ScalarsLikeOpImpl::operator==(rhs);
  }
};

Tensor MakeOnesLikeOp(Tensor input, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hspmd
