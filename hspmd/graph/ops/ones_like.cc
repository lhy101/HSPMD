#include "hspmd/graph/ops/ones_like.h"
#include "hspmd/graph/headers.h"

namespace hspmd {
namespace graph {

Tensor MakeOnesLikeOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<OnesLikeOpImpl>(), {std::move(input)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hspmd
