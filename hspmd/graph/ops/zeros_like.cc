#include "hspmd/graph/ops/zeros_like.h"
#include "hspmd/graph/headers.h"

namespace hspmd {
namespace graph {

Tensor MakeZerosLikeOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ZerosLikeOpImpl>(), {std::move(input)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hspmd
