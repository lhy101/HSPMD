#include "hspmd/core/ndarray.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/cuda_utils.h"
#include "hspmd/impl/utils/offset_calculator.cuh"
#include "hspmd/impl/kernel/Vectorized.cuh"

namespace hspmd {
namespace impl {

void WhereCuda(const NDArray& cond, const NDArray& inputA,
               const NDArray& inputB, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_SAME_SHAPE(inputA, inputB);

  size_t size = cond->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "WhereCuda", [&]() {
      using InType = std::tuple<int64_t, spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({cond, inputA, inputB}, {output}, size, stream,
        [] __device__ (int64_t cond, spec_t a, spec_t b) -> spec_t { return bool(cond) ? a : b; });
  });
  NDArray::MarkUsedBy({cond, inputA, inputB, output}, stream);
}

} // namespace impl
} // namespace hspmd
