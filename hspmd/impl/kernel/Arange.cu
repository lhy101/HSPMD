#include "hspmd/core/ndarray.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/cuda_utils.h"
#include "hspmd/impl/kernel/Vectorized.cuh"

namespace hspmd {
namespace impl {

void ArangeCuda(double start, double step, NDArray& output, const Stream& stream) {

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "RangeCuda", [&]() {
      using InType = std::tuple<>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel_with_idx<InType, OutType>({}, {output}, size, stream,
                                                  [start, step] __device__ (int x) -> spec_t {
                                                    return static_cast<spec_t>(start + step * size_t(x));
                                                  });
  });
  NDArray::MarkUsedBy({output}, stream);
}

} // namespace impl
} // namespace hspmd
