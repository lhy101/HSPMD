#include "hspmd/core/ndarray.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/cuda_utils.h"
#include "hspmd/impl/utils/offset_calculator.cuh"
#include "hspmd/impl/kernel/Vectorized.cuh"

namespace hspmd {
namespace impl {

void BoolCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(output->dtype() == DataType::BOOL);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BoolCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return float(x > 0) ? spec_t(1) : spec_t(0);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hspmd
