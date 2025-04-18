#include "hspmd/core/ndarray.h"
#include "hspmd/core/memory_pool.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/cuda_utils.h"
#include "hspmd/impl/kernel/Binary.cuh"
#include "hspmd/impl/utils/offset_calculator.cuh"
#include "hspmd/impl/kernel/Vectorized.cuh"

#define BINARYCONST(input, value, output, op, stream)                                                   \
        HT_ASSERT_CUDA_DEVICE(input);                                                                   \
        HT_ASSERT_SAME_DEVICE(input, output);                                                           \
        size_t size = input->numel();                                                                   \
        if (size == 0)                                                                                  \
          return;                                                                                       \
        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(                                                         \
          input->dtype(), spec_t, "BinaryConstCuda", [&]() {                                            \
            using InType = std::tuple<spec_t>;                                                          \
            using OutType = thrust::tuple<spec_t>;                                                      \
            launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,                        \
                                               [=] __device__ (spec_t x) -> spec_t {                    \
                                                 return op<spec_t, spec_t>{}                            \
                                                        (static_cast<spec_t>(value), x);                \
                                               });                                                      \
          });                                                                                           \
        NDArray::MarkUsedBy({input, output}, stream);

namespace hspmd {
namespace impl {

void AddConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kplus, stream)
}

void SubConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kminus, stream)
}

void MulConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  BINARYCONST(input, value, output, kmultiplies, stream)
}

void DivConstCuda(const NDArray& input, double value,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT(value != 0) << "Divided by 0.";
  BINARYCONST(input, value, output, kdivides, stream)
}


} // namespace impl
} // namespace hspmd
