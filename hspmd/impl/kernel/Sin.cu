#include "hspmd/core/ndarray.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/cuda_utils.h"
#include "hspmd/impl/utils/cuda_math.h"
#include "hspmd/impl/utils/offset_calculator.cuh"
#include "hspmd/impl/kernel/Vectorized.cuh"

namespace hspmd {
namespace impl {

void SinCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SinCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hspmd::cuda::cuda_sin(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void CosCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "CosCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hspmd::cuda::cuda_cos(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SinGradientCuda(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SinGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, size, stream,
                                         [] __device__ (spec_t x, spec_t y) -> spec_t {
                                           return y * hspmd::cuda::cuda_cos(x);
                                          });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void CosGradientCuda(const NDArray& input, const NDArray& output_grad,
                     NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "CosGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, size, stream,
                                         [] __device__ (spec_t x, spec_t y) -> spec_t {
                                           return -y * hspmd::cuda::cuda_sin(x);
                                          });
  });
  NDArray::MarkUsedBy({input, input_grad}, stream);
}

} // namespace impl
} // namespace hspmd
