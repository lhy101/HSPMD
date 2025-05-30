#include "hspmd/core/ndarray.h"
#include "hspmd/core/stream.h"
#include "hspmd/impl/utils/common_utils.h"
#include "hspmd/impl/utils/omp_utils.h"
#include "hspmd/impl/stream/CPUStream.h"
#include <cmath>

namespace hspmd {
namespace impl {

template <typename spec_t>
void exp_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = std::exp(input[idx]);
}

template <typename spec_t>
void exp_cpu(const spec_t* input, size_t size, spec_t* output,
                          int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hspmd::impl::get_index(idx, ndims, stride, c_shape);
    output[i_idx] = std::exp(input[i_idx]);
  }
}

template <typename spec_t>
void exp_cpu(const spec_t* input, size_t size, spec_t* output,
             int64_t ndims, const int64_t* stride, const int64_t* stride_out,
             const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hspmd::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hspmd::impl::get_index(idx, ndims, stride_out, c_shape);
    output[o_idx] = std::exp(input[i_idx]);
  }
}

void ExpCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ExpCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      if (input->is_contiguous() && output->is_contiguous()) {
        exp_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>());
      }
      else {
        exp_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                        output->data_ptr<spec_t>(), input->ndim(),
                        input->stride().data(), output->stride().data(),
                        input->shape().data());
      }
      },
      "Exp");  
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hspmd
