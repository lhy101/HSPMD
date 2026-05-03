#include "hspmd/core/ndarray.h"
#include "hspmd/impl/stream/CUDAStream.h"
#include "hspmd/impl/utils/numeric_utils.h"
#include "hspmd/impl/kernel/Reduce.cuh"

namespace hspmd {
namespace impl {

template <typename acc_t>
struct MaxOp {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (hspmd::_isnan(a) || a > b) ? a : b;
  }
};

template <typename spec_t, typename acc_t = spec_t, typename out_t = spec_t>
struct max_functor {
  void operator()(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                  int64_t num_ax, const Stream& stream) {
    launch_reduce_kernel<spec_t, out_t, acc_t>(in_arr, out_arr, axes, num_ax,
                                               func_wrapper<acc_t, acc_t>(MaxOp<acc_t>()),
                                               hspmd::numeric_limits<acc_t>::lower_bound(),
                                               stream);
  }
};

void ReduceMaxCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  if (in_arr->dtype() == DataType::FLOAT16) {
    max_functor<hspmd::float16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else if (in_arr->dtype() == DataType::BFLOAT16) {
    max_functor<hspmd::bfloat16, float>{}(in_arr, out_arr, axes, num_ax, stream);
  } else {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMaxCuda", [&]() {
          max_functor<spec_t>{}(in_arr, out_arr, axes, num_ax, stream);
      });
  }
}

} // namespace impl
} // namespace hspmd
