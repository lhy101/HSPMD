#pragma once

#include "hspmd/core/ndarray.h"
#include "hspmd/impl/utils/ndarray_utils.h"
#include "hspmd/impl/utils/dispatch.h"

using hspmd::operator<<;

void assert_eq(const hspmd::NDArray& a, const hspmd::NDArray& b) {
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_SHAPE(a, b);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(a->dtype(), spec_t, "AssertEq", [&]() {
    auto stream_id = hspmd::kBlockingStream;
    auto a_cpu = hspmd::NDArray::cpu(a, stream_id);
    auto b_cpu = hspmd::NDArray::cpu(a, stream_id);
    auto* a_ptr = a_cpu->data_ptr<spec_t>();
    auto* b_ptr = b_cpu->data_ptr<spec_t>();
    size_t numel = a->numel();
    for (size_t i = 0; i < numel; i++) {
      HT_ASSERT_EQ(a_ptr[i], b_ptr[i]) << "Mismatched on position " << i << ": "
                                       << a_ptr[i] << ", " << b_ptr[i];
    }
  });
}

void assert_eq(const hspmd::NDArray& data, double value) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "AssertEq", [&]() {
      auto stream_id = hspmd::kBlockingStream;
      auto data_cpu = hspmd::NDArray::cpu(data, stream_id);
      auto* ptr = data_cpu->data_ptr<spec_t>();
      size_t numel = data->numel();
      for (size_t i = 0; i < numel; i++) {
        HT_ASSERT_EQ(ptr[i], value)
          << "Mismatched on position " << i << ": " << ptr[i] << ", " << value;
      }
    });
}

void assert_fuzzy_eq(const hspmd::NDArray& a, const hspmd::NDArray& b,
                     double atol = 1e-8, double rtol = 1e-5) {
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_SHAPE(a, b);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    a->dtype(), spec_t, "AssertFuzzyEq", [&]() {
      auto stream_id = hspmd::kBlockingStream;
      auto a_cpu = hspmd::NDArray::cpu(a, stream_id);
      auto b_cpu = hspmd::NDArray::cpu(a, stream_id);
      auto* a_ptr = a_cpu->data_ptr<spec_t>();
      auto* b_ptr = b_cpu->data_ptr<spec_t>();
      size_t numel = a->numel();
      for (size_t i = 0; i < numel; i++) {
        HT_ASSERT_FUZZY_EQ(a_ptr[i], b_ptr[i], atol, rtol)
          << "Mismatched on position " << i << ": " << a_ptr[i] << ", "
          << b_ptr[i];
      }
    });
}

void assert_fuzzy_eq(const hspmd::NDArray& data, double value,
                     double atol = 1e-8, double rtol = 1e-5) {
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "AssertFuzzyEq", [&]() {
      auto stream_id = hspmd::kBlockingStream;
      auto data_cpu = hspmd::NDArray::cpu(data, stream_id);
      auto* ptr = data_cpu->data_ptr<spec_t>();
      size_t numel = data->numel();
      for (size_t i = 0; i < numel; i++) {
        HT_ASSERT_FUZZY_EQ(ptr[i], value, atol, rtol)
          << "Mismatched on position " << i << ": " << ptr[i] << ", " << value;
      }
    });
}
