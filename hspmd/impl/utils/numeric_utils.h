#pragma once

#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <float.h>

namespace hspmd {

template <typename T>
struct numeric_limits {
};

template <>
struct numeric_limits<bool> {
  static inline __host__ __device__ bool lowest() { return false; }
  static inline __host__ __device__ bool max() { return true; }
  static inline __host__ __device__ bool lower_bound() { return false; }
  static inline __host__ __device__ bool upper_bound() { return true; }
};

template <>
struct numeric_limits<uint8_t> {
  static inline __host__ __device__ uint8_t lowest() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UINT8_MAX; }
  static inline __host__ __device__ uint8_t lower_bound() { return 0; }
  static inline __host__ __device__ uint8_t upper_bound() { return UINT8_MAX; }
};

template <>
struct numeric_limits<int8_t> {
  static inline __host__ __device__ int8_t lowest() { return INT8_MIN; }
  static inline __host__ __device__ int8_t max() { return INT8_MAX; }
  static inline __host__ __device__ int8_t lower_bound() { return INT8_MIN; }
  static inline __host__ __device__ int8_t upper_bound() { return INT8_MAX; }
};

template <>
struct numeric_limits<int16_t> {
  static inline __host__ __device__ int16_t lowest() { return INT16_MIN; }
  static inline __host__ __device__ int16_t max() { return INT16_MAX; }
  static inline __host__ __device__ int16_t lower_bound() { return INT16_MIN; }
  static inline __host__ __device__ int16_t upper_bound() { return INT16_MAX; }
};

template <>
struct numeric_limits<int32_t> {
  static inline __host__ __device__ int32_t lowest() { return INT32_MIN; }
  static inline __host__ __device__ int32_t max() { return INT32_MAX; }
  static inline __host__ __device__ int32_t lower_bound() { return INT32_MIN; }
  static inline __host__ __device__ int32_t upper_bound() { return INT32_MAX; }
};

template <>
struct numeric_limits<int64_t> {
  static inline __host__ __device__ int64_t lowest() { return INT64_MIN; }
  static inline __host__ __device__ int64_t max() { return INT64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return INT64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return INT64_MAX; }
};

template <>
struct numeric_limits<hspmd::float16> {
  static inline __host__ __device__ hspmd::float16 lowest() { return hspmd::float16(0xFBFF, hspmd::float16::from_bits()); }
  static inline __host__ __device__ hspmd::float16 max() { return hspmd::float16(0x7BFF, hspmd::float16::from_bits()); }
  static inline __host__ __device__ hspmd::float16 lower_bound() { return hspmd::float16(0xFC00, hspmd::float16::from_bits()); }
  static inline __host__ __device__ hspmd::float16 upper_bound() { return hspmd::float16(0x7C00, hspmd::float16::from_bits()); }
};

template <>
struct numeric_limits<hspmd::bfloat16> {
  static inline __host__ __device__ hspmd::bfloat16 lowest() { return hspmd::bfloat16(0xFF7F, hspmd::bfloat16::from_bits()); }
  static inline __host__ __device__ hspmd::bfloat16 max() { return hspmd::bfloat16(0x7F7F, hspmd::bfloat16::from_bits()); }
  static inline __host__ __device__ hspmd::bfloat16 lower_bound() { return hspmd::bfloat16(0xFF80, hspmd::bfloat16::from_bits()); }
  static inline __host__ __device__ hspmd::bfloat16 upper_bound() { return hspmd::bfloat16(0x7F80, hspmd::bfloat16::from_bits()); }
};

template <>
struct numeric_limits<float> {
  static inline __host__ __device__ float lowest() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }
  static inline __host__ __device__ float lower_bound() { return -static_cast<float>(INFINITY); }
  static inline __host__ __device__ float upper_bound() { return static_cast<float>(INFINITY); }
};

template <>
struct numeric_limits<double> {
  static inline __host__ __device__ double lowest() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }
  static inline __host__ __device__ double lower_bound() { return -INFINITY; }
  static inline __host__ __device__ double upper_bound() { return INFINITY; }
};

template <
  typename T,
  typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline __host__ __device__ bool _isnan(T /*val*/) {
  return false;
}

template <
  typename T,
  typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline __host__ __device__ bool _isnan(T val) {
#if defined(__CUDACC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <
  typename T,
  typename std::enable_if<std::is_same<T, hspmd::float16>::value, int>::type = 0>
inline __host__ __device__ bool _isnan(T val) {
  return hspmd::_isnan(static_cast<float>(val));
}

template <
  typename T,
  typename std::enable_if<std::is_same<T, hspmd::bfloat16>::value, int>::type = 0>
inline __host__ __device__ bool _isnan(T val) {
  return hspmd::_isnan(static_cast<float>(val));
}

template <
  typename T,
  typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline __host__ __device__ bool _isinf(T /*val*/) {
  return false;
}

template <
  typename T,
  typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline __host__ __device__ bool _isinf(T val) {
#if defined(__CUDACC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}

template <
  typename T,
  typename std::enable_if<std::is_same<T, hspmd::float16>::value, int>::type = 0>
inline __host__ __device__ bool _isinf(T val) {
  return hspmd::_isinf(static_cast<float>(val));
}

template <
  typename T,
  typename std::enable_if<std::is_same<T, hspmd::bfloat16>::value, int>::type = 0>
inline __host__ __device__ bool _isinf(T val) {
  return hspmd::_isinf(static_cast<float>(val));
}

} // namespace hspmd