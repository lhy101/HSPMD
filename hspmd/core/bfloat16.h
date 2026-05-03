#pragma once

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <cuda_bf16.h>
#include <cuda_bf16.hpp>
#include "hspmd/core/float16.h"
#include "hspmd/common/logging.h"
#include "hspmd/common/except.h"

#if defined(__CUDACC__)
#define HSPMD_HOSTDEVICE __host__ __device__ __inline__
#define HSPMD_HOST __host__  __inline__
#define HSPMD_DEVICE __device__  __inline__
#else
#define HSPMD_HOSTDEVICE inline
#define HSPMD_DEVICE inline
#define HSPMD_HOST inline
#endif /* defined(__CUDACC__) */

namespace hspmd {
HSPMD_HOSTDEVICE float fp32_from_bits16(uint16_t bits) {
  float res = 0;
  uint32_t tmp = bits;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

HSPMD_HOSTDEVICE uint16_t fp32_to_bits16(float f) {
  uint32_t res = 0;
  std::memcpy(&res, &f, sizeof(res));
  return res >> 16;
}

HSPMD_HOSTDEVICE uint16_t fp32_to_bf16(float f) {
  if (std::isnan(f)) {
    return UINT16_C(0x7FC0);
  } else {
    union {
      uint32_t U32;
      float F32;
    };
    F32 = f;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

struct alignas(2) bfloat16 {
  unsigned short val;
  struct from_bits_t {};
  HSPMD_HOSTDEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }
  bfloat16() = default;
  HSPMD_HOSTDEVICE constexpr bfloat16(unsigned short bits, from_bits_t) : val(bits){};
  // HSPMD_HOSTDEVICE bfloat16(int value);
  // HSPMD_HOSTDEVICE operator int() const;
  HSPMD_HOSTDEVICE bfloat16(float value);
  HSPMD_HOSTDEVICE operator float() const;
  HSPMD_HOSTDEVICE bfloat16(double value);
  HSPMD_HOSTDEVICE explicit operator float16() const;
  HSPMD_HOSTDEVICE explicit operator double() const;
  HSPMD_HOSTDEVICE explicit operator int() const;
  HSPMD_HOSTDEVICE explicit operator int64_t() const;
  HSPMD_HOSTDEVICE explicit operator size_t() const;
  HSPMD_HOSTDEVICE bfloat16(int value);
  HSPMD_HOSTDEVICE bfloat16(float16 value);
  HSPMD_HOSTDEVICE bfloat16(int64_t value);
  HSPMD_HOSTDEVICE bfloat16(size_t value);
  #if defined(__CUDACC__) 
  HSPMD_HOSTDEVICE bfloat16(const __nv_bfloat16& value);
  HSPMD_HOSTDEVICE operator __nv_bfloat16() const;
  HSPMD_HOSTDEVICE __nv_bfloat16 to_bf16() const;
  #endif
  HSPMD_HOSTDEVICE bfloat16 &operator=(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); return *this; }
  HSPMD_HOSTDEVICE bfloat16 &operator=(const float f) { val = bfloat16(f).val; return *this; }
  HSPMD_HOSTDEVICE bfloat16 &operator=(const float16 f) { val = bfloat16(float(f)).val; return *this; }
  HSPMD_HOSTDEVICE bfloat16 &operator=(const bfloat16 h) { val = h.val; return *this; }
};

HSPMD_HOSTDEVICE bfloat16::bfloat16(float value) {
  val = hspmd::fp32_to_bf16(value);
}

HSPMD_HOSTDEVICE bfloat16::bfloat16(float16 value) {
  val = hspmd::fp32_to_bf16(static_cast<float>(value));
}

HSPMD_HOSTDEVICE bfloat16::operator float() const {
  return hspmd::fp32_from_bits16(val);
}

HSPMD_HOSTDEVICE bfloat16::bfloat16(double value) {
  val = hspmd::fp32_to_bf16(float(value));
}

HSPMD_HOSTDEVICE bfloat16::operator float16() const {
  return static_cast<float16>(hspmd::fp32_from_bits16(val));
}

HSPMD_HOSTDEVICE bfloat16::operator double() const {
  return static_cast<double>(hspmd::fp32_from_bits16(val));
}

HSPMD_HOSTDEVICE bfloat16::operator int() const {
  return static_cast<int>(hspmd::fp32_from_bits16(val));
}

HSPMD_HOSTDEVICE bfloat16::operator int64_t() const {
  return static_cast<int64_t>(hspmd::fp32_from_bits16(val));
}

HSPMD_HOSTDEVICE bfloat16::operator size_t() const {
  return static_cast<size_t>(hspmd::fp32_from_bits16(val));
}

HSPMD_HOSTDEVICE bfloat16::bfloat16(int value) {
  val = hspmd::fp32_to_bf16(float(value));
}

HSPMD_HOSTDEVICE bfloat16::bfloat16(int64_t value) {
  val = hspmd::fp32_to_bf16(float(value));
}

HSPMD_HOSTDEVICE bfloat16::bfloat16(size_t value) {
  val = hspmd::fp32_to_bf16(float(value));
}

#if defined(__CUDACC__) 
HSPMD_HOSTDEVICE bfloat16::bfloat16(const __nv_bfloat16& value) {
  val = *reinterpret_cast<const unsigned short*>(&value);
}
HSPMD_HOSTDEVICE bfloat16::operator __nv_bfloat16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&val);
}
HSPMD_HOSTDEVICE __nv_bfloat16 bfloat16::to_bf16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&val);
}
#endif

/// Arithmetic
HSPMD_DEVICE bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

HSPMD_DEVICE bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

HSPMD_DEVICE bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

HSPMD_DEVICE bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  // HT_ASSERT(static_cast<float>(b) != 0)
  // << "Divided by zero.";
  return static_cast<float>(a) / static_cast<float>(b);
}

HSPMD_DEVICE bfloat16 operator-(const bfloat16& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

HSPMD_DEVICE bfloat16& operator+=(bfloat16& a, const bfloat16& b) {
  a = a + b;
  return a;
}

HSPMD_DEVICE bfloat16& operator-=(bfloat16& a, const bfloat16& b) {
  a = a - b;
  return a;
}

HSPMD_DEVICE bfloat16& operator*=(bfloat16& a, const bfloat16& b) {
  a = a * b;
  return a;
}

HSPMD_DEVICE bfloat16& operator/=(bfloat16& a, const bfloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

HSPMD_DEVICE float operator+(bfloat16 a, float b) {
  return static_cast<float>(a) + b;
}
HSPMD_DEVICE float operator-(bfloat16 a, float b) {
  return static_cast<float>(a) - b;
}
HSPMD_DEVICE float operator*(bfloat16 a, float b) {
  return static_cast<float>(a) * b;
}
HSPMD_DEVICE float operator/(bfloat16 a, float b) {
  return static_cast<float>(a) / b;
}

HSPMD_DEVICE float operator+(float a, bfloat16 b) {
  return a + static_cast<float>(b);
}
HSPMD_DEVICE float operator-(float a, bfloat16 b) {
  return a - static_cast<float>(b);
}
HSPMD_DEVICE float operator*(float a, bfloat16 b) {
  return a * static_cast<float>(b);
}
HSPMD_DEVICE float operator/(float a, bfloat16 b) {
  return a / static_cast<float>(b);
}

HSPMD_DEVICE float& operator+=(float& a, const bfloat16& b) {
  return a += static_cast<float>(b);
}
HSPMD_DEVICE float& operator-=(float& a, const bfloat16& b) {
  return a -= static_cast<float>(b);
}
HSPMD_DEVICE float& operator*=(float& a, const bfloat16& b) {
  return a *= static_cast<float>(b);
}
HSPMD_DEVICE float& operator/=(float& a, const bfloat16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

HSPMD_DEVICE double operator+(bfloat16 a, double b) {
  return static_cast<double>(a) + b;
}
HSPMD_DEVICE double operator-(bfloat16 a, double b) {
  return static_cast<double>(a) - b;
}
HSPMD_DEVICE double operator*(bfloat16 a, double b) {
  return static_cast<double>(a) * b;
}
HSPMD_DEVICE double operator/(bfloat16 a, double b) {
  return static_cast<double>(a) / b;
}

HSPMD_DEVICE double operator+(double a, bfloat16 b) {
  return a + static_cast<double>(b);
}
HSPMD_DEVICE double operator-(double a, bfloat16 b) {
  return a - static_cast<double>(b);
}
HSPMD_DEVICE double operator*(double a, bfloat16 b) {
  return a * static_cast<double>(b);
}
HSPMD_DEVICE double operator/(double a, bfloat16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

HSPMD_DEVICE bfloat16 operator+(bfloat16 a, int b) {
  return a + static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator-(bfloat16 a, int b) {
  return a - static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator*(bfloat16 a, int b) {
  return a * static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator/(bfloat16 a, int b) {
  return a / static_cast<bfloat16>(b);
}

HSPMD_DEVICE bfloat16 operator+(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HSPMD_DEVICE bfloat16 operator-(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HSPMD_DEVICE bfloat16 operator*(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HSPMD_DEVICE bfloat16 operator/(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

//// Arithmetic with int64_t

HSPMD_DEVICE bfloat16 operator+(bfloat16 a, int64_t b) {
  return a + static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator-(bfloat16 a, int64_t b) {
  return a - static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator*(bfloat16 a, int64_t b) {
  return a * static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator/(bfloat16 a, int64_t b) {
  return a / static_cast<bfloat16>(b);
}

HSPMD_DEVICE bfloat16 operator+(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HSPMD_DEVICE bfloat16 operator-(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HSPMD_DEVICE bfloat16 operator*(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HSPMD_DEVICE bfloat16 operator/(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

//// Arithmetic with size_t

HSPMD_DEVICE bfloat16 operator+(bfloat16 a, size_t b) {
  return a + static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator-(bfloat16 a, size_t b) {
  return a - static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator*(bfloat16 a, size_t b) {
  return a * static_cast<bfloat16>(b);
}
HSPMD_DEVICE bfloat16 operator/(bfloat16 a, size_t b) {
  return a / static_cast<bfloat16>(b);
}

HSPMD_DEVICE bfloat16 operator+(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HSPMD_DEVICE bfloat16 operator-(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HSPMD_DEVICE bfloat16 operator*(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HSPMD_DEVICE bfloat16 operator/(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

std::ostream& operator<<(std::ostream& out, const bfloat16& value);
} //namespace hspmd

namespace std {

template <>
class numeric_limits<hspmd::bfloat16> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr hspmd::bfloat16 min() {
    return hspmd::bfloat16(0x0080, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 lowest() {
    return hspmd::bfloat16(0xFF7F, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 max() {
    return hspmd::bfloat16(0x7F7F, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 epsilon() {
    return hspmd::bfloat16(0x3C00, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 round_error() {
    return hspmd::bfloat16(0x3F00, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 infinity() {
    return hspmd::bfloat16(0x7F80, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 quiet_NaN() {
    return hspmd::bfloat16(0x7FC0, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 signaling_NaN() {
    return hspmd::bfloat16(0x7F80, hspmd::bfloat16::from_bits());
  }
  static constexpr hspmd::bfloat16 denorm_min() {
    return hspmd::bfloat16(0x0001, hspmd::bfloat16::from_bits());
  }
};

} //namespace std