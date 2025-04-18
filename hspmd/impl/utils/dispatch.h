#pragma once

#include "hspmd/common/macros.h"
#include "hspmd/core/dtype.h"

/******************************************************
 * Dispatch Utils. Learned from PyTorch.
 ******************************************************/

#define HT_DISPATH_CASE(DATA_TYPE, SPEC_TYPE, ...)                             \
  case DATA_TYPE: {                                                            \
    using SPEC_TYPE = hspmd::DataType2SpecMeta<DATA_TYPE>::spec_type;           \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define HT_DISPATH_SWITCH(DTYPE, NAME, ...)                                    \
  do {                                                                         \
    const auto& _dtype = DTYPE;                                                \
    switch (_dtype) {                                                          \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << NAME << "\" is not implemented for "     \
                           << "\"" << _dtype << "\"";                          \
    }                                                                          \
  } while (0)

#define HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, ...)                         \
  HT_DISPATH_CASE(hspmd::DataType::FLOAT16, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hspmd::DataType::BFLOAT16, SPEC_TYPE, __VA_ARGS__)            \
  HT_DISPATH_CASE(hspmd::DataType::FLOAT32, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hspmd::DataType::FLOAT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATH_CASE_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, ...)          \
  HT_DISPATH_CASE(hspmd::DataType::FLOAT32, SPEC_TYPE, __VA_ARGS__)             \
  HT_DISPATH_CASE(hspmd::DataType::FLOAT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_FLOATING_TYPES(DTYPE, SPEC_TYPE, NAME, ...)                \
  HT_DISPATH_SWITCH(DTYPE, NAME,                                               \
                    HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, ...)                          \
  HT_DISPATH_CASE(hspmd::DataType::UINT8, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hspmd::DataType::INT8, SPEC_TYPE, __VA_ARGS__)                \
  HT_DISPATH_CASE(hspmd::DataType::INT16, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hspmd::DataType::INT32, SPEC_TYPE, __VA_ARGS__)               \
  HT_DISPATH_CASE(hspmd::DataType::INT64, SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_TYPES(DTYPE, SPEC_TYPE, NAME, ...)                 \
  HT_DISPATH_SWITCH(DTYPE, NAME,                                               \
                    HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(SPEC_TYPE, ...)             \
  HT_DISPATH_CASE_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__)                       \
  HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, ...)             \
  HT_DISPATH_CASE_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, __VA_ARGS__)                       \
  HT_DISPATH_CASE_INTEGER_TYPES(SPEC_TYPE, __VA_ARGS__)

#define HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(DTYPE, SPEC_TYPE, NAME, ...)    \
  HT_DISPATH_SWITCH(                                                           \
    DTYPE, NAME,                                                               \
    HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES(SPEC_TYPE, __VA_ARGS__))

#define HT_DISPATCH_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(DTYPE, SPEC_TYPE, NAME, ...)    \
  HT_DISPATH_SWITCH(                                                                          \
    DTYPE, NAME,                                                                              \
    HT_DISPATH_CASE_INTEGER_AND_FLOATING_TYPES_EXCEPT_FLOAT16(SPEC_TYPE, __VA_ARGS__))

/******************************************************
 * Dispatch Utils for Paired Types.
 * The current implement is tedious and hard to read.
 * Try to make it elegant in the future.
 ******************************************************/

namespace hspmd {
namespace __paired_dtype_anon {
constexpr int paired_dtype(DataType t1, DataType t2) {
  return (static_cast<int>(t1) << 8) | static_cast<int>(t2);
};
} // namespace __paired_dtype_anon
} // namespace hspmd

#define HT_DISPATH_PAIRED_CASE(DTYPE_A, DTYPE_B, SPEC_A, SPEC_B, ...)          \
  case hspmd::__paired_dtype_anon::paired_dtype(DTYPE_A, DTYPE_B): {            \
    using SPEC_A = hspmd::DataType2SpecMeta<DTYPE_A>::spec_type;                \
    using SPEC_B = hspmd::DataType2SpecMeta<DTYPE_B>::spec_type;                \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define HT_DISPATH_PAIRED_SWITCH(DTYPE_A, DTYPE_B, NAME, ...)                  \
  do {                                                                         \
    const auto& _dtype_a = DTYPE_A;                                            \
    const auto& _dtype_b = DTYPE_B;                                            \
    switch (hspmd::__paired_dtype_anon::paired_dtype(_dtype_a, _dtype_b)) {     \
      __VA_ARGS__                                                              \
      default:                                                                 \
        HT_NOT_IMPLEMENTED << "\"" << NAME << "\" is not implemented for "     \
                           << "\"" << _dtype_a << "\" x \"" << _dtype_b        \
                           << "\"";                                            \
    }                                                                          \
  } while (0)

#define HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, ...)             \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT16, hspmd::DataType::BFLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT16, hspmd::DataType::FLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT16, hspmd::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::BFLOAT16, hspmd::DataType::FLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::BFLOAT16, hspmd::DataType::BFLOAT16,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::BFLOAT16, hspmd::DataType::FLOAT32,    \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::BFLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::FLOAT16,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::FLOAT64,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::FLOAT32,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::FLOAT64,     \
                         SPEC_A, SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_FLOATING_TYPES(DTYPE_A, DTYPE_B, SPEC_A, SPEC_B,    \
                                          NAME, ...)                           \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, __VA_ARGS__))

#define HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, ...)       \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::INT8, SPEC_A,   \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::INT16, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::INT32, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::INT64, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::INT8, SPEC_A,  \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::INT16, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::INT32, SPEC_A, \
                         SPEC_B, __VA_ARGS__)                                  \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::INT64, SPEC_A, \
                         SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_SIGNED_INTEGER_TYPES(DTYPE_A, DTYPE_B, SPEC_A,      \
                                                SPEC_B, NAME, ...)             \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, __VA_ARGS__))

#define HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_AND_FLOATING_TYPES(SPEC_A,       \
                                                                 SPEC_B, ...)  \
  HT_DISPATH_PAIRED_CASE_FLOATING_TYPES(SPEC_A, SPEC_B, __VA_ARGS__)           \
  HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_TYPES(SPEC_A, SPEC_B, __VA_ARGS__)     \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::FLOAT32,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT8, hspmd::DataType::FLOAT64,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT16, hspmd::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT32, hspmd::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::FLOAT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::INT64, hspmd::DataType::FLOAT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::INT8,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::INT16,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::INT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT32, hspmd::DataType::INT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::INT8,        \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::INT16,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::INT32,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)                          \
  HT_DISPATH_PAIRED_CASE(hspmd::DataType::FLOAT64, hspmd::DataType::INT64,       \
                         SPEC_A, SPEC_B, __VA_ARGS__)

#define HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(                  \
  DTYPE_A, DTYPE_B, SPEC_A, SPEC_B, NAME, ...)                                 \
  HT_DISPATH_PAIRED_SWITCH(                                                    \
    DTYPE_A, DTYPE_B, NAME,                                                    \
    HT_DISPATH_PAIRED_CASE_SIGNED_INTEGER_AND_FLOATING_TYPES(SPEC_A, SPEC_B,   \
                                                             __VA_ARGS__))
