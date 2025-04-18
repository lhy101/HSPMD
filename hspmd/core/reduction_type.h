#pragma once

#include "hspmd/common/macros.h"

namespace hspmd {

enum class ReductionType : int8_t {
  SUM = 0,
  MEAN,
  PROD,
  MAX,
  MIN,
  NONE,
  NUM_REDUCTION_TYPES
};

constexpr ReductionType kSUM = ReductionType::SUM;
constexpr ReductionType kMEAN = ReductionType::MEAN;
constexpr ReductionType kPROD = ReductionType::PROD;
constexpr ReductionType kMAX = ReductionType::MAX;
constexpr ReductionType kMIN = ReductionType::MIN;
constexpr ReductionType kNONE = ReductionType::NONE;
constexpr int16_t NUM_REDUCTION_TYPES =
  static_cast<int16_t>(ReductionType::NUM_REDUCTION_TYPES);

std::string ReductionType2Str(const ReductionType&);
ReductionType Str2ReductionType(const std::string&);
std::ostream& operator<<(std::ostream&, const ReductionType&);

} // namespace hspmd

namespace std {
template <>
struct hash<hspmd::ReductionType> {
  std::size_t operator()(hspmd::ReductionType red_type) const noexcept {
    return std::hash<int>()(static_cast<int>(red_type));
  }
};

inline std::string to_string(hspmd::ReductionType red_type) {
  return hspmd::ReductionType2Str(red_type);
}
} // namespace std
