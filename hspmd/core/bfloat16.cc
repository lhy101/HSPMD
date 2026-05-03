#include "hspmd/core/bfloat16.h"
#include <iostream>
namespace hspmd {
std::ostream& operator<<(std::ostream& out, const bfloat16& value) {
  out << (float)value;
  return out;
}
} //namespace hspmd
