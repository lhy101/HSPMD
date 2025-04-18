#include "hspmd/core/float16.h"
#include <iostream>
namespace hspmd {
std::ostream& operator<<(std::ostream& out, const float16& value) {
  out << (float)value;
  return out;
}
} //namespace hspmd
