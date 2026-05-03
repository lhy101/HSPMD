#include "hspmd/core/stream.h"
#include "hspmd/impl/stream/CPUStream.h"
#include "hspmd/impl/stream/CUDAStream.h"

namespace hspmd {

void Stream::Sync() const {
  if (_device.is_cpu()) {
    hspmd::impl::CPUStream(*this).Sync();
  } else if (_device.is_cuda()) {
    hspmd::impl::CUDAStream(*this).Sync();
  }
}

std::ostream& operator<<(std::ostream& os, const Stream& stream) {
  os << "stream(" << stream.device()
     << ", stream_index=" << stream.stream_index() << ")";
  return os;
}

void SynchronizeAllStreams(const Device& device) {
  if (device.is_cpu()) {
    hspmd::impl::SynchronizeAllCPUStreams();
  } else if (device.is_cuda()) {
    hspmd::impl::SynchronizeAllCUDAStreams(device);
  } else {
    hspmd::impl::SynchronizeAllCPUStreams();
    hspmd::impl::SynchronizeAllCUDAStreams(device);
  }
}

} // namespace hspmd
