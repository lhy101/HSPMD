#pragma once

#include <Python.h>
#include "hspmd/impl/communication/comm_group.h"
#include "hspmd/impl/communication/nccl_comm_group.h"
#include "hspmd/core/stream.h"
#include "hspmd/_binding/utils/pybind_common.h"

namespace hspmd {

struct PyCommGroup {
  PyObject_HEAD;
};

extern PyTypeObject* PyCommGroup_Type;

void AddPyCommGroupTypeToModule(py::module_& module);

} // namespace hspmd
