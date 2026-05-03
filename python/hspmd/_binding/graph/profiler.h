#pragma once

#include <Python.h>
#include "hspmd/_binding/core/ndarray.h"
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/_binding/utils/numpy.h"
#include "hspmd/_binding/utils/pybind_common.h"
#include "hspmd/impl/profiler/profiler.h"

namespace hspmd {
namespace impl {

struct PyProfile {
  PyObject_HEAD;
  ProfileId profile_id;
};

extern PyTypeObject* PyProfile_Type;

PyObject* PyProfile_New(ProfileId profile_id);

void AddPyProfileTypeToModule(py::module_& module);
void AddProfileContextManagingFunctionsToModule(py::module_& m);

} // namespace impl
} // namespace hspmd
