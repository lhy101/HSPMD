#pragma once

#include <Python.h>
#include "hspmd/graph/autocast/autocast.h"
#include "hspmd/_binding/core/ndarray.h"
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/_binding/utils/numpy.h"
#include "hspmd/_binding/utils/pybind_common.h"

namespace hspmd {
namespace graph {

struct PyAutoCast {
  PyObject_HEAD;
  AutoCastId autocast_id;
};

extern PyTypeObject* PyAutoCast_Type;

inline bool PyAutoCast_Check(PyObject* obj) {
  return PyAutoCast_Type && PyObject_TypeCheck(obj, PyAutoCast_Type);
}

inline bool PyAutoCast_CheckExact(PyObject* obj) {
  return PyAutoCast_Type && obj->ob_type == PyAutoCast_Type;
}

PyObject* PyAutoCast_New(AutoCastId graph_id);

void AddPyAutoCastTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyAutoCast(PyObject* obj) {
  return PyAutoCast_Check(obj);
}

inline AutoCastId AutoCastId_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyAutoCast*>(obj)->autocast_id;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddAutoCastContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hspmd
