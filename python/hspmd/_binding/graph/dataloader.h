#pragma once

#include <Python.h>
#include "hspmd/graph/operator.h"
#include "hspmd/graph/data/dataloader.h"
#include "hspmd/graph/tensor.h"
#include "hspmd/_binding/utils/pybind_common.h"

namespace hspmd {
namespace graph {

struct PyDataloader {
  PyObject_HEAD;
  Dataloader dataloader;
};

extern PyTypeObject* PyDataloader_Type;

inline bool PyDataloader_Check(PyObject* obj) {
  return PyDataloader_Type && PyObject_TypeCheck(obj, PyDataloader_Type);
}

inline bool PyDataloader_CheckExact(PyObject* obj) {
  return PyDataloader_Type && obj->ob_type == PyDataloader_Type;
}

PyObject* PyDataloader_New(Dataloader&& tensor, bool return_none_if_undefined = true);

void AddPyDataloaderTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyDataloader(PyObject* obj) {
  return PyDataloader_Check(obj);
}

inline Dataloader Dataloader_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyDataloader*>(obj)->dataloader;
}

} // namespace graph
} // namespace hspmd
