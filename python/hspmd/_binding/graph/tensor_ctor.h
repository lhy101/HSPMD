#pragma once

#include <Python.h>
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/graph/init/initializer.h"

namespace hspmd {
namespace graph {

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs);

} // namespace graph
} // namespace hspmd
