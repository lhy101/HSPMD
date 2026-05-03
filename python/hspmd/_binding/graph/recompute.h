#pragma once

#include <Python.h>
#include "hspmd/graph/recompute/recompute.h"
#include "hspmd/_binding/core/ndarray.h"
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/_binding/utils/numpy.h"
#include "hspmd/_binding/utils/pybind_common.h"

namespace hspmd {
namespace graph {

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddRecomputeContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hspmd
