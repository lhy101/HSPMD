#include "hspmd/_binding/graph/cpu_offload.h"
#include "hspmd/_binding/constants.h"
#include "hspmd/_binding/utils/pybind_common.h"
#include "hspmd/_binding/utils/except.h"
#include "hspmd/_binding/utils/decl_utils.h"
#include "hspmd/_binding/utils/arg_parser.h"

namespace hspmd {
namespace graph {

PyObject* PyPushCPUOffloadCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({
    "push_cpu_offload_ctx(List[List[bool]] multi_cpu_offload)",
  });
  auto parsed_args = parser.parse(args, kwargs);
  if (parsed_args.signature_index() == 0) {
    ActivationCPUOffload::push_cpu_offload_enabled(parsed_args.get_bool_list_list(0));
    Py_RETURN_NONE;
  } else {
    HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args);
    __builtin_unreachable();
  }
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

PyObject* PyPopCPUOffloadCtx(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  ActivationCPUOffload::pop_cpu_offload_enabled();
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

// NOLINTNEXTLINE
PyMethodDef PyCPUOffloadCtx_methods[] = { 
  {"push_cpu_offload_ctx", (PyCFunction) PyPushCPUOffloadCtx, METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pop_cpu_offload_ctx", (PyCFunction) PyPopCPUOffloadCtx, METH_NOARGS, nullptr},
  {nullptr}
};

void AddCPUOffloadContextManagingFunctionsToModule(py::module_& m) {
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(m.ptr(), PyCPUOffloadCtx_methods))
    << "Failed to add cpu offload context managing methods";
}

} // namespace graph
} // namespace hspmd
