#pragma once

#include <Python.h>
#include <type_traits>
#include "hspmd/_binding/graph/operator.h"
#include "hspmd/_binding/graph/tensor.h"
#include "hspmd/_binding/utils/function_registry.h"
#include "hspmd/_binding/utils/pybind_common.h"
#include "hspmd/_binding/utils/except.h"
#include "hspmd/_binding/utils/decl_utils.h"
#include "hspmd/_binding/utils/arg_parser.h"
#include "hspmd/graph/ops/op_headers.h"
