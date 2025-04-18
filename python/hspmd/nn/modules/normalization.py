import hspmd
from hspmd import Tensor
import numbers
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional, List

__all__ = [
    'LayerNorm',
]

class LayerNorm(Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, device_groups = None) -> None:
        with hspmd.graph("define_and_run"):
            super(LayerNorm, self).__init__()
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = [normalized_shape]  # type: ignore[assignment]
            self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.weight = hspmd.ones(self.normalized_shape, requires_grad=True, device_groups=device_groups)    #hspmd.nn.Parameter(hspmd.ones(self.normalized_shape))
            self.bias = hspmd.zeros(self.normalized_shape, requires_grad=True, device_groups=device_groups)    #hspmd.nn.Parameter(hspmd.zeros(self.normalized_shape))

    def forward(self, input: Tensor) -> Tensor:
        return hspmd.layer_norm(input, self.weight, self.bias, self.normalized_shape, self.eps)[0]

