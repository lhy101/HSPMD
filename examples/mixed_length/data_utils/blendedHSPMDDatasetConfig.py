# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


from .tokenizer.tokenizer import HSPMDTokenizer
from .utils import Split, normalize

logger = logging.getLogger(__name__)


@dataclass
class BlendedHSPMDDatasetConfig:

    random_seed: int

    sequence_length: int

    blend: Optional[List[str]] = None

    blend_per_split: Optional[List[Optional[List[str]]]] = None

    split: Optional[str] = None

    split_matrix: Optional[List[Tuple[float, float]]] = field(init=False, default=None)

    path_to_cache: Optional[str] = None

    tokenizer: Optional[HSPMDTokenizer] = None

    def __post_init__(self) -> None:

        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.blend is None, "blend and blend_per_split are incompatible"
            assert self.split is None, "split and blend_per_split are incompatible"
            assert len(self.blend_per_split) == len(
                Split
            ), f"blend_per_split must contain {len(Split)} blends"
        else:
            assert (
                self.blend is not None
            ), "one of either blend or blend_per_split must be provided"
            assert self.split is not None, "both blend and split must be provided"
            split_vector = parse_and_normalize_split(self.split)
            self.split_matrix = convert_split_vector_to_split_matrix(split_vector)
            print(f"Let split_matrix = {self.split_matrix}")


def parse_and_normalize_split(split: str) -> List[float]:

    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    split = normalize(split)

    return split


def convert_split_vector_to_split_matrix(
    vector_a: List[float], vector_b: Optional[List[float]] = None
) -> List[Optional[Tuple[float, float]]]:
    if vector_b is None:
        vector_b = vector_a

    # [.900, .090, .010] -> [0.00, .900, .990, 100]
    expansion_a = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_a])
    expansion_b = functools.reduce(lambda a, b: a + [a[len(a) - 1] + b], [[0], *vector_b])

    # [0.00, .900, .990, 100.0] -> [(0.00, .900), (.900, .990), (.990, 100)]
    bookends_a = list(zip(expansion_a[:-1], expansion_a[1:]))
    bookends_b = list(zip(expansion_b[:-1], expansion_b[1:]))

    # gather per-split overlap or None
    matrix = []
    for bookend_a, bookend_b in zip(bookends_a, bookends_b):
        if min(bookend_a[1], bookend_b[1]) <= max(bookend_a[0], bookend_b[0]):
            overlap = None
        else:
            overlap = (max(bookend_a[0], bookend_b[0]), min(bookend_a[1], bookend_b[1]))
        matrix.append(overlap)

    return matrix
