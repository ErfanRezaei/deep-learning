from __future__ import annotations

from typing import Tuple

from ptflops import get_model_complexity_info
from torch import nn


def compute_model_complexity(
    model: nn.Module,
    input_shape: tuple[int, int, int],
) -> Tuple[str, str]:

    macs, params = get_model_complexity_info(
        model,
        input_res=input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params