from typing import Any, Sequence, Union

import torch


def gaussian_blur(
    img: torch.Tensor,
    kernel_size: Union[int, Sequence[int]],
    sigma: Any = ...,
) -> torch.Tensor: ...
