import numpy as np
import torch


def set_seed(seed: int, strict_determinism: bool = True) -> int:
    """Set random seed for reproducibility across all RNGs.

    Args:
        seed: Seed value to apply across libraries.
        strict_determinism: If True, enables cuDNN deterministic mode which
            can reduce performance but guarantees reproducibility.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
