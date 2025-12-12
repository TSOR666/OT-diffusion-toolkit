from typing import Any

from torch.utils.data import Dataset


class LSUN(Dataset[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class CelebA(Dataset[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class CIFAR10(Dataset[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ImageFolder(Dataset[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class FakeData(Dataset[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
