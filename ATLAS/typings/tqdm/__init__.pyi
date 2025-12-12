from typing import Any, Iterable, TypeVar, overload

_T = TypeVar("_T")


@overload
def tqdm(iterable: Iterable[_T], *args: Any, **kwargs: Any) -> Iterable[_T]: ...


@overload
def tqdm(*args: Any, **kwargs: Any) -> Any: ...
