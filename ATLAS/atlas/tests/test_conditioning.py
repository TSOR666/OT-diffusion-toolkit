import torch

from atlas.conditioning.clip_interface import CLIPConditioningInterface
from atlas.config.conditioning_config import ConditioningConfig


def _dummy_payload() -> dict[str, torch.Tensor | None]:
    return {
        "context": torch.zeros(1, 1),
        "mask": None,
        "pooled": torch.zeros(1, 1),
    }


def test_clip_conditioning_cache_eviction() -> None:
    config = ConditioningConfig(use_clip=False, cache_encodings=True, cache_max_entries=2)
    interface = CLIPConditioningInterface(config=config, device=torch.device("cpu"))

    interface._store_cache_entry(("a",), _dummy_payload())
    interface._store_cache_entry(("b",), _dummy_payload())
    interface._store_cache_entry(("c",), _dummy_payload())

    assert len(interface.cache) == 2
    assert ("a",) not in interface.cache


def test_clip_conditioning_cache_clear() -> None:
    config = ConditioningConfig(use_clip=False, cache_encodings=True, cache_max_entries=1)
    interface = CLIPConditioningInterface(config=config, device=torch.device("cpu"))
    interface._store_cache_entry(("a",), _dummy_payload())
    interface.clear_cache()
    assert len(interface.cache) == 0
