"""Dataset factory: maps DatasetSchema config dicts to BaseDataset instances."""

from typing import Any

from .base import BaseDataset
from .gsm8k import load_gsm8k

# Registry maps dataset.name → loader callable.
# Each loader receives the full dataset config dict and returns a BaseDataset.
# Add a new entry here (plus the implementation) to extend the system.
_LOADERS: dict[str, Any] = {
    "gsm8k": lambda cfg, split: load_gsm8k(
        cfg[f"{split}_path"],
        prompt_template_path=cfg.get("prompt_template_path"),
    ),
}


def build_dataset(
    cfg: dict[str, Any],
    *,
    split: str = "train",
) -> BaseDataset | None:
    """Instantiate a dataset from a resolved DatasetSchema config dict.

    Args:
        cfg: Fully-resolved dataset config dict (``resolved["dataset"]``
            from ``TrainConfig``).
        split: ``"train"`` or ``"eval"``. Returns ``None`` when the
            corresponding ``*_path`` key is ``None``.

    Returns:
        A ``BaseDataset`` instance, or ``None`` if the split is not configured.

    Raises:
        ValueError: If ``split`` is not ``"train"`` or ``"eval"``, or if
            ``cfg["name"]`` is not registered in ``_LOADERS``.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

    name = cfg.get("name")
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Registered names: {sorted(_LOADERS)}"
        )

    path_key = f"{split}_path"
    if cfg.get(path_key) is None:
        return None

    dataset = _LOADERS[name](cfg, split)

    # Apply cross-cutting post-load options from DatasetSchema.
    max_samples = cfg.get("max_samples")
    if max_samples is not None and len(dataset) > max_samples:
        import random
        rng = random.Random(cfg.get("seed", 42))
        indices = rng.sample(range(len(dataset)), max_samples)
        indices.sort()
        dataset = BaseDataset([dataset[i] for i in indices])

    return dataset
