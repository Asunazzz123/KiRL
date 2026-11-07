"""Minimal in-memory, map-style dataset for normalized training samples."""

from collections.abc import Mapping, Sequence
from operator import index as integer_index
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Store sample dictionaries and expose the PyTorch Dataset interface.
    """

    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a sequence of sample mappings")

        self._samples: list[dict[str, Any]] = []
        for i, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise TypeError(f"samples[{i}] must be a mapping")
            if any(not isinstance(key, str) for key in sample):
                raise TypeError(f"samples[{i}] must have string keys")
            self._samples.append(dict(sample))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read one sample; negative indices work, out-of-range raises IndexError.

        Return a shallow copy so a collator can add/remove top-level fields
        without changing the stored sample. Slices are intentionally unsupported.
        """
        return dict(self._samples[integer_index(index)])
