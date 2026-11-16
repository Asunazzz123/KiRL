"""Container contracts and DataLoader interoperability."""

import pytest
from torch.utils.data import DataLoader

from kirl.datasets import BaseDataset


def test_samples_preserve_task_fields_and_order():
    samples = [
        {"sample_id": "a", "prompt": "1+1?", "solution": "2"},
        {"sample_id": "b", "prompt": "2+2?", "solution": "4"},
    ]
    dataset = BaseDataset(samples)
    assert len(dataset) == 2
    assert dataset[0] == samples[0]
    assert dataset[-1] == samples[1]
    assert list(dataset) == samples
    with pytest.raises(IndexError):
        dataset[2]
    with pytest.raises(TypeError):
        dataset[:1]


def test_outer_mutation_does_not_change_stored_samples():
    messages = [{"role": "user", "content": "1+1?"}]
    samples = [{"prompt": messages, "solution": "2"}]
    dataset = BaseDataset(samples)
    samples[0]["solution"] = "wrong"
    samples.clear()
    item = dataset[0]
    item.pop("solution")
    assert len(dataset) == 1
    assert dataset[0]["solution"] == "2"
    # The documented contract deliberately avoids deep copies of nested values.
    assert dataset[0]["prompt"] is messages


def test_empty_dataset():
    dataset = BaseDataset([])
    assert len(dataset) == 0
    assert list(dataset) == []
    with pytest.raises(IndexError):
        dataset[0]


@pytest.mark.parametrize("samples", ["text", {"prompt": "x"}, ["text"], [{1: "x"}]])
def test_invalid_sample_structure(samples):
    with pytest.raises(TypeError):
        BaseDataset(samples)


def test_dataloader_can_batch_without_losing_reward_metadata():
    dataset = BaseDataset([
        {"prompt": "1+1?", "solution": "2", "sample_id": "a"},
        {"prompt": "2+2?", "solution": "4", "sample_id": "b"},
        {"prompt": "3+3?", "solution": "6", "sample_id": "c"},
    ])
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    assert batches[0]["solution"] == ["2", "4"]
    assert batches[1]["sample_id"] == ["c"]
