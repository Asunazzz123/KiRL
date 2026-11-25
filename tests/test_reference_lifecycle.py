"""CPU checks for reference ownership and Trainer update boundaries."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("accelerate")
from kirl.algorithms.base import _BaseTrainer, RolloutWeightSyncCallback, ReferencePolicyCallback
from kirl.algorithms.grpo import GRPOTrainer
from kirl.distributed.ref_policy import create_reference_policy, prepare_reference_policy
from accelerate.utils import DistributedType
from kirl.profiler import NullProbe
from tests.test_fsdp_reference import _model, _tokenizer, _length_reward, RecordingBackend
from kirl.datasets import BaseDataset
from transformers import TrainingArguments


def test_reference_copy_is_independent_and_frozen():
    model = torch.nn.Linear(3, 2)
    ref = create_reference_policy(model)
    assert not ref.training
    for p, r in zip(model.parameters(), ref.parameters()):
        assert torch.equal(p, r) and p.data_ptr() != r.data_ptr()
        assert not r.requires_grad
    before = ref.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(ref.weight, before)


@pytest.mark.parametrize("model", [
    torch.nn.Linear(3, 2, device="meta"),
    torch.nn.DataParallel(torch.nn.Linear(3, 2)),
    torch.nn.LazyLinear(2),
])
def test_invalid_reference_sources(model):
    with pytest.raises(ValueError):
        create_reference_policy(model)


@pytest.mark.parametrize("fail", [False, True])
def test_prepare_preserves_policy_registration(fail):
    policy = torch.nn.Linear(3, 2)
    registry = [policy]
    def prepare(model, **kwargs):
        assert kwargs == {"evaluation_mode": True}
        registry.append(model)
        if fail:
            raise RuntimeError("prepare failed")
        return model
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO, _models=registry, prepare_model=prepare,
    )
    ref = create_reference_policy(policy)
    if fail:
        with pytest.raises(RuntimeError, match="prepare failed"):
            prepare_reference_policy(ref, accelerator)
    else:
        assert prepare_reference_policy(ref, accelerator) is ref
    assert len(registry) == 1 and registry[0] is policy


def test_weight_loader_failure_closes_parameter_scope():
    exited = []
    def params():
        try:
            yield "weight", torch.ones(1)
        finally:
            exited.append(True)
    def load(iterator):
        next(iterator)
        raise RuntimeError("load failed")
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(optimizer_step_was_skipped=False),
        _probe=NullProbe(), _iter_policy_params=params,
        _generation_backend=SimpleNamespace(update_weight=load),
    )
    trainer._policy_params_for_rollout = lambda: _BaseTrainer._policy_params_for_rollout(trainer)
    with pytest.raises(RuntimeError, match="load failed"):
        RolloutWeightSyncCallback(trainer).on_step_end(None, None, object())
    assert exited == [True]


def test_skipped_update_does_not_export():
    trainer = SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=True))
    control = object()
    assert RolloutWeightSyncCallback(trainer).on_step_end(None, None, control) is control


def test_rollout_export_reuses_binding_and_reads_current_parameters():
    unwrapped = []

    def unwrap(model, *, keep_torch_compile):
        assert not keep_torch_compile
        unwrapped.append(model)
        return model

    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=unwrap),
        is_deepspeed_enabled=False,
        is_fsdp_enabled=False,
        model_wrapped=model,
        _rollout_policy=None,
    )
    trainer._bind_rollout_policy = lambda: _BaseTrainer._bind_rollout_policy(trainer)
    RolloutWeightSyncCallback(trainer).on_train_begin(None, None, None)
    assert torch.equal(dict(_BaseTrainer._iter_policy_params(trainer))["weight"], model.weight)
    model.weight = torch.nn.Parameter(torch.ones_like(model.weight))
    assert torch.equal(dict(_BaseTrainer._iter_policy_params(trainer))["weight"], model.weight)
    assert unwrapped == [model]

    replacement = torch.nn.Linear(1, 1)
    trainer.model_wrapped = replacement
    assert torch.equal(dict(_BaseTrainer._iter_policy_params(trainer))["weight"], replacement.weight)
    assert unwrapped == [model, replacement]


def test_unprepared_reference_fails_before_rollout():
    trainer = SimpleNamespace(
        algo_config={"beta": 0.1}, reference_model=None, _reference_prepared=False,
    )
    with pytest.raises(RuntimeError, match="Reference policy must be created and prepared"):
        GRPOTrainer.compute_loss(trainer, None, {})


def test_reference_preparation_is_idempotent(monkeypatch):
    from kirl.algorithms import base

    raw, prepared = object(), object()
    calls = []
    def prepare(ref, accelerator):
        calls.append((ref, accelerator))
        return prepared
    monkeypatch.setattr(base, "prepare_reference_policy", prepare)
    trainer = SimpleNamespace(reference_model=raw, _reference_prepared=False, accelerator=object())
    _BaseTrainer._prepare_reference_policy(trainer)
    _BaseTrainer._prepare_reference_policy(trainer)
    assert len(calls) == 1 and calls[0][0] is raw
    assert trainer.reference_model is prepared and trainer._reference_prepared


@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_train_accumulation_syncs_updated_values_once(tmp_path, beta):
    class Backend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.weights = []
        def update_weight(self, params):
            self.weights.append({n: p.detach().clone() for n, p in params})
    backend = Backend()
    trainer = GRPOTrainer(
        _model(), reward_fn=_length_reward,
        train_dataset=BaseDataset([{"prompt": "w1 w2", "solution": "w3"}] * 4),
        processing_class=_tokenizer(), generation_backend=backend,
        algo_config={"beta": beta, "group_size": 2},
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, max_steps=1,
            gradient_accumulation_steps=2, per_device_train_batch_size=1,
            learning_rate=1e-2, remove_unused_columns=False,
            report_to="none", save_strategy="no", logging_strategy="no",
        ),
    )
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    assert (trainer.reference_model is None) == (beta == 0)
    assert not trainer._reference_prepared
    assert any(isinstance(c, ReferencePolicyCallback) for c in trainer.callback_handler.callbacks) == (beta > 0)
    trainer.train()
    assert len(backend.weights) == 1
    after = dict(trainer.model.named_parameters())
    assert any(not torch.equal(before[n], after[n]) for n in before)
    for name, value in backend.weights[0].items():
        assert torch.equal(value, after[name])
    if beta > 0:
        assert trainer._reference_prepared
        for name, value in trainer.reference_model.named_parameters():
            assert not value.requires_grad
            assert torch.equal(value, before[name])
