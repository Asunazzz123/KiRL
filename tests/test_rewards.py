"""Reward orchestration tests; real parser coverage is dependency-gated."""
import builtins
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from kirl.algorithms import reward
from kirl.pipeline.rewards import math_accuracy_reward


@pytest.fixture
def math_stub(monkeypatch):
    calls = []

    @dataclass
    class Normalization:
        units: bool

    @dataclass
    class Extraction:
        normalization_config: Normalization
        boxed_match_priority: int
        try_extract_without_anchor: bool

    def parse(text, **kwargs):
        calls.append(("parse", text, kwargs))
        return [] if text == "" else [text]

    def verify(gold, prediction, **kwargs):
        calls.append(("verify", gold, prediction, kwargs))
        return gold == prediction

    monkeypatch.setattr(reward, "_math_tools", lambda: (parse, verify, Extraction, Normalization))
    return calls


def test_binary_scores_skip_and_alignment(math_stub):
    scores = math_accuracy_reward(["2", "3", "", "2"], ["2", "2", "2", ""])
    assert scores == [1.0, 0.0, 0.0, None]
    verification = [c for c in math_stub if c[0] == "verify"]
    assert len(verification) == 2
    assert verification[0][1:3] == (["2"], ["2"])


def test_chat_format_and_batch_diagnostics(math_stub):
    extra = {}
    scores = math_accuracy_reward(
        [[{"role": "assistant", "content": "2"}], [{"content": "3"}]],
        ["2", ""], log_extra=lambda key, values: extra.update({key: values}),
        prompts=["ignored", "ignored"],
    )
    assert scores == [1.0, None]
    assert extra["reward_status"] == ["correct", "unparseable_reference"]
    assert all(len(values) == 2 for values in extra.values())


def test_reference_cache_and_extraction_contract(math_stub):
    math_accuracy_reward(["pred1", "pred2"], ["gold", "gold"], parsing_timeout=3, verify_timeout=2)
    gold_calls = [c for c in math_stub if c[0] == "parse" and c[1] == "gold"]
    assert len(gold_calls) == 1
    for call in (c for c in math_stub if c[0] == "parse"):
        assert call[2]["fallback_mode"] == "no_fallback"
        assert call[2]["parsing_timeout"] == 3
    prediction_call = next(c for c in math_stub if c[0] == "parse" and c[1] == "pred1")
    extraction = prediction_call[2]["extraction_config"][0]
    assert extraction.boxed_match_priority == 0
    assert extraction.try_extract_without_anchor is False
    assert extraction.normalization_config.units is True
    assert all(c[3]["timeout_seconds"] == 2 for c in math_stub if c[0] == "verify")


@pytest.mark.parametrize("completions,solutions", [
    (["2"], []),
    ([[]], ["2"]),
    ([[{"content": "2"}, {"content": "3"}]], ["2"]),
    ([[{"role": "user", "content": "2"}]], ["2"]),
    ([[{"content": 2}]], ["2"]),
    (["2"], [2]),
    ("2", ["2"]),
])
def test_reject_bad_contract_before_loading_dependency(monkeypatch, completions, solutions):
    monkeypatch.setattr(reward, "_math_tools", lambda: pytest.fail("should validate before imports"))
    with pytest.raises((TypeError, ValueError)):
        math_accuracy_reward(completions, solutions)


def test_empty_batch_needs_no_dependency(monkeypatch):
    monkeypatch.setattr(reward, "_math_tools", lambda: pytest.fail("empty input"))
    assert math_accuracy_reward([], []) == []


@pytest.mark.parametrize("kwargs", [{"parsing_timeout": 0}, {"verify_timeout": True}, {"log_extra": 3}])
def test_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        math_accuracy_reward([], [], **kwargs)


def test_missing_dependency_has_actionable_error(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "math_verify":
            raise ModuleNotFoundError("test: math_verify is unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="optional math dependencies"):
        math_accuracy_reward(["2"], ["2"])


def test_threads_do_not_silently_disable_timeout(math_stub):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(math_accuracy_reward, ["2"], ["2"])
        with pytest.raises(RuntimeError, match="process-based workers"):
            future.result()
    assert not math_stub


def test_unexpected_verifier_failure_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("verifier bug")
    monkeypatch.setattr(reward, "_math_tools", lambda: (
        lambda text, **kwargs: [text], fail,
        lambda **kwargs: object(), lambda **kwargs: object(),
    ))
    with pytest.raises(RuntimeError, match="verifier bug"):
        math_accuracy_reward(["2"], ["2"])


@pytest.mark.parametrize("prediction,solution,expected", [
    (r"The answer is \boxed{\frac{2}{4}}", "0.5", 1.0),
    (r"The answer is \boxed{4}", "5", 0.0),
    ("", "2", 0.0),
    (r"\boxed{2}", "", None),
])
def test_real_math_verify(prediction, solution, expected):
    pytest.importorskip("math_verify")
    pytest.importorskip("latex2sympy2_extended")
    assert math_accuracy_reward([prediction], [solution]) == [expected]
