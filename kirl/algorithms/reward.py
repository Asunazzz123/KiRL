"""
Task-level mathematical correctness rewards, independent of policy losses.

Design reference: Hugging Face TRL's accuracy_reward (Apache-2.0):
https://github.com/huggingface/trl/blob/main/trl/rewards/accuracy_rewards.py
KiRL additionally validates inputs, disables string parse fallbacks, and keeps
signal-based timeouts enabled by requiring execution on a worker's main thread.
"""
from collections.abc import Callable
import threading


def _math_tools():
    """Optional dependency: importing KiRL must not require a math parser."""
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig
    except ImportError as exc:
        raise ImportError(
            "math_accuracy_reward requires the optional math dependencies. "
            "Install KiRL with the 'math' extra in the training environment "
            "(pip install -e '.[math]')."
        ) from exc
    return parse, verify, LatexExtractionConfig, NormalizationConfig


def _completion_text(completion, index: int) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"completions[{index}] must be text or a single assistant message")
    message = completion[0]
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise TypeError(f"completions[{index}] requires a string content field")
    if message.get("role", "assistant") != "assistant":
        raise ValueError(f"completions[{index}] must contain the assistant completion, not the prompt")
    return message["content"]


def math_accuracy_reward(
    completions: list[str] | list[list[dict[str, str]]],
    solution: list[str],
    log_extra: Callable[[str, list], None] | None = None,
    *,
    parsing_timeout: int = 10,
    verify_timeout: int = 5,
    **kwargs,
) -> list[float | None]:
    r"""Return one outcome score per completion, in input order.

    Equivalent math -> 1.0; wrong/unparseable prediction -> 0.0;
    unparseable reference -> None (unscorable, NOT a wrong answer).

    Reference values should be canonical final answers prepared by the dataset
    adapter. Predictions use anchored LaTeX extraction, prioritizing \boxed{}.
    A bare number is not guaranteed to satisfy this prediction format.
    The function checks final-answer equivalence, not reasoning correctness.

    The caller must repeat solutions to match expanded rollout groups. It must
    also mask None rewards before computing group statistics or a loss. This
    function neither computes advantages nor adds KL/format/length penalties.
    """
    if not isinstance(completions, list) or not isinstance(solution, list):
        raise TypeError("completions and solution must be lists")
    if len(completions) != len(solution):
        raise ValueError("completions and solution must have the same length")
    contents = [_completion_text(value, i) for i, value in enumerate(completions)]
    if any(not isinstance(value, str) for value in solution):
        raise TypeError("Every solution must be a string")
    if log_extra is not None and not callable(log_extra):
        raise TypeError("log_extra must be callable or None")
    for name, value in (("parsing_timeout", parsing_timeout), ("verify_timeout", verify_timeout)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not contents:
        return []
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "Math reward timeouts require the main thread of a worker process; "
            "use process-based workers instead of a thread pool."
        )

    parse, verify, extraction_type, normalization_type = _math_tools()
    prediction_extraction = extraction_type(
        normalization_config=normalization_type(units=True),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
    # A GRPO group repeats the same reference. Cache only within this call;
    # avoid an unbounded cross-batch cache of symbolic expressions.
    references = {}
    scores = []
    diagnostic_gold, diagnostic_prediction, diagnostic_status = [], [], []
    for content, reference in zip(contents, solution, strict=True):
        if reference not in references:
            references[reference] = parse(
                reference, fallback_mode="no_fallback", parsing_timeout=parsing_timeout,
            )
        gold = references[reference]
        predicted = []
        if not gold:
            score, status = None, "unparseable_reference"
        else:
            predicted = parse(
                content, extraction_config=[prediction_extraction],
                extraction_mode="first_match", fallback_mode="no_fallback",
                parsing_timeout=parsing_timeout,
            )
            if not predicted:
                score, status = 0.0, "unparseable_prediction"
            else:
                score = float(verify(gold, predicted, timeout_seconds=verify_timeout))
                status = "correct" if score else "incorrect"
        scores.append(score)
        if log_extra is not None:
            diagnostic_gold.append(str(gold) if gold else "[unparseable]")
            diagnostic_prediction.append(str(predicted) if predicted else "[unparseable_or_skipped]")
            diagnostic_status.append(status)
    if log_extra is not None:
        log_extra("solution", list(solution))
        log_extra("gold_parsed", diagnostic_gold)
        log_extra("answer_parsed", diagnostic_prediction)
        log_extra("reward_status", diagnostic_status)
    return scores
