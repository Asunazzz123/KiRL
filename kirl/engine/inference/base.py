"""Backend-independent interface for batched rollout generation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationConfig:
    """Backend-agnostic sampling parameters.

    Backends translate these into their native APIs. Sentinel defaults
    (e.g. top_k=-1 meaning 'disabled') are interpreted per backend.
    """

    num_return_sequences: int = 1
    max_new_tokens: int = 256

    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled; backends map this to their own sentinel

    stop: tuple[str, ...] = field(default_factory=tuple)
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    ignore_eos: bool = False

    seed: int | None = None
    return_logprobs: bool = False


@dataclass(frozen=True)
class GenerationOutput:
    """
    One completion, associated with a prompt in the input batch.
    """

    prompt_index: int
    completion_token_ids: tuple[int, ...]
    text: str | None = None
    finish_reason: str | None = None
    log_probs: tuple[float, ...] | None = None


class GenerationBackend(ABC):
    """Generate rollouts without exposing inference-library result types."""

    @abstractmethod
    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        """
        Generate completions from unpadded prompt token IDs, without gradients.
        """

        raise NotImplementedError
