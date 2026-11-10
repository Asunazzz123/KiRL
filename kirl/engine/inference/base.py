"""Backend-independent interface for batched rollout generation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vllm_config import GenerationConfig


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
        config: "GenerationConfig",
    ) -> list[GenerationOutput]:
        """
        Generate completions from unpadded prompt token IDs, without gradients.
        """

        raise NotImplementedError
