"""vLLM inference backend"""
from collections.abc import Sequence

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .base import GenerationBackend, GenerationOutput
from .base import GenerationConfig

class vLLMEngine(GenerationBackend):
    def __init__(
        self,
        engine: LLM
    ):
        self._engine = engine

    @staticmethod
    def _to_sampling_params(config: GenerationConfig) -> SamplingParams:
        return SamplingParams(
            n=config.num_return_sequences,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 0.0,
            top_p=config.top_p if config.do_sample else 1.0,
            top_k=config.top_k if config.do_sample else -1,
            stop=list(config.stop) or None,
            stop_token_ids=list(config.stop_token_ids) or None,
            ignore_eos=config.ignore_eos,
            seed=config.seed,
            logprobs=0 if config.return_logprobs else None,
        )


    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        if prompt_token_ids is None:
            raise TypeError("prompt_token_ids must be a batch of token seq")

        if len(prompt_token_ids) == 0:
            return []

        prompts = [
            TokensPrompt(prompt_token_ids=list(token_ids))
            for token_ids in prompt_token_ids
        ]
        sampling_params = self._to_sampling_params(config)

        requests = self._engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        results = []
        for prompt_index, request in enumerate(requests):
            for completion in request.outputs:
                log_probs = None
                if config.return_logprobs:
                    token_logprobs = completion.logprobs
                    if (
                        token_logprobs is None
                        or len(token_logprobs) != len(completion.token_ids)
                    ):
                        raise RuntimeError(
                            "Missing or misaligned completion log probabilities"
                        )

                    log_probs = tuple(
                        step[token_id].logprob
                        for token_id, step in zip(
                            completion.token_ids, token_logprobs
                        )
                    )

                results.append(
                    GenerationOutput(
                        prompt_index=prompt_index,
                        completion_token_ids=tuple(completion.token_ids),
                        text=completion.text,
                        finish_reason=completion.finish_reason,
                        log_probs=log_probs,
                    )
                )

        return results





    def client(
        self,
    ):
        ...




    def start_client(
        self
    ):
        ...


    def close_client(
        self
    ):
        ...
