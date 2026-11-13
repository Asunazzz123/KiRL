"""vLLM inference backend"""
import subprocess
import sys
import time
import torch

from collections.abc import Sequence, Iterable
from typing import Any
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .base import GenerationBackend, GenerationOutput, GenerationConfig
from .vllm_config import VLLMBackendConfig

from ...utils.logging import Log




class vLLMEngine(GenerationBackend):
    def __init__(self, engine: LLM) -> None:
        self._engine = engine
        self._client: Any = None          # openai.OpenAI, populated by start_client
        self._server_process: subprocess.Popen | None = None
        self._server_url: str | None = None
        self.logger = Log.get_logger(__name__)

    @classmethod
    def from_config(cls, config: VLLMBackendConfig) -> "vLLMEngine":
        logger = Log.get_logger(__name__)
        logger.info(
            "Initializing vLLM engine: model=%s tp=%d dtype=%s gpu_mem=%.2f",
            config.model, config.tensor_parallel_size, config.dtype,
            config.gpu_memory_utilization,
            extra={"event": "vllm.init", "phase": "init"},
        )
        engine = LLM(
            model=config.model,
            tokenizer=config.tokenizer,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enforce_eager=config.enforce_eager,
            enable_sleep_mode=config.enable_sleep_mode,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
        )
        logger.info(
            "vLLM engine ready",
            extra={"event": "vllm.ready", "phase": "init"},
        )
        return cls(engine)
    


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

        self.logger.debug(
            "generate: prompts=%d n=%d max_tokens=%d",
            len(prompt_token_ids), config.num_return_sequences, config.max_new_tokens,
            extra={"event": "vllm.generate.start", "phase": "rollout",
                   "num_prompts": len(prompt_token_ids)},
        )
        start = time.monotonic()
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

        self.logger.debug(
            "generate: done completions=%d elapsed=%.3fs",
            len(results), time.monotonic() - start,
            extra={"event": "vllm.generate.done", "phase": "rollout",
                   "num_completions": len(results)},
        )
        return results


    

    def update_weight(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]]
    ) -> None:
        """Sync updated policy weights into the vLLM engine in-place.

        Must be called after each gradient step so rollout uses the current
        policy. Only valid in embedded mode (``from_config``); has no effect
        on a server-mode deployment started via ``start_client``.

        Args:
            named_params: Iterator of ``(name, tensor)`` pairs, as returned
                by ``model.named_parameters()``.
        """
        self.logger.debug(
            "Syncing policy weights to vLLM engine",
            extra={"event": "vllm.weight_sync.start", "phase": "update"},
        )
        self._engine.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(named_params)
        self.logger.debug(
            "Weight sync complete",
            extra={"event": "vllm.weight_sync.done", "phase": "update"},
        )



    def client(self) -> Any:
        """Return the active OpenAI-compatible client handle.

        Raises:
            RuntimeError: If ``start_client`` has not been called yet.
        """
        if self._client is None:
            raise RuntimeError("No active client; call start_client() first")
        return self._client

    def start_client(
        self,
        config: VLLMBackendConfig,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        startup_timeout: float = 120.0,
    ) -> None:
        """Launch a vLLM OpenAI-compatible server and connect a client to it.

        The server runs as a child process using the same Python interpreter.
        Blocks until the server responds to health checks or ``startup_timeout``
        seconds elapse.

        Args:
            config: Backend config used to build the server launch command.
            host: Interface the server binds to.
            port: Port the server listens on.
            startup_timeout: Maximum seconds to wait for the server to become ready.

        Raises:
            RuntimeError: If a server is already running, or if the server
                does not become healthy within ``startup_timeout`` seconds.
        """
        if self._server_process is not None:
            raise RuntimeError("A server process is already running; call close_client() first")

        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package is required for client mode: pip install openai") from exc

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--dtype", config.dtype,
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--seed", str(config.seed),
            "--host", host,
            "--port", str(port),
        ]
        if config.tokenizer is not None:
            cmd += ["--tokenizer", config.tokenizer]
        if config.max_model_len is not None:
            cmd += ["--max-model-len", str(config.max_model_len)]
        if config.max_num_seqs is not None:
            cmd += ["--max-num-seqs", str(config.max_num_seqs)]
        if config.max_num_batched_tokens is not None:
            cmd += ["--max-num-batched-tokens", str(config.max_num_batched_tokens)]
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if config.enforce_eager:
            cmd.append("--enforce-eager")
        if config.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")

        self._server_process = subprocess.Popen(cmd)
        base_url = f"http://{host}:{port}/v1"
        self.logger.info(
            "vLLM server process started pid=%d url=%s",
            self._server_process.pid, base_url,
            extra={"event": "vllm.server.start", "phase": "init"},
        )

        # Poll the health endpoint until the server is ready.
        import urllib.request
        import urllib.error
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                if self._server_process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited early with code {self._server_process.returncode}"
                    )
                time.sleep(2.0)
        else:
            self._server_process.terminate()
            self._server_process = None
            raise RuntimeError(
                f"vLLM server did not become healthy within {startup_timeout}s"
            )

        self._server_url = base_url
        self._client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
        self.logger.info(
            "vLLM server ready url=%s",
            base_url,
            extra={"event": "vllm.server.ready", "phase": "init"},
        )

    def close_client(self) -> None:
        """Terminate the server process and release the client handle.

        Safe to call even if no server is running.
        """
        self._client = None
        self._server_url = None
        if self._server_process is not None:
            pid = self._server_process.pid
            self.logger.info(
                "Terminating vLLM server pid=%d",
                pid,
                extra={"event": "vllm.server.stop", "phase": "shutdown"},
            )
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "vLLM server pid=%d did not exit cleanly; sending SIGKILL",
                    pid,
                    extra={"event": "vllm.server.kill", "phase": "shutdown"},
                )
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None
            self.logger.info(
                "vLLM server pid=%d stopped",
                pid,
                extra={"event": "vllm.server.stopped", "phase": "shutdown"},
            )
