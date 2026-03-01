import asyncio
import logging
import time
from typing import Any

from typing_extensions import override

from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import Tokenizer
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_MAX_SAMPLE_ATTEMPTS = 5
_TRANSIENT_ERROR_CODES = ("502", "503", "425", "Connection")


class FireworksEngine(RolloutEngine):
    """RolloutEngine implementation using Fireworks DeploymentSampler for inference.

    Uses client-side tokenization (via HuggingFace tokenizer + chat template)
    and sends token IDs to the Fireworks deployment completions endpoint,
    mirroring the interface of TinkerEngine.

    Requires:
        ``fireworks.training.sdk.DeploymentSampler`` and a running Fireworks
        inference deployment.
    """

    def __init__(
        self,
        inference_url: str,
        model: str,
        api_key: str,
        tokenizer: Tokenizer,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        """
        Args:
            inference_url: Base URL of the Fireworks inference server
                (e.g. ``"https://api.fireworks.ai"``).
            model: Fully qualified model/deployment name
                (e.g. ``"accounts/acme/deployments/my-deploy"``).
            api_key: Fireworks API key.
            tokenizer: HuggingFace tokenizer used for chat-template rendering
                and client-side tokenization.
            max_prompt_length: Hard cap on prompt token length.
            max_response_length: Default max completion tokens when no
                ``max_tokens`` / ``max_new_tokens`` kwarg is supplied.
            max_model_length: Total (prompt + response) context window.
                Defaults to ``max_prompt_length + max_response_length - 1``.
            sampling_params: Dict with optional ``"train"`` and ``"val"``
                sub-dicts containing default sampling kwargs
                (``temperature``, ``top_p``, …).
            disable_thinking: Passed to ``ChatTemplateParser`` to suppress
                thinking tokens in the generation prompt.
            accumulate_reasoning: Whether to accumulate reasoning across turns.
            reasoning_effort: Reasoning effort hint forwarded to the parser.
        """
        from fireworks.training.sdk import DeploymentSampler

        self.inference_url = inference_url
        self.model = model
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = (
            max_model_length - 1
            if max_model_length is not None
            else max_prompt_length + max_response_length - 1
        )
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        self.train_sampling_params = (sampling_params or {}).get("train", {})
        self.val_sampling_params = (sampling_params or {}).get("val", {})

        self.sampler = DeploymentSampler(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            tokenizer=tokenizer,
        )

        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            disable_thinking=disable_thinking,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_max_tokens(self, requested: int, prompt_length: int) -> int:
        """Clamp *requested* max_tokens so the total sequence fits the context window."""
        if self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining < requested:
                logger.warning(
                    "Decreasing max_tokens from %d to %d to stay within max_model_length=%d",
                    requested,
                    remaining,
                    self.max_model_length,
                )
                return remaining
        return requested

    def _build_sampling_kwargs(self, **overrides) -> dict[str, Any]:
        """Merge default train/val params with per-call overrides."""
        base = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()
        for key in ("temperature", "top_p", "top_k"):
            if key in overrides:
                base[key] = overrides[key]
        return base

    def _sample_sync(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_kwargs: dict[str, Any],
        prompt_idx: int = 0,
    ):
        """Call ``DeploymentSampler.sample_with_tokens`` with transient-error retries."""
        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                results = self.sampler.sample_with_tokens(
                    messages=[],  # unused — we pass pre-tokenized ids below
                    n=1,
                    max_tokens=max_tokens,
                    **sampling_kwargs,
                )
                return results[0] if results else None
            except Exception as exc:
                err = str(exc)
                transient = any(code in err for code in _TRANSIENT_ERROR_CODES)
                if transient and attempt < _MAX_SAMPLE_ATTEMPTS - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %ds…",
                        attempt + 1,
                        _MAX_SAMPLE_ATTEMPTS,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("Sampling failed permanently after %d attempts: %s", attempt + 1, exc)
                raise

    def _sample_sync_from_ids(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        sampling_kwargs: dict[str, Any],
    ):
        """Call ``DeploymentSampler.completions`` directly with pre-tokenised IDs."""
        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                raw = self.sampler.completions(
                    prompt=prompt_ids,
                    n=1,
                    max_tokens=max_tokens,
                    raw_output=True,
                    logprobs=True,
                    top_logprobs=1,
                    **sampling_kwargs,
                )
                return raw
            except Exception as exc:
                err = str(exc)
                transient = any(code in err for code in _TRANSIENT_ERROR_CODES)
                if transient and attempt < _MAX_SAMPLE_ATTEMPTS - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %ds…",
                        attempt + 1,
                        _MAX_SAMPLE_ATTEMPTS,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("Sampling failed permanently after %d attempts: %s", attempt + 1, exc)
                raise

    # ------------------------------------------------------------------
    # RolloutEngine interface
    # ------------------------------------------------------------------

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    @override
    async def get_token_output_from_token_input(
        self, token_input: list[int], **kwargs
    ):
        """Generate from a pre-tokenized prompt (token-in / token-out path).

        Args:
            token_input: List of integer token IDs for the prompt.
            **kwargs: Optional overrides:
                - ``max_tokens`` / ``max_new_tokens``: completion budget.
                - ``temperature``, ``top_p``, ``top_k``: sampling params.
                - ``enforce_max_prompt_length`` (bool, default True).

        Returns:
            Raw completions API response dict (one choice, ``raw_output=True``).
        """
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        prompt_length = len(token_input)

        if enforce_max_prompt_length and prompt_length > min(self.max_prompt_length, self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        requested_max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_response_length))
        max_tokens = self._prepare_max_tokens(requested_max_tokens, prompt_length)
        sampling_kwargs = self._build_sampling_kwargs(**kwargs)

        raw = await asyncio.to_thread(
            self._sample_sync_from_ids,
            token_input,
            max_tokens,
            sampling_kwargs,
        )
        return raw

    @override
    def assemble_model_output(self, token_input: list[int], token_output: dict) -> ModelOutput:
        """Convert a raw completions response dict into a ``ModelOutput``.

        Args:
            token_input: The prompt token IDs (used to compute prompt_length).
            token_output: Raw JSON response from ``DeploymentSampler.completions``
                (with ``raw_output=True``).
        """
        choice = token_output["choices"][0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "unknown")

        raw = choice.get("raw_output") or {}
        completion_ids: list[int] = list(raw.get("completion_token_ids") or [])

        # Extract per-token logprobs if present
        logprobs: list[float] | None = None
        lp_data = choice.get("logprobs")
        if lp_data and isinstance(lp_data, dict):
            content = lp_data.get("content")
            if isinstance(content, list) and content:
                logprobs = [tok.get("logprob", 0.0) for tok in content]

        prompt_ids = list(token_input)

        return ModelOutput(
            text=text,
            content=text,
            reasoning=None,
            tool_calls=None,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Generate a response from chat messages (high-level path).

        Applies the chat template client-side, calls the deployment, and
        returns a ``ModelOutput``.

        Args:
            messages: Conversation in OpenAI message format.
            **kwargs: Forwarded to ``get_token_output_from_token_input``:
                ``tools``, ``max_tokens``/``max_new_tokens``,
                ``temperature``, ``top_p``, ``top_k``,
                ``enforce_max_prompt_length``, ``accumulate_reasoning``,
                ``reasoning_effort``.
        """
        kwargs.pop("application_id", None)

        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        prompt: str = self.chat_parser.parse(  # type: ignore[union-attr]
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            reasoning_effort=reasoning_effort,
            accumulate_reasoning=accumulate_reasoning,
        )
        prompt_ids: list[int] = self.tokenizer.encode(prompt, add_special_tokens=False)

        raw = await self.get_token_output_from_token_input(prompt_ids, **kwargs)
        return self.assemble_model_output(token_input=prompt_ids, token_output=raw)
