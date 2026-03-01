import asyncio
import logging
import time

from fireworks.training.sdk import DeploymentSampler

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_MAX_SAMPLE_ATTEMPTS = 5
_TRANSIENT_ERROR_CODES = ("502", "503", "425", "Connection")


class FiretitanEngine(RolloutEngine):
    """
    RolloutEngine implementation using Fireworks DeploymentSampler for model inference.

    Uses client-side tokenization via HuggingFace tokenizer and chat template,
    then sends token IDs directly to the Fireworks deployment completions endpoint.
    """

    def __init__(
        self,
        inference_url: str,
        model: str,
        api_key: str,
        tokenizer,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        processor=None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        """
        Initialize FiretitanEngine.

        Args:
            inference_url: Base URL of the Fireworks inference server
                (e.g. ``"https://api.fireworks.ai"``).
            model: Fully qualified model or deployment name
                (e.g. ``"accounts/acme/deployments/my-deploy"``).
            api_key: Fireworks API key.
            tokenizer: HuggingFace tokenizer used for chat-template rendering
                and client-side tokenization.
            max_prompt_length: Maximum prompt length in tokens.
            max_response_length: Maximum response length in tokens.
            max_model_length: Maximum total length (prompt + response) in tokens.
            sampling_params: Default sampling parameters (temperature, top_p, etc.).
            processor: Optional processor for multimodal models.
            disable_thinking: Whether to disable thinking in the generation prompt.
            accumulate_reasoning: Whether to accumulate reasoning across turns.
            reasoning_effort: Reasoning effort hint forwarded to the chat parser.
        """
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
        self.default_sampling_params = sampling_params or {}
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        self.sampler = DeploymentSampler(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            tokenizer=tokenizer,
        )

        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=disable_thinking,
        )

    def _prepare_max_tokens(self, requested_max_tokens: int, prompt_length: int) -> int:
        """
        Prepare max_tokens parameter, adjusting for max_model_length if needed.

        Args:
            requested_max_tokens: The requested max_tokens value
            prompt_length: The length of the prompt in tokens

        Returns:
            Adjusted max_tokens value
        """
        max_tokens = requested_max_tokens

        if self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining
                print(f"Warning: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return max_tokens

    def _sample_with_retry(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Call DeploymentSampler.sample_with_tokens with transient-error retries.

        Returns:
            List[SampledCompletion] (length 1).
        """
        for attempt in range(_MAX_SAMPLE_ATTEMPTS):
            try:
                return self.sampler.sample_with_tokens(
                    messages=messages,
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=1,
                )
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

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """
        Generate model response for a given set of messages.

        Args:
            messages: List of message dictionaries (OpenAI format)
            **kwargs: Additional parameters including:
                - application_id: Session/application ID (ignored)
                - validate: Whether this is validation (ignored)
                - enforce_max_prompt_length: Whether to enforce max prompt length
                - max_tokens / max_new_tokens: Completion budget
                - temperature: Sampling temperature
                - top_p: Nucleus sampling probability

        Returns:
            ModelOutput with generated text and metadata
        """
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        kwargs.pop("tools", None)
        kwargs.pop("accumulate_reasoning", None)
        kwargs.pop("reasoning_effort", None)

        requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_response_length))
        temperature = kwargs.get("temperature", self.default_sampling_params.get("temperature", 1.0))
        top_p = kwargs.get("top_p", self.default_sampling_params.get("top_p", 1.0))

        completions = await asyncio.to_thread(
            self._sample_with_retry,
            messages,
            requested_max_tokens,
            temperature,
            top_p,
        )

        if not completions:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampled = completions[0]

        prompt_ids: list[int] = list(sampled.full_tokens[: sampled.prompt_len])
        completion_ids: list[int] = list(sampled.full_tokens[sampled.prompt_len :])
        prompt_length = sampled.prompt_len

        if enforce_max_prompt_length and (
            prompt_length > self.max_prompt_length or prompt_length > self.max_model_length
        ):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        max_tokens = self._prepare_max_tokens(requested_max_tokens, prompt_length)
        if len(completion_ids) > max_tokens:
            completion_ids = completion_ids[:max_tokens]

        logprobs: list[float] | None = (
            list(sampled.inference_logprobs) if sampled.inference_logprobs else None
        )
        completion_text = sampled.text
        finish_reason = sampled.finish_reason or "stop"

        parsed_output = self.chat_parser.parse_completion(completion_ids)
        content = parsed_output.get("content", completion_text)
        reasoning = parsed_output.get("reasoning", "")
        tool_calls = parsed_output.get("tool_calls", [])

        return ModelOutput(
            text=completion_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )
