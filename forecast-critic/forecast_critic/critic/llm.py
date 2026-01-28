from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from forecast_critic.config import CriticConfig
from forecast_critic.llm_provider import call_vision, call_vision_async

logger = logging.getLogger(__name__)


@dataclass
class CriticResponse:
    label: int              # 1 = reasonable, 2 = unreasonable
    explanation: str
    raw_response: str


def _parse_answer(text: str) -> int:
    """Extract the answer label from <answer> tags."""
    match = re.search(r"<answer>\s*(\d)\s*</answer>", text)
    if match:
        return int(match.group(1))
    # Fallback: look for keywords
    lower = text.lower()
    if "unreasonable" in lower:
        return 2
    if "reasonable" in lower:
        return 1
    return -1  # unparseable


def _parse_response(text: str) -> CriticResponse:
    """Parse the full LLM response into structured output."""
    label = _parse_answer(text)
    # Explanation is everything before the <answer> tag
    explanation = re.split(r"<answer>", text, maxsplit=1)[0].strip()
    return CriticResponse(label=label, explanation=explanation, raw_response=text)


class ForecastCritic:
    """Multimodal LLM client for forecast evaluation. Supports Anthropic and Gemini."""

    def __init__(self, config: CriticConfig | None = None):
        self.config = config or CriticConfig()

    def evaluate(self, image_bytes: bytes, prompt: str) -> CriticResponse:
        """Send a single image + prompt to the LLM and parse the response."""
        text = call_vision(
            image_bytes,
            prompt,
            provider=self.config.provider,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return _parse_response(text)

    async def _evaluate_async(
        self,
        image_bytes: bytes,
        prompt: str,
        semaphore: asyncio.Semaphore,
        index: int,
    ) -> tuple[int, CriticResponse]:
        """Single async evaluation with rate limiting and retries."""
        for attempt in range(self.config.max_retries):
            try:
                async with semaphore:
                    text = await call_vision_async(
                        image_bytes,
                        prompt,
                        provider=self.config.provider,
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                return index, _parse_response(text)

            except Exception as e:
                delay = self.config.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "API error on sample %d (attempt %d/%d): %s. Retrying in %.1fs",
                    index, attempt + 1, self.config.max_retries, e, delay,
                )
                await asyncio.sleep(delay)

        # Final fallback: return unparseable response
        logger.error("Failed all retries for sample %d", index)
        return index, CriticResponse(label=-1, explanation="API failure", raw_response="")

    async def _evaluate_batch_async(
        self,
        items: list[tuple[bytes, str]],
    ) -> list[CriticResponse]:
        """Run batch evaluation with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.concurrency)

        tasks = [
            self._evaluate_async(img, prompt, semaphore, i)
            for i, (img, prompt) in enumerate(items)
        ]

        results_unordered = await asyncio.gather(*tasks)

        # Sort by index to maintain original order
        results_unordered_list = list(results_unordered)
        results_unordered_list.sort(key=lambda x: x[0])
        return [resp for _, resp in results_unordered_list]

    def evaluate_batch(
        self,
        items: list[tuple[bytes, str]],
    ) -> list[CriticResponse]:
        """Evaluate a batch of (image_bytes, prompt) pairs concurrently.

        Uses asyncio for parallel API calls with rate limiting.
        """
        logger.info(
            "Starting batch evaluation: %d items, concurrency=%d",
            len(items), self.config.concurrency,
        )
        t0 = time.monotonic()
        results = asyncio.run(self._evaluate_batch_async(items))
        elapsed = time.monotonic() - t0
        logger.info("Batch evaluation complete in %.1fs", elapsed)
        return results

