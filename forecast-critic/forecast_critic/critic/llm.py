from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from dataclasses import dataclass

import anthropic

from forecast_critic.config import CriticConfig

logger = logging.getLogger(__name__)


@dataclass
class CriticResponse:
    label: int              # 1 = reasonable, 2 = unreasonable
    explanation: str
    raw_response: str


def _parse_answer(text: str) -> int:
    match = re.search(r"<answer>\s*(\d)\s*</answer>", text)
    if match:
        return int(match.group(1))
    lower = text.lower()
    if "unreasonable" in lower:
        return 2
    if "reasonable" in lower:
        return 1
    return -1


def _parse_response(text: str) -> CriticResponse:
    label = _parse_answer(text)
    explanation = re.split(r"<answer>", text, maxsplit=1)[0].strip()
    return CriticResponse(label=label, explanation=explanation, raw_response=text)


class ForecastCritic:
    def __init__(self, config: CriticConfig | None = None):
        self.config = config or CriticConfig()
        self._client = anthropic.Anthropic()
        self._async_client = anthropic.AsyncAnthropic()

    def evaluate(self, image_bytes: bytes, prompt: str) -> CriticResponse:
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        message = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = message.content[0].text
        return _parse_response(text)

    async def _evaluate_async(self, image_bytes, prompt, semaphore, index):
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        for attempt in range(self.config.max_retries):
            try:
                async with semaphore:
                    message = await self._async_client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                    )
                text = message.content[0].text
                return index, _parse_response(text)
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                delay = self.config.retry_base_delay * (2 ** attempt)
                logger.warning("API error on sample %d (attempt %d/%d): %s. Retrying in %.1fs",
                    index, attempt + 1, self.config.max_retries, e, delay)
                await asyncio.sleep(delay)
        logger.error("Failed all retries for sample %d", index)
        return index, CriticResponse(label=-1, explanation="API failure", raw_response="")

    async def _evaluate_batch_async(self, items):
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = [self._evaluate_async(img, prompt, semaphore, i) for i, (img, prompt) in enumerate(items)]
        results_unordered = await asyncio.gather(*tasks)
        results_list = list(results_unordered)
        results_list.sort(key=lambda x: x[0])
        return [resp for _, resp in results_list]

    def evaluate_batch(self, items):
        logger.info("Starting batch evaluation: %d items, concurrency=%d", len(items), self.config.concurrency)
        t0 = time.monotonic()
        results = asyncio.run(self._evaluate_batch_async(items))
        elapsed = time.monotonic() - t0
        logger.info("Batch evaluation complete in %.1fs", elapsed)
        return results
