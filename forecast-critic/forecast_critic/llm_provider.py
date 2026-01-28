"""Unified LLM provider abstraction.

Supports Anthropic (paid) and Google Gemini (free tier).
All vision + text calls throughout the codebase route through here.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os

logger = logging.getLogger(__name__)


# ── Anthropic ───────────────────────────────────────────────────────────


def _call_anthropic_vision(
    image_bytes: bytes,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import anthropic

    client = anthropic.Anthropic()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return message.content[0].text


def _call_anthropic_text(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


async def _call_anthropic_vision_async(
    image_bytes: bytes,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return message.content[0].text


# ── Gemini ──────────────────────────────────────────────────────────────


def _call_gemini_vision(
    image_bytes: bytes,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text


def _call_gemini_text(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text


async def _call_gemini_vision_async(
    image_bytes: bytes,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text



# ── Public API ──────────────────────────────────────────────────────────

_VISION_DISPATCH = {
    "anthropic": _call_anthropic_vision,
    "gemini": _call_gemini_vision,
}

_TEXT_DISPATCH = {
    "anthropic": _call_anthropic_text,
    "gemini": _call_gemini_text,
}

_VISION_ASYNC_DISPATCH = {
    "anthropic": _call_anthropic_vision_async,
    "gemini": _call_gemini_vision_async,
}


def call_vision(
    image_bytes: bytes,
    prompt: str,
    *,
    provider: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send image + prompt to LLM, return text response."""
    fn = _VISION_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_VISION_DISPATCH)}")
    return fn(image_bytes, prompt, model, max_tokens, temperature)


def call_text(
    prompt: str,
    *,
    provider: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send text-only prompt to LLM, return text response."""
    fn = _TEXT_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_TEXT_DISPATCH)}")
    return fn(prompt, model, max_tokens, temperature)


async def call_vision_async(
    image_bytes: bytes,
    prompt: str,
    *,
    provider: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Async: send image + prompt to LLM, return text response."""
    fn = _VISION_ASYNC_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_VISION_ASYNC_DISPATCH)}")
    return await fn(image_bytes, prompt, model, max_tokens, temperature)


# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
}


def get_default_model(provider: str) -> str:
    """Return the default model ID for a provider."""
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])

