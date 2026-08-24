"""Thin, paced, retrying wrapper around POST /api/lens/prompt.

Always requests the buffered (`stream: false`) response: the SDK-free way to
get a single JSON object back over urllib without an NDJSON parser.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

import config

load_dotenv()

_last_call_monotonic: float | None = None


def _wait_for_rate_limit_slot() -> None:
    global _last_call_monotonic
    now = time.monotonic()
    if _last_call_monotonic is not None:
        elapsed = now - _last_call_monotonic
        remaining = config.RATE_LIMIT_MIN_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)
    _last_call_monotonic = time.monotonic()


def _api_key() -> str:
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        raise RuntimeError("NEURONPEDIA_API_KEY is not set (expected in a .env file at the repo root)")
    return key


def call_lens_prompt(chat: list[dict[str, str]], num_completion_tokens: int) -> dict:
    """Call /api/lens/prompt with the locked Phase-0 config and return the buffered response.

    Paces itself against the shared per-process rate limiter and retries with
    exponential backoff on 429 (rate limited) and 5xx (server error).
    """
    payload = json.dumps(
        {
            "modelId": config.MODEL_ID,
            "chat": chat,
            "type": [config.LENS_TYPE],
            "topN": config.TOP_N_API,
            "temperature": config.TEMPERATURE,
            "numCompletionTokens": num_completion_tokens,
            "filterNonWordTokens": config.FILTER_NON_WORD_TOKENS,
            "stream": False,
        }
    ).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        _wait_for_rate_limit_slot()
        request = urllib.request.Request(
            config.API_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": _api_key(),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=config.REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 429 or error.code >= 500:
                last_error = error
                backoff = config.RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Lens request failed ({error.code}): {error.read().decode('utf-8', 'replace')}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # Network-level failure (stalled/reset connection, DNS hiccup, the
            # REQUEST_TIMEOUT_SECONDS timeout firing, ...) rather than an HTTP
            # response -- always worth retrying, unlike a deterministic 4xx.
            last_error = error
            backoff = config.RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
            time.sleep(backoff)
            continue

    raise RuntimeError(f"Lens request failed after {config.MAX_RETRIES} retries") from last_error
