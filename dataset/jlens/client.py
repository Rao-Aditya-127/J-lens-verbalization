"""Thin, paced, retrying wrapper around POST /api/lens/prompt.

Always requests the buffered (`stream: false`) response: the SDK-free way to
get a single JSON object back over urllib without an NDJSON parser.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

import config

load_dotenv()

_last_call_monotonic: float | None = None
# The server reports remaining quota on every response. Pacing blind to it made
# the workers oscillate: burst -> saturate the window -> all sleep -> window
# refills unused -> burst again, averaging ~64 rows/hour against a 240/hour
# allowance. Track it and slow down BEFORE hitting the wall instead of
# recovering after.
_limit_remaining: int | None = None


def _adaptive_extra_wait() -> float:
    """Extra seconds to wait based on how much quota is left.

    Full headroom -> no extra delay. As the window fills, back off smoothly so
    the limit is approached rather than collided with. A 429 costs far more than
    the seconds spent avoiding it.
    """
    if _limit_remaining is None or _limit_remaining >= 60:
        return 0.0
    return (60 - _limit_remaining) * 2.0


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
    rate_limit_attempts = 0
    attempt = 0
    while attempt < config.MAX_RETRIES:
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
                global _limit_remaining
                header = response.headers.get("X-Limit-Remaining")
                if header is not None and header.isdigit():
                    _limit_remaining = int(header)
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 429:
                _limit_remaining = 0
                # Flat, long waits -- and they do not consume the general retry
                # budget, so a rate-limit pause cannot exhaust the retries that
                # exist for genuine transient failures.
                last_error = error
                rate_limit_attempts += 1
                if rate_limit_attempts > config.RATE_LIMIT_MAX_RETRIES:
                    raise RuntimeError(
                        f"rate limited after {rate_limit_attempts - 1} waits of "
                        f"{config.RATE_LIMIT_BACKOFF_SECONDS:.0f}s"
                    ) from error
                print(
                    f"  [rate limited] waiting {config.RATE_LIMIT_BACKOFF_SECONDS:.0f}s "
                    f"({rate_limit_attempts}/{config.RATE_LIMIT_MAX_RETRIES})",
                    flush=True,
                )
                # jitter: without it every worker waits the same time and they
                # resynchronise, hitting the limit together on every retry.
                time.sleep(config.RATE_LIMIT_BACKOFF_SECONDS * random.uniform(0.6, 1.6))
                continue
            if error.code >= 500:
                last_error = error
                backoff = config.RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
                attempt += 1
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Lens request failed ({error.code}): {error.read().decode('utf-8', 'replace')}") from error
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            # Network-level failure (stalled/reset connection, DNS hiccup, the
            # REQUEST_TIMEOUT_SECONDS timeout firing, a truncated body raising
            # http.client.IncompleteRead, ...) rather than an HTTP response --
            # always worth retrying, unlike a deterministic 4xx.
            #
            # IncompleteRead is an HTTPException, NOT an OSError, so it escaped
            # every clause above and killed a run mid-collection. Over thousands
            # of calls it recurs; this clause is what makes long runs survivable.
            last_error = error
            backoff = config.RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
            attempt += 1
            time.sleep(backoff)
            continue

    raise RuntimeError(f"Lens request failed after {config.MAX_RETRIES} retries") from last_error
