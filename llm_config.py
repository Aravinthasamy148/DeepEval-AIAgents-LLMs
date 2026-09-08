"""Shared Gemini model names — env-overridable, separate agent vs judge quotas."""
import os
import re
import time
from typing import Callable, TypeVar

from dotenv import load_dotenv

load_dotenv()

# Distinct free-tier quota buckets for agent vs judges.
AGENT_MODEL = os.getenv("GEMINI_AGENT_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.7-flash")

# Rotate judges across metrics so RPM quotas don't stack on one model.
JUDGE_MODEL_POOL = [
    os.getenv("GEMINI_JUDGE_MODEL_1", "gemini-3.7-flash"),
    os.getenv("GEMINI_JUDGE_MODEL_2", "gemini-3.8-flash"),
    os.getenv("GEMINI_JUDGE_MODEL_3", "gemini-3.5-flash"),
]

T = TypeVar("T")


def _retry_delay_seconds(exc: Exception, fallback: float) -> float:
    msg = str(exc)
    match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", msg)
    if match:
        return float(match.group(1)) + 1.0
    match = re.search(r"'retryDelay': '([0-9]+)s'", msg)
    if match:
        return float(match.group(1)) + 1.0
    return fallback


def with_quota_retry(fn: Callable[[], T], *, retries: int = 6, base_delay: float = 20.0) -> T:
    """Retry on Gemini 429 / 503 / RESOURCE_EXHAUSTED / UNAVAILABLE."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            transient = any(
                token in msg
                for token in (
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "503",
                    "UNAVAILABLE",
                    "high demand",
                    "try again later",
                )
            )
            if not transient:
                raise
            last_exc = exc
            delay = _retry_delay_seconds(exc, base_delay * (1.4 ** attempt))
            print(f"[quota] transient Gemini error — retry {attempt + 1}/{retries} in {delay:.0f}s…")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
