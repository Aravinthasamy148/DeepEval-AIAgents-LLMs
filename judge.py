"""Shared judge LLM for DeepEval metrics (Gemini), with quota retry."""
import os

from dotenv import load_dotenv
from deepeval.models import GeminiModel

from llm_config import JUDGE_MODEL, JUDGE_MODEL_POOL, with_quota_retry

load_dotenv()


class RetryGeminiModel(GeminiModel):
    """GeminiModel that retries transient quota and availability failures."""

    def generate(self, *args, **kwargs):
        def _call():
            return GeminiModel.generate(self, *args, **kwargs)

        return with_quota_retry(_call)

    async def a_generate(self, *args, **kwargs):
        # Prefer sync metrics (async_mode=False); still retry if async is used.
        import asyncio
        from llm_config import _retry_delay_seconds

        last_exc = None
        for attempt in range(6):
            try:
                return await GeminiModel.a_generate(self, *args, **kwargs)
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
                delay = _retry_delay_seconds(exc, 20.0 * (1.4 ** attempt))
                print(f"[quota] transient Gemini error — retry {attempt + 1}/6 in {delay:.0f}s…")
                await asyncio.sleep(delay)
        raise last_exc


def get_judge(model_name: str | None = None):
    """Return a Gemini judge. Pass model_name to pin a specific quota bucket."""
    return RetryGeminiModel(
        model=model_name or JUDGE_MODEL,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )


def get_judges(n: int = 3):
    """Return n judges from the pool (round-robin) for multi-metric tests."""
    return [get_judge(JUDGE_MODEL_POOL[i % len(JUDGE_MODEL_POOL)]) for i in range(n)]
