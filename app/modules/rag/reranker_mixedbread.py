import asyncio
import logging
import inspect
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)

_mixedbread_import_error_logged = False


def _extract_status_code(exc: BaseException) -> Optional[int]:
    val = getattr(exc, "status_code", None)
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)

    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
        val = getattr(resp, "status", None)
        if isinstance(val, int):
            return val

    return None


def _is_retryable_error(exc: BaseException) -> bool:
    status = _extract_status_code(exc)
    if status in {429, 500, 502, 503, 504}:
        return True

    msg = str(exc).lower()
    if any(s in msg for s in ["429", "too many requests", "rate limit", "503", "service unavailable"]):
        return True

    return False


@dataclass
class MixedbreadRerankResult:
    index: int
    score: float


class MixedbreadReranker:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.MIXEDBREAD_API_KEY
        self.model = model or settings.MIXEDBREAD_RERANK_MODEL
        self.top_k = int(top_k or settings.MIXEDBREAD_RERANK_TOP_K)

        if not self.api_key:
            raise ValueError("MIXEDBREAD_API_KEY is not configured")

    async def rerank(self, *, query: str, documents: Sequence[str], top_k: Optional[int] = None) -> List[MixedbreadRerankResult]:
        """Return reranked indices for `documents`.

        Uses official mixedbread-ai python SDK.
        """
        if not documents:
            return []

        if not query or not query.strip():
            return [MixedbreadRerankResult(index=i, score=0.0) for i in range(min(len(documents), int(top_k or self.top_k)))]

        # Import inside method to avoid hard dependency at import time.
        try:
            from mixedbread_ai.client import AsyncMixedbreadAI  # type: ignore
        except Exception as e:  # pragma: no cover
            global _mixedbread_import_error_logged
            if not _mixedbread_import_error_logged:
                logger.warning("mixedbread-ai package is not installed")
                _mixedbread_import_error_logged = True
            k = int(top_k or self.top_k)
            return [
                MixedbreadRerankResult(index=i, score=0.0)
                for i in range(min(len(documents), k))
            ]

        client = AsyncMixedbreadAI(api_key=self.api_key)
        k = int(top_k or self.top_k)

        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "input": list(documents),
            "top_k": k,
            "return_input": False,
            "rewrite_query": False,
        }

        try:
            sig = inspect.signature(client.reranking)
            supported = set(sig.parameters.keys())
            kwargs_primary = {k: v for k, v in base_kwargs.items() if k in supported}
        except Exception:
            kwargs_primary = {k: v for k, v in base_kwargs.items() if k in {"model", "query", "input", "top_k"}}

        kwargs_fallback = {k: v for k, v in base_kwargs.items() if k in {"model", "query", "input", "top_k"}}

        last_exc: Exception | None = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                try:
                    resp = await client.reranking(**kwargs_primary)
                except TypeError:
                    kwargs_primary = kwargs_fallback
                    resp = await client.reranking(**kwargs_primary)
                break
            except Exception as e:  # pragma: no cover
                last_exc = e
                if not _is_retryable_error(e) or attempt >= max_attempts - 1:
                    raise
                delay_s = min(8.0, 0.5 * (2**attempt))
                logger.warning(
                    "Mixedbread rerank request failed; retrying",
                    extra={
                        "attempt": int(attempt + 1),
                        "max_attempts": int(max_attempts),
                        "delay_s": float(delay_s),
                        "status_code": _extract_status_code(e),
                    },
                )
                await asyncio.sleep(delay_s)

        if last_exc is not None and "resp" not in locals():
            raise last_exc

        data = getattr(resp, "data", None) or []
        results: List[MixedbreadRerankResult] = []
        for item in data:
            idx = int(getattr(item, "index", -1))
            score = float(getattr(item, "score", 0.0))
            if 0 <= idx < len(documents):
                results.append(MixedbreadRerankResult(index=idx, score=score))

        return results


_mixedbread_reranker_singleton: MixedbreadReranker | None = None
_mixedbread_lock = asyncio.Lock()


async def get_mixedbread_reranker() -> MixedbreadReranker:
    global _mixedbread_reranker_singleton
    if _mixedbread_reranker_singleton is not None:
        return _mixedbread_reranker_singleton

    async with _mixedbread_lock:
        if _mixedbread_reranker_singleton is None:
            _mixedbread_reranker_singleton = MixedbreadReranker()
        return _mixedbread_reranker_singleton
