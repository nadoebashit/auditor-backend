import asyncio
import logging
import inspect
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


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
            raise ImportError("mixedbread-ai package is not installed") from e

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
            kwargs = {k: v for k, v in base_kwargs.items() if k in supported}
            resp = await client.reranking(**kwargs)
        except TypeError:
            kwargs = {k: v for k, v in base_kwargs.items() if k in {"model", "query", "input", "top_k"}}
            resp = await client.reranking(**kwargs)

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
