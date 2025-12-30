from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filter: Optional[dict[str, Any]] = None


class KnowledgeChunk(BaseModel):
    content: str
    source: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None


class KnowledgeSearchResponse(BaseModel):
    chunks: list[KnowledgeChunk]
    sources: list[str]
