from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    project_id: UUID
    message: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    project_id: UUID
    items: list[ChatHistoryItem]
