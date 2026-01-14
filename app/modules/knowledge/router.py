from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.modules.auth.models import User
from app.modules.auth.router import get_current_user
from app.modules.rag.router import _get_rag_service
from app.modules.rag.service import RAGService
from app.modules.knowledge.schemas import KnowledgeSearchRequest, KnowledgeSearchResponse


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    payload: KnowledgeSearchRequest,
    service: RAGService = Depends(_get_rag_service),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Retrieval-only search against KB (ADMIN_LAW)
    result = await service.evidence(
        question=payload.query,
        customer_id=None,
        include_admin_laws=True,
        include_customer_docs=False,
        owner_id=None,
        top_k=payload.top_k,
    )

    chunks = []
    sources = []

    for item in result.get("context", []) or []:
        meta = {}
        if isinstance(item, dict):
            meta = {
                "kb_file_id": item.get("kb_file_id"),
                "block": item.get("block"),
                "section": item.get("section"),
                "isa_reference": item.get("isa_reference"),
                "ifrs_reference": item.get("ifrs_reference"),
                "cycle": item.get("cycle"),
            }
        src = item.get("filename") if isinstance(item, dict) else None
        if src and src not in sources:
            sources.append(src)

        chunks.append(
            {
                "content": (item.get("text") if isinstance(item, dict) else "") or "",
                "source": src,
                "score": float(item.get("score", 0.0)) if isinstance(item, dict) else None,
                "metadata": meta,
            }
        )

    return {"chunks": chunks, "sources": sources}
