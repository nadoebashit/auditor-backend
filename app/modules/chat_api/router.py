from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.engine.response_formatter import ResponseFormatter
from app.modules.auth.models import User
from app.modules.auth.router import get_current_user
from app.modules.projects.models import ChatHistory, Project
from app.modules.projects.service import build_project_context, ensure_project_access
from app.modules.rag.router import _get_rag_service
from app.modules.rag.service import RAGService
from app.modules.chat_api.schemas import ChatHistoryResponse, ChatRequest


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=dict)
async def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(_get_rag_service),
):
    project: Project = ensure_project_access(db, payload.project_id, current_user)

    tenant_id = str(project.id)

    # Recent history for context
    history_rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.project_id == project.id)
        .order_by(ChatHistory.created_at.desc())
        .limit(6)
        .all()
    )

    chat_context = []
    for row in reversed(history_rows):
        chat_context.append({"role": "user", "content": row.user_message})
        chat_context.append({"role": "assistant", "content": row.ai_response})

    project_ctx = build_project_context(db, project)

    rag_result = await rag_service.query(
        question=payload.message,
        customer_id=tenant_id,
        include_admin_laws=True,
        include_customer_docs=True,
        owner_id=None if getattr(current_user, "is_admin", False) else str(current_user.id),
        mode="hybrid",
        top_k=8,
        temperature=0.3,
        chat_context=chat_context,
        tenant_id=tenant_id,
        user_id=str(current_user.id),
        project_context=project_ctx,
        project_id=str(project.id),
    )

    processing = rag_result.get("processing_metadata") or {}
    intent = processing.get("intent") if isinstance(processing, dict) else None

    # Persist chat history
    hist = ChatHistory(
        project_id=project.id,
        user_id=current_user.id,
        user_message=payload.message,
        ai_response=rag_result.get("answer") or "",
        intent=str(intent) if intent else None,
        tool_calls={
            "tool_outputs": rag_result.get("tool_outputs"),
            "processing_metadata": processing,
        },
        tokens_used=int(processing.get("total_tokens")) if isinstance(processing, dict) and processing.get("total_tokens") else None,
        response_time_ms=int(processing.get("total_time_ms")) if isinstance(processing, dict) and processing.get("total_time_ms") else None,
        created_at=datetime.utcnow(),
    )
    db.add(hist)
    db.commit()

    formatted = ResponseFormatter.format(
        answer_text=rag_result.get("answer") or "",
        processing_intent=str(intent) if intent else None,
        project_id=str(project.id),
        tool_outputs=rag_result.get("tool_outputs") if isinstance(rag_result.get("tool_outputs"), dict) else None,
        citations=rag_result.get("context") if isinstance(rag_result.get("context"), list) else None,
        file=None,
    )

    # Return docs-style response: response/intent/buttons/table/file/redirect
    data = formatted.model_dump()
    if "text" not in data:
        data["text"] = data.get("response")
    return data


@router.get("/{project_id}", response_model=ChatHistoryResponse)
def get_chat_history(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project: Project = ensure_project_access(db, project_id, current_user)

    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.project_id == project.id)
        .order_by(ChatHistory.created_at.asc())
        .limit(200)
        .all()
    )

    items = []
    for r in rows:
        items.append({"role": "user", "content": r.user_message})
        items.append({"role": "assistant", "content": r.ai_response})

    return {"project_id": project_id, "items": items}


@router.delete("/{project_id}")
def clear_chat_history(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project: Project = ensure_project_access(db, project_id, current_user)

    db.query(ChatHistory).filter(ChatHistory.project_id == project.id).delete(synchronize_session=False)
    db.commit()

    return {"success": True}
