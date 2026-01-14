from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.modules.auth.models import User
from app.modules.auth.router import get_current_user
from app.modules.projects.models import Project
from app.modules.projects.schemas import (
    ProjectBase,
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectUpdateRequest,
    ProjectContextResponse,
)
from app.modules.projects.service import build_project_context, ensure_project_access


router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse)
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    q = db.query(Project)

    if not getattr(current_user, "is_admin", False):
        q = q.filter(
            (Project.assigned_employee_id == current_user.id)
            | (Project.created_by_id == current_user.id)
        )

    if status_filter:
        q = q.filter(Project.audit_status == status_filter)

    items = q.order_by(Project.updated_at.desc()).offset(offset).limit(limit).all()
    total = q.count()

    return {"items": items, "total": total}


@router.post("", response_model=ProjectBase, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = Project(
        project_code=payload.project_code,
        client_name=payload.client_name,
        fiscal_year_end=payload.fiscal_year_end,
        engagement_partner=payload.engagement_partner,
        audit_status=payload.audit_status,
        customer_id=payload.customer_id,
        assigned_employee_id=payload.assigned_employee_id
        if payload.assigned_employee_id
        else (None if getattr(current_user, "is_admin", False) else current_user.id),
        created_by_id=current_user.id,
    )

    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectBase)
def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ensure_project_access(db, project_id, current_user)
    return project


@router.put("/{project_id}", response_model=ProjectBase)
def update_project(
    project_id: UUID,
    payload: ProjectUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ensure_project_access(db, project_id, current_user)

    data = payload.dict(exclude_unset=True)
    for k, v in data.items():
        setattr(project, k, v)

    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}/context", response_model=ProjectContextResponse)
def get_project_context(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ensure_project_access(db, project_id, current_user)
    ctx = build_project_context(db, project)
    # ctx already JSON-like; ProjectContextResponse expects project as ProjectBase
    return {
        "project": project,
        "materiality": ctx.get("materiality"),
        "risks": ctx.get("risks", []),
        "legal_matters": ctx.get("legal_matters", []),
        "pbc": ctx.get("pbc"),
    }
