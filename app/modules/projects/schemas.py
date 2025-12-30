from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreateRequest(BaseModel):
    project_code: str = Field(..., max_length=50)
    client_name: str = Field(..., max_length=200)
    fiscal_year_end: date

    engagement_partner: str | None = None
    audit_status: str = "planning"

    customer_id: UUID | None = None
    assigned_employee_id: UUID | None = None


class ProjectUpdateRequest(BaseModel):
    client_name: str | None = None
    fiscal_year_end: date | None = None
    engagement_partner: str | None = None
    audit_status: str | None = None

    customer_id: UUID | None = None
    assigned_employee_id: UUID | None = None


class ProjectBase(BaseModel):
    id: UUID
    project_code: str
    client_name: str
    fiscal_year_end: date
    engagement_partner: str | None
    audit_status: str
    customer_id: UUID | None
    assigned_employee_id: UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectListResponse(BaseModel):
    items: list[ProjectBase]
    total: int


class ProjectContextResponse(BaseModel):
    project: ProjectBase
    materiality: dict | None = None
    risks: list[dict] = Field(default_factory=list)
    legal_matters: list[dict] = Field(default_factory=list)
    pbc: dict | None = None


class PBCStatusResponse(BaseModel):
    items: list[dict]
    stats: dict
