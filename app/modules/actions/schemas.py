from __future__ import annotations

from datetime import date
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ActionResponse(BaseModel):
    success: bool
    message: str
    record_id: Optional[UUID] = None
    extra: Optional[dict[str, Any]] = None


class SaveMaterialityRequest(BaseModel):
    project_id: UUID
    benchmark: str
    benchmark_value: float
    om: float
    pm: float
    ct: float
    risk_level: str
    rationale: str | None = None


class AddRiskRequest(BaseModel):
    project_id: UUID
    cycle: str
    assertion: str | None = None
    risk_description: str
    inherent_risk: str
    control_risk: str
    detection_risk: str | None = None
    response: str | None = None
    is_significant: bool = False
    is_fraud_risk: bool = False


class AddLegalMatterRequest(BaseModel):
    project_id: UUID
    matter_name: str
    claim_amount: float | None = None
    probability: str
    outcome_estimable: bool = True
    is_material: bool = False
    disclosure_required: bool = False
    provision_required: bool = False
    is_kam: bool = False
    rationale: str | None = None


class UpdatePBCRequest(BaseModel):
    project_id: UUID
    item_code: str
    item_name: str | None = None
    cycle: str | None = None
    priority: str | None = None
    status: str | None = None
    due_date: date | None = None
    received_date: date | None = None
    notes: str | None = None


class GenerateDocumentRequest(BaseModel):
    project_id: UUID
    template_id: str = Field(..., description="E01/E12/...")
    data: dict[str, Any] = Field(default_factory=dict)
    output_format: str = Field(default="docx")
