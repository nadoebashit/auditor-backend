from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.modules.auth.models import User
from app.modules.customers.models import Customer
from app.modules.projects.models import (
    LegalMatterRecord,
    MaterialityRecord,
    PBCItem,
    Project,
    RiskRecord,
)


def ensure_project_access(db: Session, project_id: UUID, user: User) -> Project:
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    if getattr(user, "is_admin", False):
        return project

    if project.assigned_employee_id and project.assigned_employee_id == user.id:
        return project

    if project.customer_id:
        customer = db.query(Customer).get(project.customer_id)
        if customer and customer.assigned_employee_id == user.id:
            return project

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def build_project_context(db: Session, project: Project) -> dict:
    materiality = (
        db.query(MaterialityRecord)
        .filter(MaterialityRecord.project_id == project.id)
        .order_by(MaterialityRecord.created_at.desc())
        .first()
    )

    risks = (
        db.query(RiskRecord)
        .filter(RiskRecord.project_id == project.id)
        .order_by(RiskRecord.created_at.desc())
        .all()
    )

    legal_matters = (
        db.query(LegalMatterRecord)
        .filter(LegalMatterRecord.project_id == project.id)
        .order_by(LegalMatterRecord.created_at.desc())
        .all()
    )

    pbc_items = (
        db.query(PBCItem)
        .filter(PBCItem.project_id == project.id)
        .order_by(PBCItem.created_at.desc())
        .all()
    )

    pbc_stats = {
        "total": len(pbc_items),
        "pending": len([x for x in pbc_items if (x.status or "") == "pending"]),
        "requested": len([x for x in pbc_items if (x.status or "") == "requested"]),
        "received": len([x for x in pbc_items if (x.status or "") == "received"]),
        "reviewed": len([x for x in pbc_items if (x.status or "") == "reviewed"]),
        "issue": len([x for x in pbc_items if (x.status or "") == "issue"]),
    }

    return {
        "project": {
            "id": str(project.id),
            "project_code": project.project_code,
            "client_name": project.client_name,
            "fiscal_year_end": project.fiscal_year_end.isoformat() if project.fiscal_year_end else None,
            "engagement_partner": project.engagement_partner,
            "audit_status": project.audit_status,
            "customer_id": str(project.customer_id) if project.customer_id else None,
            "assigned_employee_id": str(project.assigned_employee_id) if project.assigned_employee_id else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None,
        },
        "materiality": (
            {
                "benchmark": materiality.benchmark,
                "benchmark_value": float(materiality.benchmark_value),
                "risk_level": materiality.risk_level,
                "om": float(materiality.overall_materiality),
                "pm": float(materiality.performance_materiality),
                "ct": float(materiality.clearly_trivial_threshold),
                "rationale": materiality.rationale,
                "confirmed_by": str(materiality.confirmed_by) if materiality.confirmed_by else None,
                "confirmed_at": materiality.confirmed_at.isoformat() if materiality.confirmed_at else None,
            }
            if materiality
            else None
        ),
        "risks": [
            {
                "id": str(r.id),
                "cycle": r.cycle,
                "assertion": r.assertion,
                "risk_description": r.risk_description,
                "inherent_risk": r.inherent_risk,
                "control_risk": r.control_risk,
                "detection_risk": r.detection_risk,
                "response": r.response,
                "is_significant": bool(r.is_significant),
                "is_fraud_risk": bool(r.is_fraud_risk),
                "confirmed_by": str(r.confirmed_by) if r.confirmed_by else None,
                "confirmed_at": r.confirmed_at.isoformat() if r.confirmed_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in risks
        ],
        "legal_matters": [
            {
                "id": str(lm.id),
                "matter_name": lm.matter_name,
                "claim_amount": float(lm.claim_amount) if lm.claim_amount is not None else None,
                "probability": lm.probability,
                "outcome_estimable": bool(lm.outcome_estimable),
                "is_material": bool(lm.is_material),
                "disclosure_required": bool(lm.disclosure_required),
                "provision_required": bool(lm.provision_required),
                "is_kam": bool(lm.is_kam),
                "rationale": lm.rationale,
                "confirmed_by": str(lm.confirmed_by) if lm.confirmed_by else None,
                "confirmed_at": lm.confirmed_at.isoformat() if lm.confirmed_at else None,
                "created_at": lm.created_at.isoformat() if lm.created_at else None,
            }
            for lm in legal_matters
        ],
        "pbc": {
            "items": [
                {
                    "id": str(i.id),
                    "item_code": i.item_code,
                    "item_name": i.item_name,
                    "cycle": i.cycle,
                    "priority": i.priority,
                    "status": i.status,
                    "due_date": i.due_date.isoformat() if i.due_date else None,
                    "received_date": i.received_date.isoformat() if i.received_date else None,
                    "notes": i.notes,
                    "created_at": i.created_at.isoformat() if i.created_at else None,
                }
                for i in pbc_items
            ],
            "stats": pbc_stats,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
