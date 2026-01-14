import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.db import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_code = Column(String(50), unique=True, nullable=False, index=True)
    client_name = Column(String(200), nullable=False)
    fiscal_year_end = Column(Date, nullable=False)

    engagement_partner = Column(String(100), nullable=True)
    audit_status = Column(String(50), nullable=False, default="planning")

    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=True)
    assigned_employee_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    customer = relationship("Customer")
    assigned_employee = relationship("User", foreign_keys=[assigned_employee_id])
    created_by = relationship("User", foreign_keys=[created_by_id])


class MaterialityRecord(Base):
    __tablename__ = "materiality"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    benchmark = Column(String(50), nullable=False)
    benchmark_value = Column(Numeric(20, 2), nullable=False)
    risk_level = Column(String(50), nullable=False)

    overall_materiality = Column(Numeric(20, 2), nullable=False)
    performance_materiality = Column(Numeric(20, 2), nullable=False)
    clearly_trivial_threshold = Column(Numeric(20, 2), nullable=False)

    rationale = Column(Text, nullable=True)

    confirmed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")
    confirmer = relationship("User")

    __table_args__ = (
        UniqueConstraint("project_id", name="uq_materiality_project"),
    )


class RiskRecord(Base):
    __tablename__ = "risk_register"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    cycle = Column(String(100), nullable=False)
    assertion = Column(String(50), nullable=True)
    risk_description = Column(Text, nullable=False)

    inherent_risk = Column(String(20), nullable=False)
    control_risk = Column(String(20), nullable=False)
    detection_risk = Column(String(20), nullable=True)

    response = Column(Text, nullable=True)

    is_significant = Column(Boolean, default=False, nullable=False)
    is_fraud_risk = Column(Boolean, default=False, nullable=False)

    confirmed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")
    confirmer = relationship("User")


class LegalMatterRecord(Base):
    __tablename__ = "legal_matrix"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    matter_name = Column(String(255), nullable=False)
    claim_amount = Column(Numeric(20, 2), nullable=True)

    probability = Column(String(20), nullable=False)
    outcome_estimable = Column(Boolean, default=True, nullable=False)

    is_material = Column(Boolean, default=False, nullable=False)
    disclosure_required = Column(Boolean, default=False, nullable=False)
    provision_required = Column(Boolean, default=False, nullable=False)
    is_kam = Column(Boolean, default=False, nullable=False)

    rationale = Column(Text, nullable=True)

    confirmed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")
    confirmer = relationship("User")


class PBCItem(Base):
    __tablename__ = "pbc_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    item_code = Column(String(50), nullable=False)
    item_name = Column(String(255), nullable=False)
    cycle = Column(String(100), nullable=True)

    priority = Column(String(20), nullable=False, default="medium")
    status = Column(String(50), nullable=False, default="pending")

    due_date = Column(Date, nullable=True)
    received_date = Column(Date, nullable=True)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")


class GeneratedDocument(Base):
    __tablename__ = "generated_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    template_id = Column(String(50), nullable=False)
    output_format = Column(String(10), nullable=False, default="docx")

    stored_file_id = Column(UUID(as_uuid=True), ForeignKey("stored_files.id"), nullable=True)

    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    mime_type = Column(String(200), nullable=True)
    size_bytes = Column(Integer, nullable=True)

    generated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")
    generator = relationship("User")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)

    intent = Column(String(50), nullable=True)
    tool_calls = Column(JSON, nullable=True)

    tokens_used = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project")
    user = relationship("User")
