"""
Shared types for RAG modules.
Defines core intent classification and query planning structures.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class IntentClass(Enum):
    """Intent classes for query routing."""
    CONTRACT_SIGNATORIES = "contract_signatories"
    CONTRACT_STRUCTURE = "contract_structure"
    DOC_QA = "doc_qa"
    COMPANY_FAQ = "company_faq"
    INDUSTRY_GUIDANCE = "industry_guidance"
    PLANNING_MATERIALITY = "planning_materiality"
    SAMPLING = "sampling"
    RISK_ASSESSMENT = "risk_assessment"
    CYCLE_DEEP_DIVE = "cycle_deep_dive"
    LEGAL_SUBSEQUENT_EVENTS = "legal_subsequent_events"
    ACCEPTANCE_CONTINUANCE = "acceptance_continuance"
    OPINION_FORMING = "opinion_forming"
    GOING_CONCERN = "going_concern"
    KAM = "kam"
    TCWG_COMMUNICATIONS = "tcwg_communications"
    PBC_WAVES = "pbc_waves"
    FORENSIC_RED_FLAGS = "forensic_red_flags"
    TRANSLATION_TERMINOLOGY = "translation_terminology"
    DISCLOSURE_DRAFTING = "disclosure_drafting"
    MODEL_OPS_FORMATTING = "model_ops_formatting"
    SMALLTALK = "smalltalk"


@dataclass
class QueryPlan:
    """Query execution plan with budgets and requirements."""
    intent: IntentClass
    required_evidence: str  # "must_cite" / "helpful" / "optional"
    admin_law_budget: int
    customer_doc_budget: int
    chat_memory_budget: int
    total_context_limit: int
    temperature: float
    exact_patterns: List[str]
    governing_standards: List[str]
