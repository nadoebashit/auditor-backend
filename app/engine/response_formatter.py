from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel


class Button(BaseModel):
    id: str
    label: str
    action: str
    payload: Optional[dict[str, Any]] = None
    style: Literal["primary", "secondary", "outline", "danger"]


class TableData(BaseModel):
    title: str
    columns: list[str]
    rows: list[list[str]]
    footer: Optional[str] = None


class FileInfo(BaseModel):
    filename: str
    url: str
    size: str
    mime_type: str
    generated_at: Optional[str] = None


class RedirectOption(BaseModel):
    label: str
    action: str
    target: Optional[str] = None


class RedirectData(BaseModel):
    message: str
    options: list[RedirectOption]


class ChatResponse(BaseModel):
    intent: str
    response: str
    buttons: Optional[list[Button]] = None
    table: Optional[TableData] = None
    file: Optional[FileInfo] = None
    redirect: Optional[RedirectData] = None
    citations: Optional[list[dict[str, Any]]] = None


_INTENT_MAP: dict[str, str] = {
    "planning_materiality": "J1_MATERIALITY",
    "risk_assessment": "J2_RISK",
    "legal_subsequent_events": "J3_LEGAL",
    "sampling": "J4_SAMPLING",
    "pbc_waves": "J5_PBC",
    "tcwg_communications": "J6_TCWG",
    "kam": "J7_KAM",
    "model_ops_formatting": "J8_KNOWLEDGE",
    "acceptance_continuance": "J11_ACCEPTANCE",
    "going_concern": "J14_GOING_CONCERN",
    "opinion_forming": "J19_OPINION",
}


def _fmt_money(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    return f"{x:,.0f}".replace(",", " ")


class ResponseFormatter:
    @staticmethod
    def format(
        *,
        answer_text: str,
        processing_intent: str | None,
        project_id: str | None,
        tool_outputs: dict[str, Any] | None = None,
        file: FileInfo | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        intent = _INTENT_MAP.get(str(processing_intent or "").strip(), str(processing_intent or "UNKNOWN"))
        buttons: list[Button] | None = None
        table: TableData | None = None
        redirect: RedirectData | None = None

        tools = tool_outputs or {}

        if intent == "J1_MATERIALITY" and isinstance(tools.get("materiality"), dict):
            m = tools["materiality"]
            table = TableData(
                title="Расчёт существенности",
                columns=["Показатель", "Значение"],
                rows=[
                    ["OM", f"{_fmt_money(m.get('om'))}"],
                    ["PM", f"{_fmt_money(m.get('pm'))}"],
                    ["CT", f"{_fmt_money(m.get('ct'))}"],
                ],
                footer=f"Benchmark: {m.get('benchmark')} | Risk: {m.get('risk_level')}",
            )

            if project_id:
                buttons = [
                    Button(
                        id="save_materiality",
                        label="💾 Сохранить",
                        action="POST /api/v1/actions/save-materiality",
                        payload={
                            "project_id": project_id,
                            "benchmark": m.get("benchmark"),
                            "benchmark_value": m.get("benchmark_value"),
                            "om": m.get("om"),
                            "pm": m.get("pm"),
                            "ct": m.get("ct"),
                            "risk_level": m.get("risk_level"),
                            "rationale": m.get("rationale"),
                        },
                        style="primary",
                    ),
                    Button(
                        id="cancel",
                        label="❌ Отмена",
                        action="dismiss",
                        style="secondary",
                    ),
                ]

                redirect = RedirectData(
                    message="✅ После сохранения перейти к оценке рисков?",
                    options=[
                        RedirectOption(label="Да, перейти к рискам", action="navigate", target=f"/project/{project_id}/risks"),
                        RedirectOption(label="Нет, остаться здесь", action="dismiss"),
                    ],
                )

        if intent == "J3_LEGAL" and isinstance(tools.get("legal"), dict):
            lm = tools["legal"]
            table = TableData(
                title="Legal matter assessment",
                columns=["Поле", "Значение"],
                rows=[
                    ["Claim amount", _fmt_money(lm.get("claim_amount"))],
                    ["Probability", str(lm.get("probability"))],
                    ["Material (>= PM)", str(lm.get("is_material"))],
                    ["Disclosure required", str(lm.get("disclosure_required"))],
                    ["Provision required", str(lm.get("provision_required"))],
                    ["KAM candidate", str(lm.get("is_kam"))],
                ],
                footer=str(lm.get("fs_action") or ""),
            )

            if project_id:
                buttons = [
                    Button(
                        id="add_legal_matter",
                        label="💾 Сохранить",
                        action="POST /api/v1/actions/add-legal-matter",
                        payload={
                            "project_id": project_id,
                            "matter_name": "Legal matter",
                            "claim_amount": lm.get("claim_amount"),
                            "probability": lm.get("probability"),
                            "outcome_estimable": True,
                            "is_material": lm.get("is_material"),
                            "disclosure_required": lm.get("disclosure_required"),
                            "provision_required": lm.get("provision_required"),
                            "is_kam": lm.get("is_kam"),
                            "rationale": lm.get("rationale"),
                        },
                        style="primary",
                    ),
                    Button(
                        id="cancel",
                        label="❌ Отмена",
                        action="dismiss",
                        style="secondary",
                    ),
                ]

        if intent == "J4_SAMPLING" and isinstance(tools.get("sampling"), dict):
            s = tools["sampling"]
            table = TableData(
                title="Расчёт выборки",
                columns=["Показатель", "Значение"],
                rows=[
                    ["Method", str(s.get("method"))],
                    ["Sample size", str(s.get("sample_size"))],
                    ["Confidence", str(s.get("confidence_level"))],
                ],
                footer=str(s.get("rationale") or ""),
            )

        if file is not None:
            buttons = (buttons or []) + [
                Button(
                    id="download",
                    label="⬇ Скачать файл",
                    action="download",
                    payload={"url": file.url},
                    style="primary",
                )
            ]

        return ChatResponse(
            intent=intent,
            response=answer_text,
            buttons=buttons,
            table=table,
            file=file,
            redirect=redirect,
            citations=citations,
        )
