"""
LLM-based Query Planner (NBLM-style)
====================================
Заменяет keyword-based _route_and_plan() на semantic understanding.

Использует Azure OpenAI (или Gemini fallback) для понимания смысла вопроса.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional

from app.modules.rag.types import IntentClass, QueryPlan
from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# ПРОМПТ ДЛЯ LLM ПЛАНИРОВЩИКА
# =============================================================================

QUERY_PLANNER_PROMPT = """Ты — планировщик запросов для аудиторской системы OSON.

Проанализируй вопрос пользователя и определи:
1. **intent** — тип намерения (из 21 варианта ниже)
2. **admin_budget** — сколько искать в методологии (0-8)
3. **customer_budget** — сколько искать в документах клиента (0-12)
4. **required_evidence** — нужны ли цитаты ("must_cite", "helpful", "optional")
5. **standards** — применимые стандарты ISA/IFRS

## Доступные intents (21):

**Договоры:**
- contract_signatories: стороны, кто подписал, руководители, заемщик, кредитор, в лице кого
- contract_structure: структура, разделы, пункты, приложения договора

**Документы:**
- doc_qa: общие вопросы по документам, файлам, актам

**Аудиторские процедуры:**
- planning_materiality: существенность, пороги, OM/PM/CT, качественные аспекты, benchmark
- sampling: выборка, размер выборки, ISA 530
- risk_assessment: оценка рисков, регистр рисков, ISA 315

**Юридические:**
- legal_subsequent_events: претензии, иски, суды, споры, арбитраж, IAS 37, IAS 10

**Непрерывность:**
- going_concern: непрерывность деятельности, банкротство, ликвидация, ISA 570, способность выжить

**Мнение:**
- opinion_forming: аудиторское мнение, модификация, оговорка, ISA 700

**KAM/TCWG:**
- kam: ключевые вопросы аудита, ISA 701
- tcwg_communications: коммуникации с руководством, ISA 260

**Прочие:**
- cycle_deep_dive: выручка, аренда, запасы, банки, реквизиты, IFRS 15/16
- acceptance_continuance: принятие клиента, независимость, ISQM
- pbc_waves: запросы документов, PBC list
- forensic_red_flags: мошенничество, fraud, ISA 240
- company_faq: контакты TRI-S-AUDIT, услуги компании
- industry_guidance: отраслевые риски, типичные контроли
- translation_terminology: перевод, глоссарий
- disclosure_drafting: раскрытие, disclosure, примечания
- model_ops_formatting: форматирование, шаблоны
- smalltalk: приветствия (ИЗБЕГАТЬ! Использовать только для "привет", "спасибо")

---

**Контекст диалога:**
{context}

**Вопрос пользователя:**
"{question}"

---

**ПРАВИЛА БЮДЖЕТОВ:**

Договоры: admin=0-2, customer=8-12
Методология (существенность, выборка, мнение): admin=6-8, customer=2-3
Юридика/риски: admin=6-8, customer=3-7
Документы клиента: admin=0, customer=10-12
Компания/отрасль: admin=4, customer=0

**ВАЖНО:** НЕ СТАВЬ customer=0 если вопрос про документы клиента!

---

Ответь СТРОГО в JSON формате:

```json
{{
  "intent": "going_concern",
  "admin_budget": 6,
  "customer_budget": 4,
  "required_evidence": "must_cite",
  "standards": ["ISA 570"],
  "reasoning": "Пользователь спрашивает о способности компании продолжать деятельность"
}}
```
"""


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

async def llm_query_plan(
    question: str,
    conversation_state: Dict[str, Any],
    llm_api,  # GeminiAPI or AzureOpenAIAPI
) -> QueryPlan:
    """
    LLM-based query planning (NBLM-style).

    Args:
        question: Вопрос пользователя
        conversation_state: Состояние диалога
        llm_api: API для LLM (Azure OpenAI или Gemini)

    Returns:
        QueryPlan совместимый с существующим pipeline
    """
    # Валидация входных данных
    if not question or not question.strip():
        logger.warning("Empty question provided, using fallback plan")
        return _fallback_plan("")

    if not llm_api:
        logger.error("LLM API not provided, using fallback plan")
        return _fallback_plan(question)

    conversation_state = conversation_state or {}

    try:
        # Собрать контекст
        context = _build_context(conversation_state)

        # Сформировать промпт
        prompt = QUERY_PLANNER_PROMPT.format(
            context=context,
            question=question
        )

        # Вызвать LLM (sync API -> async через asyncio.to_thread)
        response = await asyncio.to_thread(
            llm_api.generate_text,
            prompt,
            temperature=0.0,
            max_output_tokens=500,
        )

        # Парсить JSON
        plan_data = _parse_json_response(response)

        # Конвертировать в QueryPlan
        return _to_query_plan(plan_data, question)

    except Exception as e:
        logger.error(f"LLM query planning failed: {e}, using fallback")
        return _fallback_plan(question)


def _build_context(conversation_state: Dict[str, Any]) -> str:
    """Собрать контекст из conversation_state."""
    parts = []

    # Rolling summary
    if conversation_state.get("rolling_summary"):
        summary = conversation_state["rolling_summary"][:200]
        parts.append(f"Резюме диалога: {summary}")

    # Последние сообщения
    last_turns = conversation_state.get("last_turns", [])
    for turn in last_turns[-2:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")[:100]
        if content:
            parts.append(f"{role}: {content}")

    return "\n".join(parts) if parts else "(нет контекста)"


def _parse_json_response(response: str) -> Dict[str, Any]:
    """Извлечь JSON из ответа с поддержкой вложенных объектов."""
    # Попытка 1: JSON в ```json блоке
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Попытка 2: Stack-based parsing для вложенных объектов
    start = response.find('{')
    if start == -1:
        raise ValueError(f"No JSON found in response: {response[:200]}")

    stack = []
    end = start
    for i in range(start, len(response)):
        if response[i] == '{':
            stack.append(i)
        elif response[i] == '}':
            if stack:
                stack.pop()
                if not stack:
                    end = i + 1
                    break

    if stack:
        raise ValueError("Unbalanced braces in JSON response")

    json_str = response[start:end]
    return json.loads(json_str)


def _to_query_plan(data: Dict[str, Any], question: str) -> QueryPlan:
    """Конвертировать JSON в QueryPlan."""
    # Получить intent
    intent_str = data.get("intent", "doc_qa")
    try:
        intent = IntentClass(intent_str)
    except ValueError:
        logger.warning(f"Unknown intent: {intent_str}, defaulting to DOC_QA")
        intent = IntentClass.DOC_QA

    # Извлечь паттерны из вопроса (даты, стандарты, суммы)
    patterns = []
    patterns.extend(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', question))  # Даты
    patterns.extend(re.findall(r'\b[A-Z]{2,4}\s*\d{1,4}\b', question))  # ISA 320
    patterns.extend(re.findall(r'\b(?:USD|KZT|EUR|Т)\s*[\d,]+\.?\d*\b', question))  # Суммы

    return QueryPlan(
        intent=intent,
        required_evidence=data.get("required_evidence", "helpful"),
        admin_law_budget=data.get("admin_budget", 3),
        customer_doc_budget=data.get("customer_budget", 5),
        chat_memory_budget=data.get("chat_budget", 3),
        total_context_limit=data.get("total_limit", 8000),
        temperature=data.get("temperature", 0.3),
        exact_patterns=patterns,
        governing_standards=data.get("standards", []),
    )


def _fallback_plan(question: str) -> QueryPlan:
    """
    Fallback на DOC_QA если LLM недоступен.
    Безопасный default: искать везде с умеренными бюджетами.
    """
    patterns = []
    if question and question.strip():
        patterns.extend(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', question))
        patterns.extend(re.findall(r'\b[A-Z]{2,4}\s*\d{1,4}\b', question))

    logger.warning("Using fallback plan: DOC_QA with balanced budgets")

    return QueryPlan(
        intent=IntentClass.DOC_QA,
        required_evidence="helpful",
        admin_law_budget=5,
        customer_doc_budget=7,
        chat_memory_budget=3,
        total_context_limit=10000,
        temperature=0.3,
        exact_patterns=patterns,
        governing_standards=[],
    )


# =============================================================================
# ТЕСТЫ (для локальной проверки)
# =============================================================================

if __name__ == "__main__":
    import sys
    print("llm_query_planner.py: Module ready for import")
    print("Use: from app.modules.rag.llm_query_planner import llm_query_plan")
