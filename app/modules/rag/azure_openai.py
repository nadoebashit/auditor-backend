from __future__ import annotations

import logging
from typing import Any, List, Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class AzureOpenAIAPI:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        responses_endpoint: Optional[str] = None,
        responses_model: Optional[str] = None,
        embeddings_endpoint: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ):
        self.api_key = api_key or settings.AZURE_OPENAI_API_KEY
        self.responses_endpoint = responses_endpoint or settings.AZURE_OPENAI_RESPONSES_ENDPOINT
        self.responses_model = responses_model or settings.AZURE_OPENAI_RESPONSES_MODEL
        self.embeddings_endpoint = embeddings_endpoint or settings.AZURE_OPENAI_EMBEDDINGS_ENDPOINT
        self.embeddings_model = embeddings_model or settings.AZURE_OPENAI_EMBEDDINGS_MODEL
        self.timeout_s = int(timeout_s or settings.AZURE_OPENAI_TIMEOUT_S)

        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not configured")
        if not self.responses_endpoint:
            raise ValueError("AZURE_OPENAI_RESPONSES_ENDPOINT is not configured")

        logger.info(
            "AzureOpenAIAPI initialized",
            extra={
                "has_embeddings": bool(self.embeddings_endpoint),
                "responses_model": self.responses_model,
                "embeddings_model": self.embeddings_model,
            },
        )

    def _verify_setting(self) -> Any:
        return settings.REQUESTS_CA_BUNDLE or settings.REQUESTS_VERIFY_SSL

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": str(self.api_key),
        }

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
    ) -> str:
        input_items: list[dict[str, object]] = []
        if system_instruction:
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": str(system_instruction)}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": str(prompt)}],
            }
        )

        input_string = str(prompt)
        if system_instruction:
            input_string = f"{system_instruction}\n\n{prompt}"

        endpoint_lower = str(self.responses_endpoint).lower()

        resolved_model = model or self.responses_model
        # Deployment-scoped endpoints usually don't require (and may reject) the "model" field.
        include_model = "/deployments/" not in endpoint_lower

        base_body: dict[str, object] = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
        }
        if include_model:
            base_body["model"] = resolved_model

        def _post(body: dict[str, object]) -> requests.Response:
            return requests.post(
                str(self.responses_endpoint),
                headers=self._headers(),
                json=body,
                timeout=self.timeout_s,
                verify=self._verify_setting(),
            )

        # Azure endpoints differ by route/version. Try progressively simpler payloads on HTTP 400.
        attempts: list[dict[str, object]] = [
            {**base_body, "input": input_items},
            {**base_body, "input": input_string},
        ]

        # Minimal payload (some previews reject optional fields).
        minimal_body: dict[str, object] = {"input": input_string}
        if include_model:
            minimal_body["model"] = resolved_model
        attempts.append(minimal_body)

        resp: requests.Response | None = None
        for body in attempts:
            resp = _post(body)
            if resp.status_code != 400:
                break

        if resp is None:
            raise RuntimeError("Azure responses request failed: no response")

        try:
            resp.raise_for_status()
        except Exception as e:
            error_payload: str | None = None
            try:
                if resp.content:
                    error_payload = str(resp.json())
            except Exception:
                error_payload = None
            logger.error(
                "Azure responses request failed: status=%s include_model=%s model=%s endpoint=%s body=%s error_payload=%s error=%s",
                getattr(resp, "status_code", None),
                bool(include_model),
                (resolved_model if include_model else None),
                str(self.responses_endpoint),
                (resp.text or "")[:500],
                ((error_payload or "")[:500] if error_payload else None),
                str(e),
            )
            raise

        data = resp.json() if resp.content else {}

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        # Defensive extraction from output[].content[].text
        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        parts.append(c["text"])

        joined = "".join(parts).strip()
        if joined:
            return joined

        # Fallbacks
        if isinstance(data.get("text"), str) and data.get("text").strip():
            return data.get("text")

        logger.warning(
            "Azure responses: could not extract output_text",
            extra={"keys": list(data.keys())[:30]},
        )
        return ""

    def embed_documents(self, texts: List[str], *, model: Optional[str] = None) -> List[List[float]]:
        if not self.embeddings_endpoint:
            raise ValueError("AZURE_OPENAI_EMBEDDINGS_ENDPOINT is not configured")
        if not texts:
            return []

        endpoint_lower = str(self.embeddings_endpoint).lower()
        body: dict[str, object] = {
            "input": [t[:32000] if len(t) > 32000 else t for t in texts],
        }
        if "/deployments/" not in endpoint_lower:
            body["model"] = model or self.embeddings_model

        resp = requests.post(
            str(self.embeddings_endpoint),
            headers=self._headers(),
            json=body,
            timeout=self.timeout_s,
            verify=self._verify_setting(),
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        items = data.get("data")
        if not isinstance(items, list):
            raise ValueError("Invalid embeddings response: missing data")

        vectors: list[list[float]] = []
        for item in items:
            if not isinstance(item, dict) or "embedding" not in item:
                raise ValueError("Invalid embeddings response item")
            vectors.append([float(x) for x in item["embedding"]])

        if len(vectors) != len(texts):
            raise ValueError("Invalid embeddings response length")

        return vectors

    def embed_document(self, text: str, *, model: Optional[str] = None) -> List[float]:
        return self.embed_documents([text], model=model)[0]

    def embed_query(self, query: str, *, model: Optional[str] = None) -> List[float]:
        return self.embed_document(query, model=model)
