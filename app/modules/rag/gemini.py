"""
Gemini API client wrapper using the official Google Gen AI SDK (google-genai).

This module provides:
- Text generation via Gemini models (e.g. "gemini-3-pro-preview")
- Text embeddings via Gemini embedding models (e.g. "models/text-embedding-004")

Requirements:
- pip install google-genai
- Set GEMINI_API_KEY in environment / .env (loaded by app.core.config.settings)

References:
- https://ai.google.dev/gemini-api/docs
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
import inspect
import re
import time
from typing import Iterable, List, Optional, Sequence, Union

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class _GeminiRpmLimiter:
    """
    Global rate limiter for Gemini API calls.
    
    Enforces:
    - RPM (requests per minute) limit
    - Minimum interval between calls (prevents bursts)
    - Global semaphore to limit concurrent requests
    """
    
    def __init__(self, rpm: int, max_concurrent: int = 2):
        self._rpm = max(1, int(rpm))
        self._max_concurrent = max(1, int(max_concurrent))
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()
        # Global semaphore to limit concurrent Gemini calls
        self._semaphore = threading.Semaphore(self._max_concurrent)
        
    @property
    def rpm(self) -> int:
        return self._rpm
    
    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def acquire(self) -> None:
        """Acquire rate limit slot. Blocks until slot is available."""
        # First acquire semaphore (limits concurrent requests)
        self._semaphore.acquire()
        
        try:
            self._wait_for_rpm_slot()
        except Exception:
            # Release semaphore if RPM wait fails
            self._semaphore.release()
            raise
    
    def release(self) -> None:
        """Release the semaphore slot after request completes."""
        try:
            self._semaphore.release()
        except ValueError:
            pass  # Already released
    
    def _wait_for_rpm_slot(self) -> None:
        """Wait until we have an available RPM slot."""
        window_s = 60.0
        min_interval_s = window_s / float(self._rpm)

        while True:
            sleep_s = 0.0
            now = time.monotonic()
            with self._lock:
                # Drop timestamps outside the window
                while self._calls and (now - self._calls[0]) > window_s:
                    self._calls.popleft()

                # Check if we're at RPM limit
                if len(self._calls) >= self._rpm:
                    # Wait until oldest call expires from window
                    sleep_s = window_s - (now - self._calls[0]) + 0.1
                # Enforce min spacing between calls (prevents bursts)
                elif self._calls:
                    elapsed = now - self._calls[-1]
                    if elapsed < min_interval_s:
                        sleep_s = min_interval_s - elapsed
                
                if sleep_s <= 0.0:
                    self._calls.append(now)
                    return

            time.sleep(sleep_s)


# Rate limiter configuration from environment
# GEMINI_RPM_LIMIT: requests per minute (default 10 for free tier safety)
# GEMINI_MAX_CONCURRENT: max concurrent requests (default 2 to avoid bursts)
_GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "10") or "10")
_GEMINI_MAX_CONCURRENT = int(os.getenv("GEMINI_MAX_CONCURRENT", "2") or "2")
_gemini_rpm_limiter = _GeminiRpmLimiter(_GEMINI_RPM_LIMIT, _GEMINI_MAX_CONCURRENT)

logger.info(
    "Gemini rate limiter initialized",
    extra={"rpm_limit": _GEMINI_RPM_LIMIT, "max_concurrent": _GEMINI_MAX_CONCURRENT},
)

@dataclass
class GeminiModels:
    """Configuration for Gemini model names."""
    llm: str
    embedding: str

try:
    # Official SDK per docs: from google import genai; client = genai.Client()
    from google import genai
    try:
        from google.genai import types as genai_types  # type: ignore
    except Exception:  # pragma: no cover
        genai_types = None  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

from app.core.config import settings

GEMINI_API_KEY = settings.GEMINI_API_KEY
MODEL_NAME = settings.GEMINI_MODEL or 'gemini-2.0-flash' 
# FILE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/files"

# Singleton instance for GeminiAPI to avoid re-initialization on every request
_gemini_api_instance: Optional["GeminiAPI"] = None
_gemini_api_lock = threading.Lock()


def get_gemini_api() -> "GeminiAPI":
    """
    Get singleton GeminiAPI instance.
    Thread-safe, creates instance on first call.
    """
    global _gemini_api_instance
    if _gemini_api_instance is None:
        with _gemini_api_lock:
            if _gemini_api_instance is None:
                if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_RESPONSES_ENDPOINT:
                    from app.modules.rag.azure_openai import AzureOpenAIAPI

                    _gemini_api_instance = AzureOpenAIAPI()  # type: ignore[assignment]
                else:
                    _gemini_api_instance = GeminiAPI()
    return _gemini_api_instance


class GeminiAPI:
    """
    Production-ready Gemini client wrapper (sync).

    Notes:
    - API key is read from settings.GEMINI_API_KEY by default.
    - Uses the official `google-genai` SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        if genai is None:
            raise ImportError(
                "google-genai SDK is required for GeminiAPI. Install with: pip install google-genai"
            )
        api_key = api_key or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self._client = genai.Client(api_key=api_key)

        embedding = embedding_model or getattr(
            settings, "LIGHTRAG_EMBEDDING_MODEL", "models/text-embedding-004"
        )
        if embedding == "models/text-embedding-001":
            embedding = "models/text-embedding-004"

        self.models = GeminiModels(
            llm=llm_model
            or getattr(settings, "LIGHTRAG_LLM_MODEL", "gemini-2.5-flash"),
            embedding=embedding,
        )

        logger.info(
            "GeminiAPI initialized",
            extra={
                "llm_model": self.models.llm,
                "embedding_model": self.models.embedding,
            },
        )

    # -------------------------
    # LLM generation
    # -------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
    ) -> str:
        """
        Generates text using Gemini LLM.

        Args:
            prompt: User prompt / question.
            model: Override model name.
            system_instruction: Optional system instruction to guide the model.
            temperature: Sampling temperature.
            max_output_tokens: Output token cap.

        Returns:
            Generated text.
        """
        model_name = model or self.models.llm

        contents = prompt
        if system_instruction:
            # The SDK supports system instruction as separate parameter, but keeping
            # a safe universal approach by prepending as well can be redundant.
            # We pass it explicitly if supported by the SDK.
            pass

        def _parse_retry_seconds(msg: str) -> float | None:
            if not msg:
                return None
            m = re.search(r"retryDelay\"\s*:\s*\"(\d+(?:\.\d+)?)s\"", msg)
            if m:
                return float(m.group(1))
            m = re.search(r"retry in\s*~?(\d+(?:\.\d+)?)s", msg, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            m = re.search(r"retry after\s*(\d+(?:\.\d+)?)s", msg, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            return None

        last_exc: Exception | None = None
        for attempt in range(1, 8):  # Increased retries for rate limiting
            acquired = False
            try:
                _gemini_rpm_limiter.acquire()
                acquired = True
                
                config_value: object = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                try:
                    if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
                        cfg_sig = inspect.signature(genai_types.GenerateContentConfig)
                        cfg_kwargs: dict[str, object] = {
                            "temperature": temperature,
                            "max_output_tokens": max_output_tokens,
                        }
                        if "automatic_function_calling" in cfg_sig.parameters:
                            cfg_kwargs["automatic_function_calling"] = None
                        if "tools" in cfg_sig.parameters:
                            cfg_kwargs["tools"] = None
                        config_value = genai_types.GenerateContentConfig(**cfg_kwargs)
                except Exception:
                    config_value = {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    }
                
                # Try to pass system_instruction if supported
                try:
                    if system_instruction:
                        response = self._client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            system_instruction=system_instruction,
                            config=config_value,
                        )
                    else:
                        response = self._client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config_value,
                        )
                except (TypeError, AttributeError):
                    # Fallback: try without config parameter
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )

                # Extract text from response - try multiple formats
                text = None
                if hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    # Some SDK versions return candidates[0].content.parts[0].text
                    candidate = response.candidates[0]
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        if candidate.content.parts:
                            text = getattr(candidate.content.parts[0], "text", None)

                if not text:
                    logger.warning(
                        "Empty response from Gemini generate_content",
                        extra={
                            "model": model_name,
                            "response_type": type(response).__name__,
                            "response_attrs": dir(response) if hasattr(response, "__dict__") else None,
                        },
                    )
                    raise ValueError("Empty response.text from Gemini generate_content")
                return text
            except Exception as e:
                last_exc = e
                msg = str(e)
                is_rate = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower()) or ("quota" in msg.lower())
                if attempt >= 7 or not is_rate:
                    logger.error(
                        "Gemini generate_text failed",
                        extra={
                            "model": model_name,
                            "prompt_len": len(prompt),
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": attempt,
                        },
                    )
                    raise

                # Exponential backoff with jitter for rate limiting
                retry_s = _parse_retry_seconds(msg)
                if retry_s is None:
                    # Exponential backoff: 4, 8, 16, 32, 64, 128 seconds
                    retry_s = min(128.0, 4.0 * (2.0 ** (attempt - 1)))
                retry_s = max(4.0, float(retry_s))  # Minimum 4 seconds

                logger.warning(
                    "Gemini rate-limited; retrying",
                    extra={"attempt": attempt, "sleep_s": retry_s, "model": model_name},
                )
                time.sleep(retry_s)
            finally:
                if acquired:
                    _gemini_rpm_limiter.release()

        assert last_exc is not None
        raise last_exc

    # Backwards-compatible alias (some parts of the project used this name)
    def generate_content(self, prompt: str) -> str:
        return self.generate_text(prompt)

    # -------------------------
    # Embeddings
    # -------------------------

    def embed_query(self, query: str, *, model: Optional[str] = None) -> List[float]:
        """
        Embedding for search queries (retrieval_query).

        Returns:
            Vector embedding (list of floats).
        """
        return self._embed_one(query, task_type="retrieval_query", model=model)

    def embed_document(self, text: str, *, model: Optional[str] = None) -> List[float]:
        """
        Embedding for documents/chunks (retrieval_document).

        Returns:
            Vector embedding (list of floats).
        """
        return self._embed_one(text, task_type="retrieval_document", model=model)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Batch embeddings for documents/chunks.

        Notes:
        - The SDK supports embedding a single content at a time; we batch by looping.
        - If your SDK version supports true batch embeddings, you can optimize here.

        Returns:
            List of embeddings, aligned with `texts`.
        """
        vectors: List[List[float]] = []
        for t in texts:
            vectors.append(self.embed_document(t, model=model))
        return vectors

    def _embed_one(
        self,
        content: str,
        *,
        task_type: str,
        model: Optional[str] = None,
    ) -> List[float]:
        model_name = model or self.models.embedding
        acquired = False
        try:
            _gemini_rpm_limiter.acquire()
            acquired = True
            # Official REST is models.embedContent; SDK exposes equivalent under client.models.
            # Try different formats for contents parameter
            try:
                # First try: contents as string (some SDK versions)
                resp = self._client.models.embed_content(
                    model=model_name,
                    contents=content,
                    task_type=task_type,
                )
            except (TypeError, AttributeError, ValueError) as e1:
                # Second try: contents as list (other SDK versions)
                try:
                    resp = self._client.models.embed_content(
                        model=model_name,
                        contents=[content],
                        task_type=task_type,
                    )
                except Exception as e2:
                    # Third try: without task_type (fallback)
                    try:
                        resp = self._client.models.embed_content(
                            model=model_name,
                            contents=content,
                        )
                    except Exception as e3:
                        # Check for 403 Forbidden (API key or permissions issue)
                        error_str = str(e3).lower()
                        if "403" in error_str or "forbidden" in error_str or "permission" in error_str:
                            logger.error(
                                "Gemini API 403 Forbidden - check API key and permissions",
                                extra={
                                    "model": model_name,
                                    "task_type": task_type,
                                    "error": str(e3),
                                    "hint": "Verify GEMINI_API_KEY is valid and has embedding API access",
                                },
                            )
                            raise ValueError(
                                "Gemini API access denied (403 Forbidden). "
                                "Please check your GEMINI_API_KEY and ensure it has access to embedding models."
                            ) from e3
                        
                        logger.error(
                            "Gemini embedding failed with all formats",
                            extra={
                                "model": model_name,
                                "task_type": task_type,
                                "error1": str(e1),
                                "error2": str(e2),
                                "error3": str(e3),
                            },
                        )
                        raise e3
            
            # The SDK returns an object with `.embedding` or dict-like.
            # Try multiple ways to extract embedding
            # SDK may use batch API even for single items, so check for batch structure
            emb = None
            
            # Check for batch response structure (embeddings list)
            if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list) and len(resp.embeddings) > 0:
                # Batch response: take first embedding
                first_emb = resp.embeddings[0]
                if hasattr(first_emb, "values"):
                    emb = first_emb.values
                elif hasattr(first_emb, "embedding"):
                    emb = first_emb.embedding
                elif isinstance(first_emb, dict):
                    emb = first_emb.get("values") or first_emb.get("embedding")
            
            # Check for direct embedding attribute
            if emb is None and hasattr(resp, "embedding"):
                emb = resp.embedding
            
            # Check for values attribute
            if emb is None and hasattr(resp, "values"):
                emb = resp.values
            
            # Check for dict response
            if emb is None and isinstance(resp, dict):
                # Check for batch structure in dict
                if "embeddings" in resp and isinstance(resp["embeddings"], list) and len(resp["embeddings"]) > 0:
                    first_emb = resp["embeddings"][0]
                    if isinstance(first_emb, dict):
                        emb = first_emb.get("values") or first_emb.get("embedding")
                else:
                    emb = resp.get("embedding") or resp.get("values")
            
            # Check for data attribute (some SDK versions)
            if emb is None and hasattr(resp, "data") and isinstance(resp.data, list) and len(resp.data) > 0:
                # Some SDK versions return embedding in data[0].embedding
                first_item = resp.data[0]
                if hasattr(first_item, "embedding"):
                    emb = first_item.embedding
                elif hasattr(first_item, "values"):
                    emb = first_item.values
                elif isinstance(first_item, dict):
                    emb = first_item.get("embedding") or first_item.get("values")
            
            # Check for result/response wrapper
            if emb is None and hasattr(resp, "result"):
                result = resp.result
                if hasattr(result, "embeddings") and isinstance(result.embeddings, list) and len(result.embeddings) > 0:
                    first_emb = result.embeddings[0]
                    if hasattr(first_emb, "values"):
                        emb = first_emb.values
                    elif hasattr(first_emb, "embedding"):
                        emb = first_emb.embedding
                elif hasattr(result, "values"):
                    emb = result.values
                elif hasattr(result, "embedding"):
                    emb = result.embedding
            
            # Try to convert to dict if possible (some SDK objects have to_dict method)
            if emb is None:
                try:
                    if hasattr(resp, "to_dict"):
                        resp_dict = resp.to_dict()
                        if isinstance(resp_dict, dict):
                            if "embeddings" in resp_dict and isinstance(resp_dict["embeddings"], list) and len(resp_dict["embeddings"]) > 0:
                                first_emb = resp_dict["embeddings"][0]
                                if isinstance(first_emb, dict):
                                    emb = first_emb.get("values") or first_emb.get("embedding")
                            else:
                                emb = resp_dict.get("embedding") or resp_dict.get("values")
                except Exception:
                    pass
            
            if emb is None:
                # Log the response structure for debugging
                resp_dict = {}
                resp_attrs = []
                try:
                    if hasattr(resp, "__dict__"):
                        resp_dict = {k: str(type(v).__name__) + ":" + str(v)[:100] for k, v in resp.__dict__.items()}
                        resp_attrs = [attr for attr in dir(resp) if not attr.startswith("_")][:30]
                    elif isinstance(resp, dict):
                        resp_dict = {k: str(type(v).__name__) + ":" + str(v)[:100] for k, v in resp.items()}
                except Exception:
                    pass
                
                # Try to get string representation
                resp_str = str(resp)[:500] if resp else "None"
                
                logger.error(
                    "No embedding found in response",
                    extra={
                        "model": model_name,
                        "task_type": task_type,
                        "resp_type": type(resp).__name__,
                        "resp_attrs": resp_attrs,
                        "resp_dict_keys": list(resp_dict.keys())[:30] if resp_dict else None,
                        "resp_preview": resp_str,
                    },
                )
                raise ValueError(f"No embedding returned by Gemini embed_content. Response type: {type(resp).__name__}")
            
            # Ensure we return a list of floats
            if isinstance(emb, list):
                result = [float(x) for x in emb]
                logger.debug(
                    "Successfully extracted embedding",
                    extra={
                        "model": model_name,
                        "embedding_len": len(result),
                        "resp_type": type(resp).__name__,
                    },
                )
                return result
            elif hasattr(emb, "__iter__"):
                result = [float(x) for x in emb]
                logger.debug(
                    "Successfully extracted embedding from iterable",
                    extra={
                        "model": model_name,
                        "embedding_len": len(result),
                        "resp_type": type(resp).__name__,
                    },
                )
                return result
            else:
                raise ValueError(f"Unexpected embedding type: {type(emb)}")
                
        except Exception as e:
            # Check for 403 Forbidden (API key or permissions issue)
            error_str = str(e).lower()
            if "403" in error_str or "forbidden" in error_str or "permission" in error_str:
                logger.error(
                    "Gemini API 403 Forbidden - check API key and permissions",
                    extra={
                        "model": model_name,
                        "task_type": task_type,
                        "content_len": len(content),
                        "error": str(e),
                        "hint": "Verify GEMINI_API_KEY is valid and has embedding API access",
                    },
                )
                raise ValueError(
                    "Gemini API access denied (403 Forbidden). "
                    "Please check your GEMINI_API_KEY and ensure it has access to embedding models."
                ) from e
            
            logger.error(
                "Gemini embedding failed",
                extra={
                    "model": model_name,
                    "task_type": task_type,
                    "content_len": len(content),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise
        finally:
            if acquired:
                _gemini_rpm_limiter.release()