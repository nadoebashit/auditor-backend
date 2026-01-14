from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from arq.connections import RedisSettings
from arq import create_pool, ArqRedis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.logging import configure_logging, get_logger
from app.modules.files.models import FileIndexStatus, StoredFile
from app.modules.files.qdrant_client import QdrantVectorStore
from app.modules.files.service_hybrid import HybridFileService
from app.modules.files.storage import FileStorage, S3Config
from app.modules.files.file_text_extractor import extract_text
from app.modules.rag.lightrag_integration import create_lightrag_service
from app.modules.embeddings.service import get_embedding_service
from app.modules.rag.gemini import GeminiAPI, get_gemini_api

logger = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging()
    logger.info("Hybrid file indexer worker starting up")

    # Debug: confirm interpreter + package location
    import sys
    logger.info("PYTHON executable: %s", sys.executable)
    try:
        import lightrag
        logger.info("lightrag module file: %s", lightrag.__file__)
    except Exception as e:
        logger.warning("Could not import lightrag in worker startup: %s", e)

    import app.modules.auth.models as _auth_models

    ctx["storage_cfg"] = S3Config(
        endpoint_url=settings.S3_ENDPOINT_URL,
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
        region=settings.S3_REGION,
        bucket_admin_laws=settings.S3_BUCKET_ADMIN_LAWS,
        bucket_customer_docs=settings.S3_BUCKET_CUSTOMER_DOCS,
    )

    # Gemini client (LLM + embeddings) - used by HybridFileService (Qdrant pipeline)
    ctx["gemini"] = get_gemini_api()

    # Qdrant vector stores (two namespaces per TZ)
    # G1: oson_knowledge (Knowledge Base / Block B)
    # G1_Client: client_documents (Client documents)
    admin_collection = settings.QDRANT_COLLECTION_ADMIN
    client_collection = settings.QDRANT_COLLECTION_CLIENT

    ctx["qdrant_admin"] = QdrantVectorStore(
        url=settings.QDRANT_URL,
        collection_name=admin_collection,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    ctx["qdrant_client"] = QdrantVectorStore(
        url=settings.QDRANT_URL,
        collection_name=client_collection,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    
    logger.info(
        "Qdrant namespaces initialized",
        extra={
            "admin_collection": admin_collection,
            "client_collection": client_collection,
            "vector_size": settings.QDRANT_VECTOR_SIZE,
        },
    )

    # LightRAG (graph-based RAG) is initialized per-workspace inside index_file_task.
    # Initializing it here without a workspace creates an empty root workspace on disk.
    ctx["lightrag"] = None


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Hybrid file indexer worker shutting down")


async def index_file_task(ctx: dict[str, Any], file_id: str) -> None:
    configure_logging()
    logger.info("Index file task started", extra={"file_id": file_id})

    try:
        file_uuid = uuid.UUID(file_id)
    except Exception:
        logger.error("Invalid file_id (expected UUID)", extra={"file_id": file_id})
        return

    db: Session = SessionLocal()  # новая сессия для воркера
    try:
        storage = FileStorage(cfg=ctx["storage_cfg"])
        gemini_api: GeminiAPI = ctx["gemini"]
        qdrant_admin: QdrantVectorStore = ctx["qdrant_admin"]
        qdrant_client: QdrantVectorStore = ctx["qdrant_client"]
        lightrag_service = ctx.get("lightrag")

        service = HybridFileService(
            db=db,
            storage=storage,
            vector_store_admin=qdrant_admin,
            vector_store_client=qdrant_client,
            lightrag_service=lightrag_service,
            gemini_api=gemini_api,
        )

        # Обновляем статус файла -> RUNNING
        stored_file: StoredFile | None = db.get(StoredFile, file_uuid)
        if not stored_file:
            logger.warning("Stored file not found", extra={"file_id": file_id})
            return

        stored_file.index_status = FileIndexStatus.RUNNING
        stored_file.index_error = None
        db.commit()

        # Индексация файла
        await asyncio.to_thread(service.index_file, file_uuid)

        # Обновляем статус после индексации
        stored_file = db.get(StoredFile, file_uuid)
        if stored_file and stored_file.is_indexed and not stored_file.index_error:
            stored_file.index_status = FileIndexStatus.DONE
            stored_file.indexed_at = datetime.utcnow()
            db.commit()

            # LightRAG second-signal indexing (best-effort)
            try:
                obj = storage.download_file(stored_file.bucket, stored_file.object_key)
                file_bytes = obj.read()
                if hasattr(obj, "close"):
                    obj.close()
                if hasattr(obj, "release_conn"):  # type: ignore[attr-defined]
                    obj.release_conn()

                text = extract_text(
                    file_bytes=file_bytes,
                    content_type=stored_file.content_type,
                    filename=stored_file.original_filename,
                )

                if text:
                    scope = stored_file.scope.value
                    if scope != "ADMIN_LAW" and settings.LIGHTRAG_ADMIN_ONLY:
                        pass
                    else:
                        if scope == "ADMIN_LAW":
                            workspace = "admin_law"
                        else:
                            workspace = f"customer_{stored_file.customer_id}"

                        lightrag = create_lightrag_service(
                            working_dir=settings.LIGHTRAG_WORKING_DIR,
                            workspace=workspace,
                        )

                        await lightrag.ainsert(
                            text=text,
                            file_path=stored_file.original_filename or f"{scope}/{stored_file.id}",
                        )
            except Exception as exc:
                logger.warning(
                    "LightRAG second-signal indexing failed",
                    extra={"file_id": file_id, "error": str(exc)},
                )

        logger.info(
            "Index file task completed",
            extra={
                "file_id": file_id,
                "status": stored_file.index_status if stored_file else None,
            },
        )

    except Exception as exc:
        logger.exception(
            "Index file task failed",
            extra={"file_id": file_id, "error": str(exc)},
        )
        # Пытаемся зафиксировать ошибку в БД
        try:
            db.rollback()
            stored_file = db.get(StoredFile, file_uuid)
            if stored_file:
                stored_file.index_status = FileIndexStatus.ERROR
                stored_file.is_indexed = False
                stored_file.index_error = "Index file task failed (see logs)"
                stored_file.indexed_at = datetime.utcnow()
                db.commit()
        except Exception as inner_exc:
            logger.exception(
                "Failed to update stored_file error status",
                extra={"file_id": file_id},
            )



class WorkerSettings:
    functions = [index_file_task]
    on_startup = startup
    on_shutdown = shutdown

    max_jobs = int(getattr(settings, "ARQ_MAX_JOBS", 1) or 1)
    job_timeout = int(getattr(settings, "ARQ_JOB_TIMEOUT_S", 1800) or 1800)

    # Передаём либо RedisSettings, либо именно строку (но не AnyUrl)
    redis_settings = RedisSettings.from_dsn(str(settings.REDIS_URL))


async def create_redis_pool() -> ArqRedis:
    return await create_pool(RedisSettings.from_dsn(str(settings.REDIS_URL)))

