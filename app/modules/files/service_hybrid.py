# app/modules/files/service_hybrid.py
"""
Production-ready гибридный FileService с Qdrant + LightRAG.
Включает batch processing, метрики времени и обработку ошибок.
"""

from __future__ import annotations

import io
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List

from qdrant_client.models import PointStruct
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.modules.files.chunking import (
    chunk_by_section,
    chunk_text_simple,
    Chunk,
    ChunkMetadata,
    extract_isa_references,
    extract_ifrs_references,
    detect_audit_cycle,
)
from app.modules.files.file_text_extractor import extract_text
from app.modules.files.models import FileChunk, FileIndexStatus, FileScope, StoredFile
from app.modules.files.qdrant_client import QdrantVectorStore
from app.modules.files.storage import FileStorage
from app.modules.rag.gemini import GeminiAPI
from app.modules.rag.lightrag_integration import LightRAGService

logger = get_logger(__name__)


class EmbeddingProvider:
    """Production embedding provider with batch support."""

    def __init__(self, gemini_api: GeminiAPI):
        self.gemini = gemini_api

    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        return self.gemini.embed_document(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts."""
        return self.gemini.embed_documents(texts)


def chunk_text(text: str, chunk_size: int | None = None) -> List[str]:
    """
    Разбивает текст на чанки (обратная совместимость).
    Для section-based используйте chunk_by_section().
    """
    return chunk_text_simple(text, chunk_size)


class HybridFileService:
    """
    Гибридный сервис индексации: Qdrant + LightRAG.

    Особенности:
    - Batch processing для эмбеддингов
    - Параллельная индексация в обе системы
    - Детальные метрики времени
    - Graceful degradation (если одна система упала)
    """

    def __init__(
        self,
        db: Session,
        storage: FileStorage,
        vector_store_admin: QdrantVectorStore | None,
        vector_store_client: QdrantVectorStore | None,
        lightrag_service: LightRAGService | None,
        gemini_api: GeminiAPI,
    ):
        self.db = db
        self.storage = storage
        self.vector_store_admin = vector_store_admin
        self.vector_store_client = vector_store_client
        self.lightrag = lightrag_service
        self.embedding_provider = EmbeddingProvider(gemini_api)

    def upload_admin_file(self, user, file) -> StoredFile:
        """Загрузка файла админа (без индексации здесь)."""
        logger.info(
            "Uploading admin file",
            extra={
                "user_id": str(user.id),
                "file_name": file.filename,
                "content_type": file.content_type,
            },
        )

        content = file.file.read()
        object_key = f"{uuid.uuid4()}_{file.filename}"

        self.storage.upload_admin_file(
            file_obj=io.BytesIO(content),
            object_key=object_key,
            content_type=file.content_type,
        )

        stored_file = StoredFile(
            owner_id=user.id,
            customer_id=None,
            scope=FileScope.ADMIN_LAW,
            bucket=self.storage._cfg.bucket_admin_laws,
            object_key=object_key,
            original_filename=file.filename,
            content_type=file.content_type,
            size_bytes=len(content),
            is_indexed=False,
            index_error=None,
            index_status=FileIndexStatus.QUEUED,
        )
        self.db.add(stored_file)
        self.db.commit()
        self.db.refresh(stored_file)

        logger.info(
            "Admin file uploaded",
            extra={
                "stored_file_id": str(stored_file.id),
                "size_bytes": stored_file.size_bytes,
            },
        )

        return stored_file

    def upload_customer_file(self, user, customer_id: str, file) -> StoredFile:
        """Загрузка файла заказчика."""
        logger.info(
            "Uploading customer file",
            extra={
                "user_id": str(user.id),
                "customer_id": customer_id,
                "file_name": file.filename,
            },
        )

        content = file.file.read()
        object_key = f"{uuid.uuid4()}_{file.filename}"

        self.storage.upload_customer_file(
            customer_id=customer_id,
            file_obj=io.BytesIO(content),
            object_key=object_key,
            content_type=file.content_type,
        )

        stored_file = StoredFile(
            owner_id=user.id,
            customer_id=customer_id,
            scope=FileScope.CUSTOMER_DOC,
            bucket=self.storage._cfg.bucket_customer_docs,
            object_key=f"{customer_id}/{object_key}",
            original_filename=file.filename,
            content_type=file.content_type,
            size_bytes=len(content),
            is_indexed=False,
            index_error=None,
            index_status=FileIndexStatus.QUEUED,
        )
        self.db.add(stored_file)
        self.db.commit()
        self.db.refresh(stored_file)

        logger.info(
            "Customer file uploaded",
            extra={
                "stored_file_id": str(stored_file.id),
                "customer_id": customer_id,
                "size_bytes": stored_file.size_bytes,
            },
        )

        return stored_file

    def index_file(self, file_id: uuid.UUID) -> Dict[str, Any]:
        """
        ГИБРИДНАЯ ИНДЕКСАЦИЯ: Qdrant + LightRAG.

        Returns:
            Метрики времени для каждого шага.
        """
        metrics = {
            "file_id": str(file_id),
            "started_at": time.time(),
            "times": {},
            "status": "started",
        }

        stored_file = self.db.query(StoredFile).get(file_id)
        if not stored_file:
            logger.warning("File not found", extra={"file_id": str(file_id)})
            return {"status": "error", "error": "File not found"}

        logger.info(
            "Starting hybrid indexing",
            extra={
                "file_id": str(file_id),
                "file_name": stored_file.original_filename,
                "size_bytes": stored_file.size_bytes,
            },
        )

        try:
            # ═══════════════════════════════════════════
            # ШАГ 1: СКАЧИВАНИЕ
            # ═══════════════════════════════════════════
            t_start = time.time()

            obj = self.storage.download_file(stored_file.bucket, stored_file.object_key)
            file_bytes = obj.read()

            if hasattr(obj, "close"):
                obj.close()
            if hasattr(obj, "release_conn"):
                obj.release_conn()

            metrics["times"]["download"] = time.time() - t_start

            # ═══════════════════════════════════════════
            # ШАГ 2: ПАРСИНГ
            # ═══════════════════════════════════════════
            t_start = time.time()

            text = extract_text(
                file_bytes=file_bytes,
                content_type=stored_file.content_type,
                filename=stored_file.original_filename,
            )

            metrics["times"]["parse"] = time.time() - t_start
            metrics["text_length"] = len(text)

            if not text:
                logger.warning("No text extracted")
                stored_file.is_indexed = False
                stored_file.index_error = "No text extracted"
                stored_file.index_status = FileIndexStatus.ERROR
                self.db.commit()
                return metrics

            # ═══════════════════════════════════════════
            # ШАГ 3: ЧАНКИНГ (Section-based для Block B)
            # ═══════════════════════════════════════════
            t_start = time.time()

            # Используем section-based chunking для структурированных документов
            section_chunks = chunk_by_section(
                text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=100,
                min_chunk_size=50,
            )
            
            # Извлекаем ISA ссылки из всего документа
            doc_isa_refs = extract_isa_references(text)
            doc_cycle = detect_audit_cycle(text)
            
            metrics["num_chunks"] = len(section_chunks)
            metrics["chunking_strategy"] = "section_based" if len(section_chunks) > 1 else "size_based"
            metrics["isa_references"] = doc_isa_refs
            metrics["audit_cycle"] = doc_cycle
            metrics["times"]["chunking"] = time.time() - t_start

            logger.info(
                "RAG_PIPELINE: Chunking complete",
                extra={
                    "file_id": str(file_id),
                    "num_chunks": len(section_chunks),
                    "strategy": metrics["chunking_strategy"],
                    "isa_refs": doc_isa_refs[:5] if doc_isa_refs else [],
                    "cycle": doc_cycle,
                    "chunking_time_ms": int(metrics["times"]["chunking"] * 1000),
                },
            )

            # Метаданные
            base_metadata = {
                "file_id": str(stored_file.id),
                "scope": stored_file.scope.value,
                "customer_id": stored_file.customer_id,
                "owner_id": str(stored_file.owner_id),
                "filename": stored_file.original_filename,
                "title": getattr(stored_file, "title", None),
                "version_date": (
                    stored_file.version_date.isoformat()
                    if getattr(stored_file, "version_date", None)
                    else None
                ),
                "language": getattr(stored_file, "language", None),
                "description": getattr(stored_file, "description", None),
                "source_type": stored_file.content_type,
            }

            # ═══════════════════════════════════════════
            # ШАГ 4: ИНДЕКСАЦИЯ В QDRANT
            # ═══════════════════════════════════════════
            vector_store: QdrantVectorStore | None
            if stored_file.scope == FileScope.ADMIN_LAW:
                vector_store = self.vector_store_admin
            else:
                vector_store = self.vector_store_client

            if vector_store:
                t_start = time.time()

                try:
                    self._index_to_qdrant(
                        stored_file=stored_file,
                        section_chunks=section_chunks,
                        base_metadata=base_metadata,
                        metrics=metrics,
                        vector_store=vector_store,
                        doc_isa_refs=doc_isa_refs,
                        doc_cycle=doc_cycle,
                    )

                    metrics["times"]["qdrant_total"] = time.time() - t_start
                    metrics["qdrant_status"] = "success"

                except Exception as e:
                    logger.exception("Qdrant indexing failed", extra={"file_id": str(file_id)})

                    try:
                        self.db.rollback()
                    except Exception:
                        pass

                    metrics["qdrant_status"] = "failed"
                    metrics["qdrant_error"] = str(e)
            else:
                logger.warning(
                    "Qdrant vector store not available, skipping Qdrant indexing",
                    extra={"file_id": str(file_id)},
                )
                metrics["qdrant_status"] = "skipped"

            # ═══════════════════════════════════════════
            # ШАГ 5: ИНДЕКСАЦИЯ В LIGHTRAG
            # ═══════════════════════════════════════════
            if self.lightrag:
                t_start = time.time()

                try:
                    # Для LightRAG используем только тексты
                    chunk_texts = [c.text for c in section_chunks]
                    self._index_to_lightrag(
                        stored_file=stored_file,
                        chunks=chunk_texts,
                        base_metadata=base_metadata,
                        metrics=metrics,
                    )

                    metrics["times"]["lightrag_total"] = time.time() - t_start
                    metrics["lightrag_status"] = "success"

                except Exception as e:
                    logger.exception("LightRAG indexing failed",
                     extra={"file_id": str(file_id)}
                     )

                    try:
                        self.db.rollback()
                    except Exception:
                        pass

                    metrics["lightrag_status"] = "failed"
                    metrics["lightrag_error"] = str(e)
            else:
                logger.debug(
                    "LightRAG not available, skipping LightRAG indexing",
                    extra={"file_id": str(file_id)},
                )
                metrics["lightrag_status"] = "skipped"

            # ═══════════════════════════════════════════
            # ФИНАЛИЗАЦИЯ
            # ═══════════════════════════════════════════
            qdrant_ok = metrics.get("qdrant_status") == "success"
            lightrag_ok = metrics.get("lightrag_status") == "success"
            successful = bool(qdrant_ok or lightrag_ok)

            stored_file.is_indexed = successful
            stored_file.index_status = (
                FileIndexStatus.DONE if successful else FileIndexStatus.ERROR
            )
            if successful:
                stored_file.index_error = None
            else:
                stored_file.index_error = (
                    metrics.get("qdrant_error")
                    or metrics.get("lightrag_error")
                    or "Indexing failed"
                )
            stored_file.indexed_at = datetime.utcnow()
            self.db.commit()

            metrics["total_time"] = time.time() - metrics["started_at"]
            metrics["status"] = "success"

            logger.info(
                "Hybrid indexing completed",
                extra={
                    "file_id": str(file_id),
                    "total_time": metrics["total_time"],
                    "metrics": metrics,
                },
            )

            return metrics

        except Exception as exc:
            logger.exception(
                "Hybrid indexing failed",
                extra={"file_id": str(file_id)},
            )

            try:
                self.db.rollback()
            except Exception:
                pass

            stored_file.is_indexed = False
            stored_file.index_error = str(exc)
            stored_file.index_status = FileIndexStatus.ERROR
            stored_file.indexed_at = datetime.utcnow()
            self.db.commit()

            metrics["status"] = "error"
            metrics["error"] = str(exc)
            return metrics

    def _index_to_qdrant(
        self,
        stored_file: StoredFile,
        section_chunks: List[Chunk],
        base_metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        vector_store: QdrantVectorStore,
        doc_isa_refs: List[str] | None = None,
        doc_cycle: str | None = None,
    ) -> None:
        """
        Индексация в Qdrant с batch embeddings и расширенным payload.
        
        Расширенный payload по ТЗ:
        - block: Block B для Knowledge Base
        - section: название секции из документа
        - isa_reference: список ссылок на ISA стандарты
        - cycle: цикл аудита (acceptance, planning, execution, reporting, completion)
        """
        if not section_chunks:
            logger.warning("No chunks to index to Qdrant")
            return

        kb_file_id: str | None = None
        kb_block: str | None = None
        file_name = (getattr(stored_file, "original_filename", None) or "").strip()
        if stored_file.scope == FileScope.ADMIN_LAW and file_name:
            m = re.match(r"^(?P<kb_id>[A-F]\d+)_", file_name)
            if m:
                kb_file_id = m.group("kb_id")
                kb_block = kb_file_id[0] if kb_file_id else None

        block_value = (
            kb_block
            if kb_block
            else ("ADMIN_LAW" if stored_file.scope == FileScope.ADMIN_LAW else "CLIENT_DOC")
        )

        tree_type: str | None = None
        if kb_file_id:
            if kb_file_id == "D1":
                tree_type = "legal"
            elif kb_file_id == "D2":
                tree_type = "acceptance"
            elif kb_file_id == "D3":
                tree_type = "opinion"
            elif kb_file_id == "D4":
                tree_type = "going_concern"

        # ШАГ 1: Batch эмбеддинги
        t_start = time.time()
        
        chunk_texts = [c.text for c in section_chunks]
        embeddings = self.embedding_provider.embed_batch(chunk_texts)

        doc_text = "\n".join(chunk_texts)
        decision_nodes: int | None = None
        outcomes: list[str] | None = None
        if kb_block == "D" and doc_text:
            try:
                decision_nodes = len(re.findall(r"^\s*NODE\s+\w+", doc_text, flags=re.MULTILINE))
            except Exception:
                decision_nodes = None
            try:
                outcomes = sorted(set(re.findall(r"^\s*(OUTCOME_[A-Z0-9_]+)", doc_text, flags=re.MULTILINE)))
            except Exception:
                outcomes = None

        doc_ifrs_refs: list[str] | None = None
        if doc_text:
            try:
                doc_ifrs_refs = extract_ifrs_references(doc_text)
            except Exception:
                doc_ifrs_refs = None

        metrics["times"]["qdrant_embeddings"] = time.time() - t_start
        
        logger.info(
            "RAG_PIPELINE: Embeddings generated",
            extra={
                "file_id": str(stored_file.id),
                "num_embeddings": len(embeddings),
                "embedding_time_ms": int(metrics["times"]["qdrant_embeddings"] * 1000),
            },
        )

        # ШАГ 2: Создание FileChunk + PointStruct с расширенным payload
        t_start = time.time()

        qdrant_points: List[PointStruct] = []
        chunk_rows: List[FileChunk] = []

        for section_chunk, _embedding in zip(section_chunks, embeddings):
            meta = section_chunk.metadata
            
            # Извлекаем ISA ссылки из конкретного чанка (дополняет документные)
            chunk_isa_refs = extract_isa_references(section_chunk.text)
            combined_isa_refs = list(set((doc_isa_refs or []) + chunk_isa_refs))

            # IFRS/IAS references (TZ: ifrs_reference payload)
            chunk_ifrs_refs = extract_ifrs_references(section_chunk.text)
            combined_ifrs_refs = list(set((doc_ifrs_refs or []) + chunk_ifrs_refs))
            
            # Определяем цикл для чанка (fallback на документный)
            chunk_cycle = detect_audit_cycle(section_chunk.text) or doc_cycle
            
            chunk = FileChunk(
                file_id=stored_file.id,
                chunk_index=meta.chunk_index,
                text=section_chunk.text,
                customer_id=stored_file.customer_id,
                owner_id=str(stored_file.owner_id),
                scope=stored_file.scope.value,
                source_type=stored_file.content_type or None,
                section=meta.section_title,
                char_start=meta.char_start,
                char_end=meta.char_end,
            )
            self.db.add(chunk)
            chunk_rows.append((chunk, section_chunk, combined_isa_refs, chunk_cycle, combined_ifrs_refs))

        # Ensure UUIDs are generated before we reference chunk.id
        self.db.flush()

        for chunk, section_chunk, isa_refs, cycle, ifrs_refs in chunk_rows:
            meta = section_chunk.metadata
            chunk_id = str(chunk.id)
            point_id = chunk_id
            chunk.qdrant_point_id = point_id

            formula_id: str | None = None
            method_type: str | None = None
            if kb_file_id in {"C1", "C2"}:
                section_text = (meta.section_title or "").lower()
                if kb_file_id == "C1":
                    if "performance" in section_text or "pm" in section_text:
                        formula_id = "PM"
                    elif "clearly trivial" in section_text or "ctt" in section_text:
                        formula_id = "CTT"
                    elif "specific materiality" in section_text or "sm" in section_text:
                        formula_id = "SM"
                    elif "acceptable benchmarks" in section_text or "benchmarks" in section_text:
                        formula_id = "BENCHMARK"
                    elif "policy" in section_text:
                        formula_id = "OM"
                elif kb_file_id == "C2":
                    if "mus" in section_text or "pps" in section_text:
                        method_type = "MUS"
                    elif "attribute" in section_text:
                        method_type = "Attribute"
                    elif "variables" in section_text:
                        method_type = "Variables"

            # Расширенный payload по ТЗ
            payload = {
                **base_metadata,
                "chunk_id": chunk_id,
                # TZ/G1: public KB id in file_id; keep stored_file_id separately.
                "file_id": kb_file_id if (stored_file.scope == FileScope.ADMIN_LAW and kb_file_id) else str(chunk.file_id),
                "stored_file_id": str(chunk.file_id),
                "stored_file_original_filename": file_name or None,
                "chunk_index": int(chunk.chunk_index),
                "text": chunk.text,
                # Новые поля по ТЗ
                "block": block_value,
                "kb_file_id": kb_file_id,
                "section": meta.section_title,
                "section_level": meta.section_level,
                "isa_reference": isa_refs,
                "ifrs_reference": ifrs_refs,
                "cycle": cycle,
                "char_start": meta.char_start,
                "char_end": meta.char_end,
            }

            if tree_type is not None:
                payload["tree_type"] = tree_type
            if isinstance(decision_nodes, int):
                payload["decision_nodes"] = int(decision_nodes)
            if outcomes:
                payload["outcomes"] = outcomes
            if formula_id is not None:
                payload["formula_id"] = formula_id
            if method_type is not None:
                payload["method_type"] = method_type
            
            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=_embedding,
                    payload=payload,
                )
            )

        self.db.commit()

        metrics["times"]["qdrant_prepare"] = time.time() - t_start

        # ШАГ 3: Batch upsert в Qdrant
        t_start = time.time()

        batch_size = settings.QDRANT_BATCH_SIZE
        num_batches = (len(qdrant_points) + batch_size - 1) // batch_size

        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]
            vector_store.upsert_vectors(batch)

        metrics["times"]["qdrant_upsert"] = time.time() - t_start
        metrics["qdrant_batches"] = num_batches

        logger.info(
            "RAG_PIPELINE: Qdrant indexing complete",
            extra={
                "file_id": str(stored_file.id),
                "num_points": len(qdrant_points),
                "num_batches": num_batches,
                "upsert_time_ms": int(metrics["times"]["qdrant_upsert"] * 1000),
            },
        )

    def _index_to_lightrag(
        self,
        stored_file: StoredFile,
        chunks: List[str],
        base_metadata: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Индексация в LightRAG с merge chunks.
        Важно: в insert() передаём СТРОКУ large_chunk, а не функцию chunk_text.
        """

        # Нечего индексировать
        if not chunks:
            metrics["lightrag_large_chunks"] = 0
            metrics["times"]["lightrag_merge"] = 0.0
            metrics["times"]["lightrag_insert"] = 0.0
            logger.info("LightRAG: no chunks to index")
            return

        # 1) Объединяем мелкие чанки в крупные блоки + запоминаем диапазоны индексов
        t_merge = time.time()

        merge_size = settings.MERGE_SIZE
        large_chunks: List[str] = []
        ranges: List[tuple[int, int]] = []

        for start_idx in range(0, len(chunks), merge_size):
            end_idx = min(start_idx + merge_size - 1, len(chunks) - 1)
            merged_text = "\n\n".join(chunks[start_idx : end_idx + 1])
            large_chunks.append(merged_text)
            ranges.append((start_idx, end_idx))

        metrics["lightrag_large_chunks"] = len(large_chunks)
        metrics["times"]["lightrag_merge"] = time.time() - t_merge

        # 2) Вставляем в LightRAG и проставляем lightrag_node_id для соответствующих FileChunk
        if self.lightrag is None:
            raise RuntimeError("lightrag service is not configured")

        t_insert = time.time()

        for group_idx, (large_chunk, (start_idx, end_idx)) in enumerate(zip(large_chunks, ranges)):
            # пропускаем пустые блоки
            if not large_chunk or not large_chunk.strip():
                logger.debug(f"LightRAG: skip empty large_chunk group={group_idx}")
                continue

            try:
                node_id = self.lightrag.insert(large_chunk)  # ✅ ВАЖНО: передаём строку
            except RuntimeError as e:
                if "Use ainsert() inside async context" not in str(e):
                    raise

                file_path = stored_file.original_filename or f"{stored_file.scope.value}/{stored_file.id}"

                def _run() -> str:
                    import asyncio

                    return asyncio.run(
                        self.lightrag.ainsert(
                            text=large_chunk,
                            file_path=file_path,
                        )
                    )

                with ThreadPoolExecutor(max_workers=1) as ex:
                    node_id = ex.submit(_run).result()

            # ✅ Проставляем node_id всем мелким чанкам этой группы
            self.db.query(FileChunk).filter(
                FileChunk.file_id == stored_file.id,
                FileChunk.chunk_index >= start_idx,
                FileChunk.chunk_index <= end_idx,
            ).update({"lightrag_node_id": node_id}, synchronize_session=False)

            logger.debug(
                f"LightRAG insert: chunk_group={group_idx}, node_id={node_id}, range=({start_idx}-{end_idx})"
            )

        self.db.commit()

        metrics["times"]["lightrag_insert"] = time.time() - t_insert
        logger.info(f"Indexed to LightRAG: {len(large_chunks)} large chunks")


    def get_file(self, file_id: uuid.UUID) -> StoredFile | None:
        """Получить файл по ID."""
        return self.db.query(StoredFile).get(file_id)

    def list_admin_files(self, search: str | None = None) -> List[StoredFile]:
        """Список административных файлов."""
        query = self.db.query(StoredFile).filter(
            StoredFile.scope == FileScope.ADMIN_LAW
        )
        if search:
            like = f"%{search}%"
            query = query.filter(
                or_(
                    StoredFile.original_filename.ilike(like),
                    StoredFile.title.ilike(like),
                    StoredFile.description.ilike(like),
                    StoredFile.language.ilike(like),
                )
            )
        return query.order_by(StoredFile.uploaded_at.desc()).all()

    def list_customer_files(
        self, customer_id: str, owner_id: str | None = None
    ) -> List[StoredFile]:
        """Список файлов заказчика."""
        query = self.db.query(StoredFile).filter(
            StoredFile.scope == FileScope.CUSTOMER_DOC,
            StoredFile.customer_id == customer_id,
        )
        if owner_id:
            query = query.filter(StoredFile.owner_id == owner_id)
        return query.order_by(StoredFile.uploaded_at.desc()).all()