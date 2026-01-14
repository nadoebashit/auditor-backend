"""
Sparse Search (BM25) implementation using PostgreSQL Full-Text Search.

Features:
- PostgreSQL tsvector/tsquery for keyword search
- Multi-language support (Russian, English)
- Hybrid search combining with dense vectors
- Efficient indexing with GIN indexes
"""

import logging
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from sqlalchemy import text, func
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.modules.files.models import FileChunk, StoredFile, FileScope

logger = get_logger(__name__)

_FTS_WARMUP_LOCK = threading.Lock()
_FTS_WARMED_UP = False


@dataclass
class SparseSearchResult:
    """Result from sparse search."""
    chunk_id: str
    file_id: str
    chunk_index: int
    text: str
    score: float
    scope: str
    customer_id: Optional[str]
    owner_id: str
    filename: Optional[str]
    highlights: List[str]


class PostgresFTSSearch:
    """
    PostgreSQL Full-Text Search for sparse/keyword retrieval.
    
    Uses tsvector for indexing and tsquery for searching.
    Supports Russian and English languages.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._ensure_fts_setup()
        self._maybe_warm_search_vectors()
    
    def _ensure_fts_setup(self):
        """
        Ensure FTS columns and indexes exist.
        
        Note: This should be done via migrations in production.
        """
        try:
            # Check if tsvector column exists
            result = self.db.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'file_chunks' AND column_name = 'search_vector'
            """))
            
            if not result.fetchone():
                logger.info("Creating FTS column and index for file_chunks")
                
                # Add tsvector column
                self.db.execute(text("""
                    ALTER TABLE file_chunks 
                    ADD COLUMN IF NOT EXISTS search_vector tsvector
                """))
                
                # Create GIN index
                self.db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_file_chunks_search_vector 
                    ON file_chunks USING GIN(search_vector)
                """))
                
                # Create trigger to auto-update tsvector
                self.db.execute(text("""
                    CREATE OR REPLACE FUNCTION file_chunks_search_vector_update() 
                    RETURNS trigger AS $$
                    BEGIN
                        NEW.search_vector := 
                            setweight(to_tsvector('russian', COALESCE(NEW.text, '')), 'A') ||
                            setweight(to_tsvector('english', COALESCE(NEW.text, '')), 'B');
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """))
                
                self.db.execute(text("""
                    DROP TRIGGER IF EXISTS file_chunks_search_vector_trigger ON file_chunks;
                    CREATE TRIGGER file_chunks_search_vector_trigger
                    BEFORE INSERT OR UPDATE ON file_chunks
                    FOR EACH ROW EXECUTE FUNCTION file_chunks_search_vector_update();
                """))
                
                self.db.commit()
                logger.info("FTS setup completed")
            else:
                logger.debug("FTS column already exists")
                
        except Exception as e:
            logger.error(f"Failed to setup FTS: {e}")
            self.db.rollback()
    
    def update_search_vectors(self, batch_size: int = 1000) -> int:
        """
        Update search vectors for existing chunks.
        
        Returns:
            Number of updated chunks
        """
        try:
            result = self.db.execute(text("""
                WITH to_update AS (
                    SELECT id
                    FROM file_chunks
                    WHERE search_vector IS NULL
                    LIMIT :batch_size
                )
                UPDATE file_chunks fc
                SET search_vector =
                    setweight(to_tsvector('russian', COALESCE(fc.text, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(fc.text, '')), 'B')
                FROM to_update u
                WHERE fc.id = u.id
            """), {"batch_size": batch_size})
            
            self.db.commit()
            updated = result.rowcount
            logger.info(f"Updated {updated} search vectors")
            return updated
            
        except Exception as e:
            logger.error(f"Failed to update search vectors: {e}")
            self.db.rollback()
            return 0

    def _maybe_warm_search_vectors(self) -> None:
        global _FTS_WARMED_UP

        if _FTS_WARMED_UP:
            return

        with _FTS_WARMUP_LOCK:
            if _FTS_WARMED_UP:
                return

            try:
                has_null = self.db.execute(
                    text("SELECT 1 FROM file_chunks WHERE search_vector IS NULL LIMIT 1")
                ).fetchone()
                if not has_null:
                    _FTS_WARMED_UP = True
                    return

                total_updated = 0
                for _ in range(3):
                    updated = int(self.update_search_vectors(batch_size=5000) or 0)
                    total_updated += updated
                    if updated <= 0:
                        break

                logger.info(
                    "FTS warmup completed",
                    extra={"updated": int(total_updated)},
                )
                _FTS_WARMED_UP = True
            except Exception as e:
                logger.warning(f"FTS warmup failed: {e}")
                try:
                    self.db.rollback()
                except Exception:
                    pass
    
    def search(
        self,
        query: str,
        scope: Optional[str] = None,
        customer_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        limit: int = 20,
        language: str = "russian",
    ) -> List[SparseSearchResult]:
        """
        Perform full-text search on file chunks.
        
        Args:
            query: Search query
            scope: Filter by scope (ADMIN_LAW, CUSTOMER_DOC)
            customer_id: Filter by customer
            owner_id: Filter by owner
            limit: Max results
            language: Search language (russian, english)
            
        Returns:
            List of SparseSearchResult
        """
        query_normalized = self._normalize_query(query)
        if not query_normalized:
            return []

        lang = (language or "russian").strip().lower()
        if lang not in {"russian", "english"}:
            lang = "russian"

        query_normalized_en = self._normalize_query_english(query)

        tsquery_ru = "to_tsquery('russian', :query_normalized)"
        if query_normalized_en:
            tsquery_en = "to_tsquery('english', :query_normalized_en)"
            match_expr = f"(fc.search_vector @@ {tsquery_ru} OR fc.search_vector @@ {tsquery_en})"
            score_expr = f"GREATEST(ts_rank_cd(fc.search_vector, {tsquery_ru}), ts_rank_cd(fc.search_vector, {tsquery_en}))"
        else:
            match_expr = f"(fc.search_vector @@ {tsquery_ru})"
            score_expr = f"ts_rank_cd(fc.search_vector, {tsquery_ru})"

        # Build WHERE conditions
        conditions = [match_expr]
        params = {"limit": limit}
        
        if scope:
            conditions.append("sf.scope = :scope")
            params["scope"] = scope
        
        if customer_id:
            conditions.append("sf.customer_id = :customer_id")
            params["customer_id"] = customer_id
        
        if owner_id:
            conditions.append("sf.owner_id = :owner_id")
            params["owner_id"] = owner_id
        
        where_clause = " AND ".join(conditions)
        
        # Execute search with ranking
        sql = text(f"""
            SELECT 
                fc.id as chunk_id,
                fc.file_id,
                fc.chunk_index,
                fc.text,
                {score_expr} as score,
                sf.scope,
                sf.customer_id,
                sf.owner_id::text,
                sf.original_filename,
                ts_headline('{lang}', fc.text, to_tsquery('{lang}', :query_normalized), 
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20') as highlights
            FROM file_chunks fc
            JOIN stored_files sf ON fc.file_id = sf.id
            WHERE {where_clause}
            ORDER BY score DESC
            LIMIT :limit
        """)
        
        params["query_normalized"] = query_normalized
        if query_normalized_en:
            params["query_normalized_en"] = query_normalized_en
        
        try:
            result = self.db.execute(sql, params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                results.append(SparseSearchResult(
                    chunk_id=str(row.chunk_id),
                    file_id=str(row.file_id),
                    chunk_index=row.chunk_index,
                    text=row.text[:500] if row.text else "",
                    score=float(row.score) if row.score else 0.0,
                    scope=row.scope,
                    customer_id=row.customer_id,
                    owner_id=row.owner_id,
                    filename=row.original_filename,
                    highlights=[row.highlights] if row.highlights else [],
                ))
            
            logger.info(f"FTS search found {len(results)} results for query: {query[:50]}")
            return results
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass
            return []
    
    def _build_tsquery(self, query: str, language: str) -> str:
        """Build tsquery string from user query."""
        # Simple tokenization and OR combination
        words = query.split()
        if not words:
            return ""
        
        # Use plainto_tsquery for simple queries
        return f"plainto_tsquery('{language}', '{query}')"
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for tsquery."""
        # Remove special characters, keep alphanumeric and spaces
        import re
        normalized = re.sub(r'[^\w\s]', ' ', query)
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        words = [w for w in normalized.split() if w]
        stop = {
            "перечисли",
            "перечислить",
            "все",
            "всё",
            "какие",
            "какой",
            "какая",
            "какое",
            "есть",
            "включая",
            "включить",
            "договор",
            "договора",
            "контракт",
            "соглашение",
        }
        filtered: list[str] = []
        for w in words:
            wl = w.lower()
            if wl in stop:
                continue
            if len(wl) < 3 and not wl.isdigit():
                continue
            if wl.isdigit():
                filtered.append(wl)
            elif len(wl) >= 4:
                filtered.append(f"{wl}:*")
            else:
                filtered.append(wl)
            if len(filtered) >= 12:
                break
        if not filtered:
            return ""
        if len(filtered) == 1:
            return filtered[0]
        return ' | '.join(filtered)

    def _normalize_query_english(self, query: str) -> str:
        import re

        normalized = re.sub(r"[^A-Za-z0-9\s]", " ", query)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        if not normalized:
            return ""

        words = [w for w in normalized.split() if w]
        filtered: list[str] = []
        for w in words:
            if len(w) < 3 and not w.isdigit():
                continue
            if w.isdigit():
                filtered.append(w)
            elif len(w) >= 4:
                filtered.append(f"{w}:*")
            else:
                filtered.append(w)
            if len(filtered) >= 12:
                break

        if not filtered:
            return ""
        if len(filtered) == 1:
            return filtered[0]
        return " | ".join(filtered)


class HybridSearch:
    """
    Hybrid search combining dense (Qdrant) and sparse (PostgreSQL FTS) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) for score combination.
    """
    
    def __init__(
        self,
        db: Session,
        qdrant_store: Any,
        embedding_service: Any,
    ):
        self.db = db
        self.qdrant_store = qdrant_store
        self.embedding_service = embedding_service
        self.fts_search = PostgresFTSSearch(db)
    
    def search(
        self,
        query: str,
        scope: Optional[str] = None,
        customer_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            scope: Filter by scope
            customer_id: Filter by customer
            owner_id: Filter by owner
            limit: Max results
            dense_weight: Weight for dense (vector) results
            sparse_weight: Weight for sparse (FTS) results
            
        Returns:
            Combined and ranked results
        """
        # 1. Dense search (Qdrant)
        dense_results = self._dense_search(
            query=query,
            scope=scope,
            customer_id=customer_id,
            owner_id=owner_id,
            limit=limit * 2,  # Get more for fusion
        )
        
        # 2. Sparse search (PostgreSQL FTS)
        sparse_results = self.fts_search.search(
            query=query,
            scope=scope,
            customer_id=customer_id,
            owner_id=owner_id,
            limit=limit * 2,
        )
        
        # 3. Reciprocal Rank Fusion
        combined = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        
        # 4. Return top results
        return combined[:limit]
    
    def _dense_search(
        self,
        query: str,
        scope: Optional[str],
        customer_id: Optional[str],
        owner_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        try:
            # Get query embedding
            query_vector = self.embedding_service.embed_single(query)

            target_size = getattr(self.qdrant_store, "vector_size", None)
            if isinstance(target_size, int) and target_size > 0:
                if len(query_vector) > target_size:
                    query_vector = query_vector[:target_size]
                elif len(query_vector) < target_size:
                    query_vector = query_vector + [0.0] * (target_size - len(query_vector))
            
            # Build filter
            filter_ = self.qdrant_store.build_filter(
                scope=scope,
                customer_id=customer_id,
                owner_id=owner_id,
            )
            
            # Search
            points = self.qdrant_store.search(
                query_vector=query_vector,
                limit=limit,
                filter_=filter_,
            )
            
            results = []
            for i, point in enumerate(points):
                payload = point.payload or {}
                results.append({
                    "id": f"{payload.get('file_id')}_{payload.get('chunk_index')}",
                    "file_id": payload.get("file_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "score": point.score,
                    "rank": i + 1,
                    "source": "dense",
                    "scope": payload.get("scope"),
                    "customer_id": payload.get("customer_id"),
                    "owner_id": payload.get("owner_id"),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[SparseSearchResult],
        dense_weight: float,
        sparse_weight: float,
        k: int = 60,  # RRF constant
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank))
        """
        scores = {}
        result_data = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result["id"]
            rank = result["rank"]
            rrf_score = dense_weight * (1.0 / (k + rank))
            
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            result_data[doc_id] = {
                **result,
                "dense_rank": rank,
                "dense_score": result["score"],
            }
        
        # Process sparse results
        for i, result in enumerate(sparse_results):
            doc_id = f"{result.file_id}_{result.chunk_index}"
            rank = i + 1
            rrf_score = sparse_weight * (1.0 / (k + rank))
            
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            
            if doc_id in result_data:
                result_data[doc_id]["sparse_rank"] = rank
                result_data[doc_id]["sparse_score"] = result.score
                result_data[doc_id]["highlights"] = result.highlights
            else:
                result_data[doc_id] = {
                    "id": doc_id,
                    "file_id": result.file_id,
                    "chunk_index": result.chunk_index,
                    "text": result.text,
                    "scope": result.scope,
                    "customer_id": result.customer_id,
                    "owner_id": result.owner_id,
                    "filename": result.filename,
                    "sparse_rank": rank,
                    "sparse_score": result.score,
                    "highlights": result.highlights,
                    "source": "sparse",
                }
        
        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build final results
        combined = []
        for doc_id in sorted_ids:
            data = result_data[doc_id]
            data["rrf_score"] = scores[doc_id]
            data["source"] = "hybrid" if "dense_rank" in data and "sparse_rank" in data else data.get("source", "unknown")
            combined.append(data)
        
        return combined


def create_hybrid_search(
    db: Session,
    qdrant_store: Any,
    embedding_service: Any = None,
) -> HybridSearch:
    """Factory function for HybridSearch."""
    from app.modules.embeddings.service import get_embedding_service
    
    if embedding_service is None:
        embedding_service = get_embedding_service()
    
    return HybridSearch(
        db=db,
        qdrant_store=qdrant_store,
        embedding_service=embedding_service,
    )
