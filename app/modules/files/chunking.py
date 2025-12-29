# app/modules/files/chunking.py
"""
Section-based chunking strategy для Block B документов.

Поддерживает:
- Разбиение по маркерам ==== и ---- (section-based)
- Fallback на размерный chunking
- Сохранение метаданных секций
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Метаданные чанка для расширенного payload."""
    chunk_index: int
    section_title: Optional[str] = None
    section_level: int = 0  # 0=root, 1=====, 2=----
    block: str = "B"  # Block B по умолчанию
    isa_reference: List[str] = field(default_factory=list)
    cycle: Optional[str] = None
    char_start: int = 0
    char_end: int = 0


@dataclass
class Chunk:
    """Чанк документа с текстом и метаданными."""
    text: str
    metadata: ChunkMetadata


def chunk_by_section(
    content: str,
    chunk_size: Optional[int] = None,
    overlap: int = 100,
    min_chunk_size: int = 50,
) -> List[Chunk]:
    """
    Разбивает документ на чанки по секциям.
    
    Стратегия:
    1. Ищем маркеры секций (==== для level 1, ---- для level 2)
    2. Каждая секция становится отдельным чанком
    3. Если секция слишком большая — разбиваем по chunk_size
    4. Если секция слишком маленькая — объединяем с предыдущей
    
    Args:
        content: Текст документа
        chunk_size: Максимальный размер чанка (default: settings.CHUNK_SIZE)
        overlap: Перекрытие между чанками
        min_chunk_size: Минимальный размер чанка
        
    Returns:
        Список чанков с метаданными
    """
    if not content or not content.strip():
        return []
    
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    
    # Паттерны для секций
    # ==== или более = символов = level 1
    # ---- или более - символов = level 2
    section_pattern = re.compile(
        r'^(?P<marker>={4,}|-{4,})\s*$|^(?P<title>.+?)\s*\n(?P<underline>={4,}|-{4,})\s*$',
        re.MULTILINE
    )
    
    # Попробуем найти секции
    sections = _split_by_sections(content, section_pattern)
    
    if len(sections) <= 1:
        # Нет секций — используем размерный chunking
        logger.debug("No sections found, using size-based chunking")
        return _chunk_by_size(content, chunk_size, overlap, min_chunk_size)
    
    logger.info(f"Found {len(sections)} sections, processing section-based chunking")
    
    chunks: List[Chunk] = []
    chunk_index = 0
    
    for section in sections:
        section_text = section["text"].strip()
        if not section_text:
            continue
        
        # Если секция слишком большая — разбиваем
        if len(section_text) > chunk_size:
            sub_chunks = _chunk_by_size(
                section_text, 
                chunk_size, 
                overlap, 
                min_chunk_size,
                start_index=chunk_index,
                section_title=section.get("title"),
                section_level=section.get("level", 0),
            )
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)
        else:
            # Секция помещается в один чанк
            chunks.append(Chunk(
                text=section_text,
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    section_title=section.get("title"),
                    section_level=section.get("level", 0),
                    char_start=section.get("start", 0),
                    char_end=section.get("end", len(section_text)),
                ),
            ))
            chunk_index += 1
    
    # Объединяем маленькие чанки
    chunks = _merge_small_chunks(chunks, min_chunk_size, chunk_size)
    
    logger.info(f"Section-based chunking complete: {len(chunks)} chunks")
    return chunks


def _split_by_sections(content: str, pattern: re.Pattern) -> List[dict]:
    """Разбивает контент на секции по паттерну."""
    sections = []
    last_end = 0
    current_title = None
    current_level = 0
    
    # Ищем все совпадения
    for match in pattern.finditer(content):
        # Текст до маркера
        if match.start() > last_end:
            text_before = content[last_end:match.start()]
            if text_before.strip():
                sections.append({
                    "text": text_before,
                    "title": current_title,
                    "level": current_level,
                    "start": last_end,
                    "end": match.start(),
                })
        
        # Определяем тип маркера
        marker = match.group("marker") or match.group("underline") or ""
        if marker.startswith("="):
            current_level = 1
        elif marker.startswith("-"):
            current_level = 2
        
        # Заголовок секции
        current_title = match.group("title")
        last_end = match.end()
    
    # Последняя секция
    if last_end < len(content):
        remaining = content[last_end:]
        if remaining.strip():
            sections.append({
                "text": remaining,
                "title": current_title,
                "level": current_level,
                "start": last_end,
                "end": len(content),
            })
    
    return sections


def _chunk_by_size(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
    start_index: int = 0,
    section_title: Optional[str] = None,
    section_level: int = 0,
) -> List[Chunk]:
    """Размерный chunking с перекрытием."""
    chunks = []
    
    if not text:
        return chunks
    
    # Разбиваем по параграфам для более естественного разделения
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    current_start = 0
    char_pos = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Если параграф сам по себе больше chunk_size — разбиваем по предложениям
        if len(para) > chunk_size:
            # Сначала сохраняем накопленный текст
            if current_chunk:
                chunks.append(Chunk(
                    text=current_chunk,
                    metadata=ChunkMetadata(
                        chunk_index=start_index + len(chunks),
                        section_title=section_title,
                        section_level=section_level,
                        char_start=current_start,
                        char_end=char_pos,
                    ),
                ))
                current_chunk = ""
            
            # Разбиваем большой параграф
            sub_chunks = _split_large_paragraph(
                para, chunk_size, overlap,
                start_index + len(chunks),
                section_title, section_level,
                char_pos,
            )
            chunks.extend(sub_chunks)
            char_pos += len(para) + 2  # +2 для \n\n
            current_start = char_pos
            continue
        
        # Проверяем, поместится ли параграф
        test_chunk = current_chunk + ("\n\n" if current_chunk else "") + para
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Сохраняем текущий чанк
            if current_chunk:
                chunks.append(Chunk(
                    text=current_chunk,
                    metadata=ChunkMetadata(
                        chunk_index=start_index + len(chunks),
                        section_title=section_title,
                        section_level=section_level,
                        char_start=current_start,
                        char_end=char_pos,
                    ),
                ))
            
            # Начинаем новый с overlap
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
            current_start = char_pos
        
        char_pos += len(para) + 2
    
    # Последний чанк
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(Chunk(
            text=current_chunk,
            metadata=ChunkMetadata(
                chunk_index=start_index + len(chunks),
                section_title=section_title,
                section_level=section_level,
                char_start=current_start,
                char_end=char_pos,
            ),
        ))
    elif current_chunk and chunks:
        # Слишком маленький — добавляем к предыдущему
        chunks[-1] = Chunk(
            text=chunks[-1].text + "\n\n" + current_chunk,
            metadata=chunks[-1].metadata,
        )
    
    return chunks


def _split_large_paragraph(
    para: str,
    chunk_size: int,
    overlap: int,
    start_index: int,
    section_title: Optional[str],
    section_level: int,
    char_offset: int,
) -> List[Chunk]:
    """Разбивает большой параграф по предложениям."""
    # Разбиваем по предложениям
    sentences = re.split(r'(?<=[.!?])\s+', para)
    
    chunks = []
    current_chunk = ""
    current_start = char_offset
    
    for sentence in sentences:
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(Chunk(
                    text=current_chunk,
                    metadata=ChunkMetadata(
                        chunk_index=start_index + len(chunks),
                        section_title=section_title,
                        section_level=section_level,
                        char_start=current_start,
                        char_end=current_start + len(current_chunk),
                    ),
                ))
                current_start += len(current_chunk) + 1
            
            # Overlap
            if overlap > 0 and current_chunk:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(Chunk(
            text=current_chunk,
            metadata=ChunkMetadata(
                chunk_index=start_index + len(chunks),
                section_title=section_title,
                section_level=section_level,
                char_start=current_start,
                char_end=current_start + len(current_chunk),
            ),
        ))
    
    return chunks


def _merge_small_chunks(
    chunks: List[Chunk],
    min_size: int,
    max_size: int,
) -> List[Chunk]:
    """Объединяет слишком маленькие чанки."""
    if not chunks:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        # Если текущий чанк слишком маленький и можно объединить
        if len(current.text) < min_size:
            combined = current.text + "\n\n" + next_chunk.text
            if len(combined) <= max_size:
                current = Chunk(
                    text=combined,
                    metadata=ChunkMetadata(
                        chunk_index=current.metadata.chunk_index,
                        section_title=current.metadata.section_title or next_chunk.metadata.section_title,
                        section_level=min(current.metadata.section_level, next_chunk.metadata.section_level),
                        char_start=current.metadata.char_start,
                        char_end=next_chunk.metadata.char_end,
                    ),
                )
                continue
        
        merged.append(current)
        current = next_chunk
    
    merged.append(current)
    
    # Переиндексируем
    for i, chunk in enumerate(merged):
        chunk.metadata.chunk_index = i
    
    return merged


def chunk_text_simple(text: str, chunk_size: Optional[int] = None) -> List[str]:
    """
    Простой размерный chunking (для обратной совместимости).
    Возвращает только тексты без метаданных.
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE

    if not text:
        return []

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def extract_isa_references(text: str) -> List[str]:
    """
    Извлекает ссылки на ISA стандарты из текста.
    
    Примеры:
    - ISA 200
    - МСА 315
    - ISA 500.12
    """
    pattern = re.compile(
        r'\b(?:ISA|МСА|MSA)\s*(\d{3}(?:\.\d+)?)\b',
        re.IGNORECASE
    )
    
    matches = pattern.findall(text)
    return list(set(f"ISA {m}" for m in matches))


def detect_audit_cycle(text: str) -> Optional[str]:
    """
    Определяет цикл аудита из текста.
    
    Возможные значения: acceptance, planning, execution, reporting, completion
    """
    text_lower = text.lower()
    
    cycle_keywords = {
        "acceptance": ["принятие", "acceptance", "engagement", "договор", "клиент"],
        "planning": ["планирование", "planning", "план", "риск", "существенность"],
        "execution": ["выполнение", "execution", "тестирование", "процедуры", "доказательства"],
        "reporting": ["отчётность", "reporting", "заключение", "мнение", "отчёт"],
        "completion": ["завершение", "completion", "события", "письмо руководству"],
    }
    
    scores = {}
    for cycle, keywords in cycle_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cycle] = score
    
    if scores:
        return max(scores, key=scores.get)
    return None
