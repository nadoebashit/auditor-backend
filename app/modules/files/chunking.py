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
    industry_code: Optional[str] = None
    lang: Optional[str] = None
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
    # ──── или более (unicode box drawing) = level 1 (часто в legal matrices)
    section_pattern = re.compile(
        r'^(?P<marker>={4,}|-{4,}|─{4,})\s*$|^(?P<title>.+?)\s*\n(?P<underline>={4,}|-{4,}|─{4,})\s*$',
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


def chunk_by_node_marker(
    content: str,
    *,
    marker_pattern: str = r"^\s*NODE\s+\w+",
    min_chunk_size: int = 20,
) -> List[Chunk]:
    if not content or not content.strip():
        return []

    pat = re.compile(marker_pattern, re.MULTILINE)
    matches = list(pat.finditer(content))
    if not matches:
        return _chunk_by_size(
            content,
            chunk_size=settings.CHUNK_SIZE,
            overlap=0,
            min_chunk_size=min_chunk_size,
        )

    out: List[Chunk] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        text = (content[start:end] or "").strip()
        if not text:
            continue
        first_line = (text.splitlines()[0] if text.splitlines() else "").strip()
        out.append(
            Chunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_index=len(out),
                    section_title=first_line or None,
                    section_level=1,
                    char_start=int(start),
                    char_end=int(end),
                ),
            )
        )
    return out


def chunk_f2_industry_pack(content: str) -> List[Chunk]:
    if not content or not content.strip():
        return []

    pat = re.compile(r"^\s*INDUSTRY:\s*$", re.MULTILINE)
    markers = list(pat.finditer(content))
    if not markers:
        return _chunk_by_size(
            content,
            chunk_size=512,
            overlap=50,
            min_chunk_size=20,
        )

    out: List[Chunk] = []
    for idx, m in enumerate(markers):
        start = m.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(content)
        block = (content[start:end] or "").strip()
        if not block:
            continue

        m_code = re.search(r"^\s*code:\s*([A-Za-z0-9_-]+)\s*$", block, flags=re.MULTILINE)
        code = (m_code.group(1).strip().upper() if m_code else None)
        title = f"INDUSTRY {code}" if code else "INDUSTRY"

        out.append(
            Chunk(
                text=block,
                metadata=ChunkMetadata(
                    chunk_index=len(out),
                    section_title=title,
                    section_level=1,
                    industry_code=code,
                    lang="EN",
                    char_start=int(start),
                    char_end=int(end),
                ),
            )
        )
    return out


def chunk_f1_company_profile(content: str) -> List[Chunk]:
    if not content or not content.strip():
        return []

    para_pat = re.compile(
        r"(?:^|\n\s*\n)(?P<para>.*?)(?=\n\s*\n|$)",
        flags=re.DOTALL,
    )

    current_section: Optional[str] = None
    out: List[Chunk] = []

    for m in para_pat.finditer(content):
        para = (m.group("para") or "").strip("\n")
        start = m.start("para")
        end = m.end("para")

        text = (para or "").strip()
        if not text:
            continue

        if ("\n" not in text) and (":" not in text) and len(text) <= 80:
            current_section = text
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        for i in range(0, len(lines), 4):
            group = "\n".join(lines[i : i + 4]).strip()
            if not group:
                continue

            lang: Optional[str] = None
            if ("EN:" in group) and ("RU:" not in group):
                lang = "EN"
            elif ("RU:" in group) and ("EN:" not in group):
                lang = "RU"

            out.append(
                Chunk(
                    text=group,
                    metadata=ChunkMetadata(
                        chunk_index=len(out),
                        section_title=current_section,
                        section_level=1 if current_section else 0,
                        lang=lang,
                        char_start=int(start),
                        char_end=int(end),
                    ),
                )
            )

    return out


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
        elif marker.startswith("─"):
            current_level = 1
        
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


def extract_ifrs_references(text: str) -> List[str]:
    pattern = re.compile(
        r"\b(?:IAS|IFRS)\s*(\d{1,3}(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    matches = pattern.findall(text)
    out: set[str] = set()
    for m in matches:
        # Keep original standard prefix when possible
        try:
            prefix_match = re.search(rf"\b(IAS|IFRS)\s*{re.escape(m)}\b", text, flags=re.IGNORECASE)
            prefix = (prefix_match.group(1) if prefix_match else "IAS").upper() if prefix_match else "IAS"
        except Exception:
            prefix = "IAS"
        out.add(f"{prefix} {m}")
    return list(out)


def chunk_excel_text(
    content: str,
    chunk_size: Optional[int] = None,
    rows_per_chunk: int = 50,
) -> List[Chunk]:
    """
    Специализированный chunking для текста, извлечённого из Excel.
    
    Распознаёт формат:
    - SHEET: <название листа>
    - строки с табуляцией (данные таблицы)
    
    Разбивает по листам и группам строк, сохраняя заголовки таблиц.
    """
    if not content or not content.strip():
        return []
    
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    
    chunks: List[Chunk] = []
    chunk_index = 0

    def _clean_excel_row(line: str) -> str:
        if not line:
            return ""
        parts = [p.strip() for p in line.split("\t")]
        cleaned: list[str] = []
        for p in parts:
            if p in {"#REF!", "#DIV/0!", "#N/A", "#VALUE!", "#NAME?", "#NULL!", "#NUM!"}:
                cleaned.append("EXCEL_ERROR")
            else:
                cleaned.append(p)

        while cleaned and (cleaned[-1] == "" or cleaned[-1] in {"0", "0.0"}):
            cleaned.pop()

        return "\t".join(cleaned).strip()

    def _has_nonzero_data(line: str) -> bool:
        if not line:
            return False
        parts = [p.strip() for p in line.split("\t") if p is not None]
        for p in parts:
            if not p:
                continue
            if p in {"0", "0.0"}:
                continue
            return True
        return False

    def _is_numeric(token: str) -> bool:
        if not token:
            return False
        try:
            float(token.replace(" ", ""))
            return True
        except Exception:
            return False
    
    # Разбиваем по листам
    sheet_pattern = re.compile(r'^SHEET:\s*(.+)$', re.MULTILINE)
    sheet_matches = list(sheet_pattern.finditer(content))
    
    if not sheet_matches:
        # Нет маркеров листов — используем размерный chunking
        return _chunk_by_size(content, chunk_size, overlap=50, min_chunk_size=30)
    
    for i, match in enumerate(sheet_matches):
        sheet_name = match.group(1).strip()
        start_pos = match.end()
        end_pos = sheet_matches[i + 1].start() if i + 1 < len(sheet_matches) else len(content)
        
        sheet_content = content[start_pos:end_pos].strip()
        if not sheet_content:
            # Still create a minimal chunk so the file can be indexed and surfaced in retrieval.
            chunk_text = f"SHEET: {sheet_name}"
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        section_title=sheet_name,
                        section_level=1,
                        char_start=match.start(),
                        char_end=end_pos,
                    ),
                )
            )
            chunk_index += 1
            continue
        
        lines = sheet_content.split('\n')
        
        # Первая непустая строка — заголовок таблицы
        header_line = None
        data_lines = []
        for line in lines:
            cleaned = _clean_excel_row(line)
            if not _has_nonzero_data(cleaned):
                continue
            if header_line is None:
                first_cell = (cleaned.split("\t")[0].strip() if cleaned else "")
                if first_cell and (not _is_numeric(first_cell)):
                    header_line = cleaned
                else:
                    data_lines.append(cleaned)
            else:
                data_lines.append(cleaned)
        
        if not data_lines:
            # Только заголовок или пусто
            if header_line:
                chunk_text = f"SHEET: {sheet_name}\n{header_line}"
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        section_title=sheet_name,
                        section_level=1,
                        char_start=match.start(),
                        char_end=end_pos,
                    )
                ))
                chunk_index += 1
            continue
        
        # Разбиваем данные на группы по rows_per_chunk
        for batch_start in range(0, len(data_lines), rows_per_chunk):
            batch_end = min(batch_start + rows_per_chunk, len(data_lines))
            batch_lines = data_lines[batch_start:batch_end]
            
            # Каждый чанк включает заголовок для контекста
            chunk_text = f"SHEET: {sheet_name}\n"
            if header_line:
                chunk_text += f"{header_line}\n"
            chunk_text += '\n'.join(batch_lines)
            
            # Проверяем размер
            if len(chunk_text) > chunk_size:
                # Слишком большой — разбиваем дополнительно
                sub_chunks = _chunk_by_size(chunk_text, chunk_size, overlap=50, min_chunk_size=30)
                for sub in sub_chunks:
                    sub.metadata.chunk_index = chunk_index
                    sub.metadata.section_title = sheet_name
                    chunks.append(sub)
                    chunk_index += 1
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        section_title=sheet_name,
                        section_level=1,
                        char_start=match.start(),
                        char_end=end_pos,
                    )
                ))
                chunk_index += 1
    
    logger.info(f"Excel chunking: {len(chunks)} chunks from {len(sheet_matches)} sheets")
    return chunks


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
