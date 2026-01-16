from __future__ import annotations

import io
import logging
import re
from typing import Final

import requests
from docx import Document
from pypdf import PdfReader

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Разрешённые / ожидаемые content-type
DOCX_MIME_TYPES: Final[set[str]] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
PDF_MIME_TYPES: Final[set[str]] = {
    "application/pdf",
}
TEXT_MIME_TYPES: Final[set[str]] = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",  # только если вы уверены, что хотите HTML как сырой текст
}
EXCEL_MIME_TYPES: Final[set[str]] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def _normalize_text(text: str) -> str:
    """
    Нормализует текст перед сохранением в БД:
    - удаляет NUL (\x00), которые Postgres не принимает
    - убирает управляющие символы кроме табуляции и перевода строки
    - нормализует переводы строк и лишние пробелы
    """
    if not text:
        return ""

    # Удаляем NUL-символы
    text = text.replace("\x00", "")

    # Удаляем прочие непечатаемые управляющие символы (кроме \n, \r, \t)
    control_chars_regex = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"
    text = re.sub(control_chars_regex, "", text)

    # Нормализуем переводы строк (CRLF -> LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Сжимаем более трёх переводов строки подряд до двух
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Удаляем лишние пробелы в начале и в конце строк
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def extract_text_from_docx(file_bytes: bytes, filename: str | None = None) -> str:
    """
    Извлекает текст из DOCX файла с помощью python-docx.
    """
    logger.info("Extracting text from DOCX file", extra={"source_filename": filename})

    with io.BytesIO(file_bytes) as buffer:
        document = Document(buffer)

    paragraphs: list[str] = []

    # Основные абзацы
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Таблицы (если в документе есть важный текст в таблицах)
    for table in document.tables:
        for row in table.rows:
            cells_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells_text:
                paragraphs.append(" | ".join(cells_text))

    raw_text = "\n\n".join(paragraphs)
    normalized = _normalize_text(raw_text)

    logger.info(
        "DOCX text extracted",
        extra={
            "source_filename": filename,
            "chars_raw": len(raw_text),
            "chars_normalized": len(normalized),
        },
    )

    return normalized


def extract_text_from_pdf(file_bytes: bytes, filename: str | None = None) -> str:
    """
    Извлекает текст из PDF с помощью pypdf.
    """
    logger.info("Extracting text from PDF file", extra={"source_filename": filename})

    with io.BytesIO(file_bytes) as buffer:
        reader = PdfReader(buffer)

        pages_text: list[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from PDF page",
                    extra={
                        "source_filename": filename,
                        "page_index": page_index,
                        "error": str(exc),
                    },
                )
                page_text = ""
            page_text = page_text.strip()
            if page_text:
                pages_text.append(page_text)

    raw_text = "\n\n".join(pages_text)
    normalized = _normalize_text(raw_text)

    if (
        getattr(settings, "AZURE_OCR_ENABLED", False)
        and isinstance(normalized, str)
        and len(normalized) < int(getattr(settings, "AZURE_OCR_MIN_TEXT_CHARS", 200) or 200)
    ):
        endpoint = getattr(settings, "AZURE_OCR_ENDPOINT", None)
        api_key = getattr(settings, "AZURE_OCR_API_KEY", None)
        if endpoint and api_key:
            try:
                timeout_s = int(getattr(settings, "AZURE_OCR_TIMEOUT_S", 120) or 120)
                headers = {"api-key": str(api_key)}
                files = {
                    "file": (
                        (filename or "document.pdf"),
                        file_bytes,
                        "application/pdf",
                    )
                }
                resp = requests.post(
                    str(endpoint),
                    headers=headers,
                    files=files,
                    timeout=timeout_s,
                    verify=(
                        str(getattr(settings, "REQUESTS_CA_BUNDLE", "") or "").strip()
                        if bool(getattr(settings, "REQUESTS_VERIFY_SSL", True))
                        and str(getattr(settings, "REQUESTS_CA_BUNDLE", "") or "").strip()
                        else bool(getattr(settings, "REQUESTS_VERIFY_SSL", True))
                    ),
                )
                resp.raise_for_status()

                data = resp.json()
                ocr_text = ""
                if isinstance(data, dict):
                    if isinstance(data.get("text"), str):
                        ocr_text = data.get("text") or ""
                    elif isinstance(data.get("content"), str):
                        ocr_text = data.get("content") or ""
                    elif isinstance(data.get("pages"), list):
                        parts: list[str] = []
                        for p in data.get("pages") or []:
                            if not isinstance(p, dict):
                                continue
                            for k in ["text", "content", "markdown"]:
                                v = p.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                                    break
                        ocr_text = "\n\n".join(parts)

                ocr_text = _normalize_text(ocr_text)
                if ocr_text:
                    logger.info(
                        "PDF OCR extracted",
                        extra={
                            "source_filename": filename,
                            "chars_ocr": len(ocr_text),
                            "chars_before_ocr": len(normalized),
                        },
                    )
                    return ocr_text
            except Exception as exc:
                logger.warning(
                    "PDF OCR failed; falling back to pypdf text",
                    extra={
                        "source_filename": filename,
                        "error": str(exc),
                    },
                )

    logger.info(
        "PDF text extracted",
        extra={
            "source_filename": filename,
            "pages": len(reader.pages),
            "chars_raw": len(raw_text),
            "chars_normalized": len(normalized),
        },
    )

    return normalized


def extract_text_from_plain(
    file_bytes: bytes,
    filename: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Извлекает текст из простого текстового файла.
    """
    logger.info(
        "Extracting text from plain text file",
        extra={"source_filename": filename, "encoding": encoding},
    )

    try:
        text = file_bytes.decode(encoding)
    except Exception as exc:
        logger.warning(
            "Failed to decode text file with encoding, using errors='ignore'",
            extra={"source_filename": filename, "encoding": encoding, "error": str(exc)},
        )
        text = file_bytes.decode(encoding, errors="ignore")

    normalized = _normalize_text(text)

    logger.info(
        "Plain text extracted",
        extra={
            "source_filename": filename,
            "chars_raw": len(text),
            "chars_normalized": len(normalized),
        },
    )

    return normalized


def extract_text_from_xls(file_bytes: bytes, filename: str | None = None) -> str:
    try:
        import xlrd  # type: ignore
    except ImportError as exc:
        logger.error(
            "xlrd is not available for XLS extraction",
            extra={"source_filename": filename, "error": str(exc)},
        )
        return ""

    logger.info("Extracting text from XLS file (xlrd)", extra={"source_filename": filename})

    max_sheets = 20
    max_rows = 300
    max_cols = 60
    max_cell_chars = 200

    try:
        wb = xlrd.open_workbook(file_contents=file_bytes)
    except Exception as exc:
        logger.error(
            "Failed to open XLS workbook with xlrd",
            extra={"source_filename": filename, "error": str(exc)},
        )
        return ""

    parts: list[str] = []
    try:
        for sheet_idx in range(min(wb.nsheets, max_sheets)):
            ws = wb.sheet_by_index(sheet_idx)
            parts.append(f"SHEET: {ws.name}")

            for row_idx in range(min(ws.nrows, max_rows)):
                values: list[str] = []
                for col_idx in range(min(ws.ncols, max_cols)):
                    try:
                        cell = ws.cell_value(row_idx, col_idx)
                        s = str(cell) if cell is not None else ""
                        if len(s) > max_cell_chars:
                            s = s[:max_cell_chars]
                        values.append(s)
                    except Exception:
                        values.append("")

                if any(v.strip() for v in values):
                    parts.append("\t".join(values).rstrip())
    except Exception as proc_exc:
        logger.error(
            "Error processing XLS workbook",
            extra={"source_filename": filename, "error": str(proc_exc)},
        )

    raw_text = "\n".join(parts)
    normalized = _normalize_text(raw_text)

    logger.info(
        "XLS text extracted",
        extra={
            "source_filename": filename,
            "chars_raw": len(raw_text),
            "chars_normalized": len(normalized),
        },
    )

    return normalized


def extract_text_from_xlsx(file_bytes: bytes, filename: str | None = None) -> str:
    if file_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        logger.info(
            "Detected old XLS format (OLE2), redirecting to xlrd",
            extra={"source_filename": filename},
        )
        return extract_text_from_xls(file_bytes, filename=filename)

    try:
        from openpyxl import load_workbook  # type: ignore
    except ImportError as exc:
        logger.error(
            "openpyxl is not available for XLSX extraction",
            extra={"source_filename": filename, "error": str(exc)},
        )
        return ""

    logger.info("Extracting text from XLSX file", extra={"source_filename": filename})

    max_sheets = 20
    max_rows = 300
    max_cols = 60
    max_cell_chars = 200

    def _extract_from_workbook(wb: Any, read_only: bool) -> tuple[list[str], bool, bool]:
        parts: list[str] = []
        had_row_errors = False
        had_data_rows = False

        sheetnames = list(getattr(wb, "sheetnames", []) or [])
        logger.debug(
            "XLSX sheets found",
            extra={
                "source_filename": filename,
                "sheets": sheetnames[:5],
                "total_sheets": len(sheetnames),
                "read_only": bool(read_only),
            },
        )

        for sheet_name in sheetnames[:max_sheets]:
            try:
                ws = wb[sheet_name]
            except Exception as sheet_exc:
                logger.warning(
                    "Failed to access sheet",
                    extra={"source_filename": filename, "sheet": sheet_name, "error": str(sheet_exc)},
                )
                continue

            parts.append(f"SHEET: {sheet_name}")

            row_count = 0
            try:
                for row in ws.iter_rows(
                    min_row=1,
                    max_row=max_rows,
                    min_col=1,
                    max_col=max_cols,
                    values_only=True,
                ):
                    if row_count >= max_rows:
                        break

                    values: list[str] = []
                    for cell in (row or ()):
                        if cell is None:
                            values.append("")
                            continue
                        s = str(cell)
                        if len(s) > max_cell_chars:
                            s = s[:max_cell_chars]
                        values.append(s)

                    if any(v.strip() for v in values):
                        parts.append("\t".join(values).rstrip())
                        row_count += 1
                        had_data_rows = True
            except Exception as row_exc:
                had_row_errors = True
                logger.exception(
                    "Error reading rows from sheet",
                    extra={
                        "source_filename": filename,
                        "sheet": sheet_name,
                        "error": str(row_exc),
                        "read_only": bool(read_only),
                    },
                )
                continue

        return parts, had_row_errors, had_data_rows

    wb = None
    try:
        with io.BytesIO(file_bytes) as buffer:
            wb = load_workbook(buffer, read_only=True, data_only=True)
        parts, had_row_errors, had_data_rows = _extract_from_workbook(wb, read_only=True)

        if (not had_data_rows) and had_row_errors:
            try:
                try:
                    wb.close()
                except Exception:
                    pass
                with io.BytesIO(file_bytes) as buffer:
                    wb = load_workbook(buffer, read_only=False, data_only=True)
                parts2, had_row_errors2, had_data_rows2 = _extract_from_workbook(wb, read_only=False)
                if had_data_rows2 or (len(parts2) > len(parts)):
                    parts = parts2
                    had_row_errors = had_row_errors2
                    had_data_rows = had_data_rows2
            except Exception as retry_exc:
                logger.exception(
                    "Failed to retry XLSX extraction with read_only=False",
                    extra={"source_filename": filename, "error": str(retry_exc)},
                )
    except Exception as exc:
        logger.exception(
            "Failed to open XLSX workbook",
            extra={"source_filename": filename, "error": str(exc)},
        )
        return ""
    finally:
        if wb is not None:
            try:
                wb.close()
            except Exception:
                pass

    raw_text = "\n".join(parts)
    normalized = _normalize_text(raw_text)

    logger.info(
        "XLSX text extracted",
        extra={
            "source_filename": filename,
            "chars_raw": len(raw_text),
            "chars_normalized": len(normalized),
        },
    )

    return normalized


def extract_text(
    file_bytes: bytes, content_type: str | None, filename: str | None
) -> str:
    """
    Высокоуровневая функция определения формата и извлечения текста.
    """
    content_type = (content_type or "").lower()
    filename = (filename or "").lower()

    logger.info(
        "Starting text extraction",
        extra={"source_filename": filename, "content_type": content_type},
    )

    try:
        # DOCX
        if content_type in DOCX_MIME_TYPES or (
            filename and (filename.endswith(".docx") or filename.endswith(".doc"))
        ):
            return extract_text_from_docx(file_bytes, filename=filename)

        # PDF
        if content_type in PDF_MIME_TYPES or (filename and filename.endswith(".pdf")):
            return extract_text_from_pdf(file_bytes, filename=filename)

        # XLSX/XLS (должен быть ДО TXT fallback, иначе binary Excel попадёт в plain text)
        if content_type in EXCEL_MIME_TYPES or (
            filename and (filename.endswith(".xlsx") or filename.endswith(".xls"))
        ):
            return extract_text_from_xlsx(file_bytes, filename=filename)

        # TXT и подобные
        if content_type in TEXT_MIME_TYPES or (filename and filename.endswith(".txt")):
            return extract_text_from_plain(file_bytes, filename=filename)

        # Fallback: пробуем как текстовый файл
        logger.info(
            "Falling back to plain text extractor for unknown content type",
            extra={"source_filename": filename, "content_type": content_type},
        )
        return extract_text_from_plain(file_bytes, filename=filename)
    except Exception as exc:
        # В проде важно не уронить весь процесс: логируем и возвращаем пустую строку
        logger.exception(
            "Failed to extract text from file",
            extra={
                "source_filename": filename,
                "content_type": content_type,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return ""