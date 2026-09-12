"""Извлечение текста и структуры из PDF.

Документы бывают двух сортов, и обрабатываются они по-разному:

* **text-based** — внутри уже лежит текстовый слой, OCR только замедлит работу;
* **скан** — текстового слоя нет, pypdf возвращает пустоту, нужен OCR.

Какой случай перед нами, модуль решает сам: сначала дёшево «прощупывает»
документ через pypdf, потом запускает Docling с подходящими настройками.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

# Ниже этого числа символов на страницу считаем, что текстового слоя нет.
MIN_CHARS_PER_PAGE = 100


@dataclass
class Page:
    """Одна страница документа после извлечения."""

    page: int
    text: str
    tables: int = 0


@dataclass
class ExtractedDoc:
    """Результат извлечения — то, из чего дальше строится индекс и ответы."""

    filename: str
    kind: str  # "text" | "scanned"
    engine: str  # чем извлекали: docling / docling+ocr / pypdf
    num_pages: int
    avg_chars_raw: float  # символов на страницу по данным pypdf (до OCR)
    elapsed_sec: float
    pages: list[Page] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n\n".join(
            f"<!-- page {p.page} -->\n{p.text}" for p in self.pages if p.text.strip()
        )

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def total_tables(self) -> int:
        return sum(p.tables for p in self.pages)

    def page_text(self, page: int) -> str | None:
        for p in self.pages:
            if p.page == page:
                return p.text
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedDoc":
        pages = [Page(**p) for p in data.pop("pages", [])]
        return cls(pages=pages, **data)


def probe(pdf_path: Path) -> tuple[int, float]:
    """Быстрая разведка: сколько страниц и сколько текста отдаёт pypdf.

    Возвращает (число страниц, среднее число символов на страницу).
    Это и есть ответ на вопрос «нужен ли здесь OCR».
    """
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    chars = 0
    for page in reader.pages:
        try:
            chars += len(page.extract_text() or "")
        except Exception:  # битая страница не должна ронять разведку
            continue
    return num_pages, chars / max(num_pages, 1)


def _docling_converter(use_ocr: bool):
    """Конвертер Docling: с OCR для сканов, без него — для текстовых PDF."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True  # структура таблиц нужна в обоих случаях
    opts.do_ocr = use_ocr
    if use_ocr:
        # Родной OCR macOS (Vision) — без внешних бинарников и тяжёлых моделей.
        opts.ocr_options = OcrMacOptions(
            lang=["ru-RU", "en-US"],
            force_full_page_ocr=True,
        )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _extract_with_docling(pdf_path: Path, use_ocr: bool) -> tuple[list[Page], int]:
    """Docling → постраничный Markdown с сохранением структуры и таблиц."""
    converter = _docling_converter(use_ocr)
    result = converter.convert(str(pdf_path))
    doc = result.document

    tables_by_page: dict[int, int] = {}
    for table in getattr(doc, "tables", []) or []:
        for prov in getattr(table, "prov", []) or []:
            tables_by_page[prov.page_no] = tables_by_page.get(prov.page_no, 0) + 1

    page_numbers = sorted(doc.pages.keys()) if getattr(doc, "pages", None) else []
    pages = [
        Page(
            page=no,
            text=doc.export_to_markdown(page_no=no).strip(),
            tables=tables_by_page.get(no, 0),
        )
        for no in page_numbers
    ]
    return pages, len(page_numbers)


def _extract_with_pypdf(pdf_path: Path) -> list[Page]:
    """Запасной путь, если Docling недоступен или упал."""
    reader = PdfReader(str(pdf_path))
    return [
        Page(page=i + 1, text=(page.extract_text() or "").strip())
        for i, page in enumerate(reader.pages)
    ]


def _cache_key(pdf_path: Path, force_ocr: bool | None) -> str:
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    return f"{digest}-{ {True: 'ocr', False: 'noocr', None: 'auto'}[force_ocr] }"


def extract(
    pdf_path: str | Path,
    cache_dir: str | Path | None = ".cache",
    force_ocr: bool | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ExtractedDoc:
    """Извлечь текст и структуру из PDF.

    force_ocr=None — решать автоматически по результатам разведки;
    True/False — принудительно включить или выключить OCR.
    """
    pdf_path = Path(pdf_path)
    say = on_progress or (lambda _msg: None)

    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"{_cache_key(pdf_path, force_ocr)}.json"
        if cache_file.exists():
            say("Беру разобранный документ из кэша")
            return ExtractedDoc.from_dict(json.loads(cache_file.read_text("utf-8")))

    started = time.perf_counter()
    say("Разведка: смотрю, есть ли текстовый слой")
    num_pages, avg_chars = probe(pdf_path)

    scanned = avg_chars < MIN_CHARS_PER_PAGE if force_ocr is None else force_ocr
    kind = "scanned" if scanned else "text"
    say(
        f"{num_pages} стр., {avg_chars:.0f} символов на страницу → "
        + ("это скан, включаю OCR" if scanned else "текстовый PDF, OCR не нужен")
    )

    try:
        pages, _ = _extract_with_docling(pdf_path, use_ocr=scanned)
        engine = "docling+ocrmac" if scanned else "docling"
    except Exception as exc:  # noqa: BLE001 — падать здесь нельзя, есть запасной путь
        say(f"Docling не справился ({exc}), откатываюсь на pypdf")
        pages = _extract_with_pypdf(pdf_path)
        engine = "pypdf (fallback)"

    doc = ExtractedDoc(
        filename=pdf_path.name,
        kind=kind,
        engine=engine,
        num_pages=num_pages,
        avg_chars_raw=round(avg_chars, 1),
        elapsed_sec=round(time.perf_counter() - started, 1),
        pages=pages,
    )
    say(f"Готово за {doc.elapsed_sec} с: {doc.total_chars} символов, {doc.total_tables} таблиц")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return doc
