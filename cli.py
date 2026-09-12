#!/usr/bin/env python3
"""Консольный запуск агента.

    python cli.py раздатка/kaztelecom.pdf "Какая выручка за 2024 год?"
    python cli.py doc.pdf            # интерактивный режим
    python cli.py scan.pdf --ocr "О чём документ?"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from pdf_qa import ask, build_agent, build_store, extract


def main() -> int:
    parser = argparse.ArgumentParser(description="QA-агент по PDF (LangChain + Docling)")
    parser.add_argument("pdf", help="путь к PDF")
    parser.add_argument("question", nargs="*", help="вопрос; без него — интерактивный режим")
    parser.add_argument("--ocr", action="store_true", help="принудительно включить OCR")
    parser.add_argument("--no-ocr", action="store_true", help="принудительно выключить OCR")
    parser.add_argument("--model", default="gpt-4.1-mini", help="модель OpenAI")
    args = parser.parse_args()

    load_dotenv()

    force_ocr = True if args.ocr else (False if args.no_ocr else None)
    doc = extract(args.pdf, force_ocr=force_ocr, on_progress=lambda m: print(f"  · {m}"))
    print(
        f"\nДокумент: {doc.filename} | тип: {doc.kind} | извлечение: {doc.engine} | "
        f"страниц: {doc.num_pages} | символов: {doc.total_chars} | таблиц: {doc.total_tables}\n"
    )

    store, chunks = build_store(doc)
    print(f"Индекс: {chunks} фрагментов\n")
    agent = build_agent(doc, store, model=args.model)

    questions = [" ".join(args.question)] if args.question else None
    if questions:
        for q in questions:
            _answer(agent, q)
        return 0

    print("Интерактивный режим. Пустая строка — выход.\n")
    while True:
        try:
            q = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        _answer(agent, q)


def _answer(agent, question: str) -> None:
    print(f"Вопрос: {question}")
    result = ask(agent, question)
    tools = ", ".join(f"{c['name']}({c['args']})" for c in result["tool_calls"]) or "—"
    print(f"Инструменты: {tools}")
    print(f"Ответ: {result['answer']}\n")


if __name__ == "__main__":
    sys.exit(main())
