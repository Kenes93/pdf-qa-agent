"""QA-агент по PDF на LangChain.

Агент, а не «цепочка»: у него три инструмента, и он сам решает, чем
воспользоваться — поискать по смыслу, прочитать конкретную страницу целиком
или посмотреть оглавление документа.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .extract import ExtractedDoc

DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы по одному PDF-документу: {filename}
({num_pages} стр., распознан как {kind}, извлечено инструментом {engine}).

Правила:
1. Отвечай ТОЛЬКО на основании содержимого документа. Перед ответом обязательно
   вызови инструмент — даже если кажется, что ответ известен.
2. Если в документе ответа нет, так и скажи: «В документе этого нет» — не додумывай.
3. В конце ответа указывай страницы-источники в формате: Источник: стр. 11, 12.
4. Числа, даты и названия переноси из документа дословно, без округлений.
5. Отвечай на языке вопроса, кратко и по делу.

Инструменты: search_document — поиск по смыслу; read_page — прочитать страницу
целиком (полезно для таблиц); document_outline — заголовки документа.
"""


def build_tools(doc: ExtractedDoc, store, k: int = 5):
    """Три инструмента агента поверх извлечённого документа."""

    @tool
    def search_document(query: str) -> str:
        """Найти в документе фрагменты, релевантные запросу. Возвращает текст с номерами страниц."""
        hits = store.similarity_search(query, k=k)
        if not hits:
            return "Ничего не найдено."
        return "\n\n---\n\n".join(
            f"[стр. {h.metadata['page']}]\n{h.page_content}" for h in hits
        )

    @tool
    def read_page(page: int) -> str:
        """Прочитать страницу документа целиком по её номеру (нумерация с 1)."""
        text = doc.page_text(page)
        if text is None:
            return f"В документе {doc.num_pages} стр., страницы {page} нет."
        return text or f"Страница {page} пустая."

    @tool
    def document_outline() -> str:
        """Показать оглавление: заголовки документа с номерами страниц."""
        lines: list[str] = []
        for p in doc.pages:
            for raw in p.text.splitlines():
                line = raw.strip()
                if line.startswith("#"):
                    lines.append(f"стр. {p.page}: {line.lstrip('# ').strip()}")
        if not lines:
            return (
                f"Явных заголовков нет. Документ: {doc.filename}, "
                f"{doc.num_pages} стр., {doc.total_tables} таблиц."
            )
        return "\n".join(lines[:120])

    return [search_document, read_page, document_outline]


def build_agent(doc: ExtractedDoc, store, model: str = DEFAULT_MODEL, k: int = 5):
    """Собрать агента LangChain по конкретному документу."""
    llm = ChatOpenAI(model=model, temperature=0)
    return create_agent(
        llm,
        tools=build_tools(doc, store, k=k),
        system_prompt=SYSTEM_PROMPT.format(
            filename=doc.filename,
            num_pages=doc.num_pages,
            kind="скан (OCR)" if doc.kind == "scanned" else "текстовый PDF",
            engine=doc.engine,
        ),
    )


def ask(agent, question: str, history: list | None = None) -> dict:
    """Задать вопрос агенту. Возвращает ответ, список вызовов инструментов и страницы."""
    messages = list(history or []) + [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})
    out = result["messages"]

    tool_calls: list[dict] = []
    pages: set[int] = set()
    for msg in out:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                tool_calls.append({"name": call["name"], "args": call["args"]})
                if call["name"] == "read_page":
                    page = call["args"].get("page")
                    if isinstance(page, int):
                        pages.add(page)
        if getattr(msg, "type", "") == "tool":
            import re

            pages.update(int(m) for m in re.findall(r"\[стр\. (\d+)\]", str(msg.content)))

    return {
        "answer": out[-1].content,
        "tool_calls": tool_calls,
        "pages": sorted(pages),
        "messages": out,
    }
