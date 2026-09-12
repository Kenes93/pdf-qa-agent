"""Streamlit-интерфейс QA-агента по PDF."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from pdf_qa import ask, build_agent, build_store, extract

load_dotenv()

UPLOADS = Path("uploads")
# Документы лабораторной, если проект лежит рядом с её раздаткой.
SAMPLES_DIR = Path("../лабораторная09/раздатка")

st.set_page_config(page_title="PDF QA-агент", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def prepare(path: str, force_ocr: bool | None, model: str):
    """Разобрать документ, построить индекс и агента. Кэшируется на сессию."""
    doc = extract(path, force_ocr=force_ocr)
    store, chunks = build_store(doc)
    agent = build_agent(doc, store, model=model)
    return doc, agent, chunks


def sidebar() -> tuple[Path | None, bool | None, str]:
    with st.sidebar:
        st.header("Документ")

        samples = sorted(SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.exists() else []
        uploaded = st.file_uploader("Загрузить PDF", type="pdf")

        path: Path | None = None
        if uploaded is not None:
            UPLOADS.mkdir(exist_ok=True)
            path = UPLOADS / uploaded.name
            path.write_bytes(uploaded.getbuffer())
        elif samples:
            choice = st.selectbox(
                "или выбрать пример", ["—"] + [p.name for p in samples], index=0
            )
            if choice != "—":
                path = SAMPLES_DIR / choice

        st.header("Настройки")
        ocr_mode = st.radio(
            "OCR",
            ["Авто", "Всегда", "Никогда"],
            help="Авто: агент сам решает по наличию текстового слоя",
        )
        force_ocr = {"Авто": None, "Всегда": True, "Никогда": False}[ocr_mode]
        model = st.selectbox("Модель", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"], index=0)

        if not os.getenv("OPENAI_API_KEY"):
            st.error("Не задан OPENAI_API_KEY — положите его в .env")

    return path, force_ocr, model


def show_stats(doc, chunks: int) -> None:
    scanned = doc.kind == "scanned"
    cols = st.columns(5)
    cols[0].metric("Страниц", doc.num_pages)
    cols[1].metric("Тип PDF", "скан" if scanned else "текст")
    cols[2].metric("Символов", f"{doc.total_chars:,}".replace(",", " "))
    cols[3].metric("Таблиц", doc.total_tables)
    cols[4].metric("Фрагментов", chunks)
    verdict = (
        f"текстового слоя нет ({doc.avg_chars_raw:.0f} символов на страницу) → включён OCR"
        if scanned
        else f"текстовый слой на месте ({doc.avg_chars_raw:.0f} символов на страницу) → OCR не нужен"
    )
    st.caption(f"Разведка pypdf: {verdict}. Разбор занял {doc.elapsed_sec} с, движок: {doc.engine}.")


def main() -> None:
    st.title("📄 QA-агент по PDF")
    st.caption(
        "Извлечение — Docling (структура и таблицы) с автоматическим OCR для сканов. "
        "Ответы — агент LangChain с инструментами поиска по документу."
    )

    path, force_ocr, model = sidebar()
    if path is None:
        st.info("Выберите PDF в левой панели — пример из раздатки или свой файл.")
        return

    with st.spinner(f"Разбираю {path.name} — для скана это дольше, идёт OCR…"):
        doc, agent, chunks = prepare(str(path), force_ocr, model)

    show_stats(doc, chunks)

    key = f"{path}:{force_ocr}:{model}"
    if st.session_state.get("doc_key") != key:
        st.session_state["doc_key"] = key
        st.session_state["chat"] = []

    tab_chat, tab_text = st.tabs(["Вопросы", "Извлечённый текст"])

    with tab_text:
        page_no = st.number_input("Страница", 1, doc.num_pages, 1)
        st.markdown(doc.page_text(int(page_no)) or "_Страница пустая_")

    with tab_chat:
        for item in st.session_state.get("chat", []):
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("tools"):
                    with st.expander("Как агент искал ответ"):
                        for call in item["tools"]:
                            st.code(f"{call['name']}({call['args']})", language="python")

        question = st.chat_input("Спросите что-нибудь по документу")
        if question:
            # История диалога, чтобы работали уточняющие вопросы («а в 2024?»).
            history = [
                HumanMessage(item["content"])
                if item["role"] == "user"
                else AIMessage(item["content"])
                for item in st.session_state["chat"]
            ]
            with st.chat_message("user"):
                st.markdown(question)
            with st.spinner("Ищу в документе…"):
                result = ask(agent, question, history=history)
            st.session_state["chat"].append({"role": "user", "content": question})
            st.session_state["chat"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "tools": result["tool_calls"],
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
