"""Индекс по документу: чанки + векторный поиск.

Документ на 46 страниц целиком в контекст не поместится (а если и поместится —
платить за это на каждый вопрос незачем), поэтому текст режется на куски,
каждый кусок помнит свою страницу, и агент ищет по ним семантически.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extract import ExtractedDoc

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def to_documents(doc: ExtractedDoc) -> list[Document]:
    """Разрезать постраничный текст на чанки, не теряя номер страницы."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks: list[Document] = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        for i, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": doc.filename,
                        "page": page.page,
                        "chunk": i,
                    },
                )
            )
    return chunks


def build_store(doc: ExtractedDoc, embedding_model: str = EMBEDDING_MODEL):
    """Собрать векторное хранилище в памяти по извлечённому документу."""
    chunks = to_documents(doc)
    if not chunks:
        raise ValueError(
            "Из документа не извлечён ни один фрагмент текста — "
            "попробуйте принудительно включить OCR."
        )
    store = InMemoryVectorStore(OpenAIEmbeddings(model=embedding_model))
    store.add_documents(chunks)
    return store, len(chunks)
