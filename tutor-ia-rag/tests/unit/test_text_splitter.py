from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.infrastructure.text_splitter import ChunkSplitter


def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_split_short_doc_single_chunk():
    splitter = ChunkSplitter()
    docs = [_make_doc("texto corto")]
    chunks = splitter.split(docs)
    assert len(chunks) == 1


def test_split_long_doc_multiple_chunks():
    splitter = ChunkSplitter()
    docs = [_make_doc("palabra " * 300)]  # ~2400 chars → al menos 5 chunks con size=500
    chunks = splitter.split(docs)
    assert len(chunks) > 1


def test_split_preserves_source_metadata():
    splitter = ChunkSplitter()
    docs = [_make_doc("texto de prueba", source="mi_archivo.txt")]
    chunks = splitter.split(docs)
    assert chunks[0].metadata["source"] == "mi_archivo.txt"


def test_split_empty_list():
    splitter = ChunkSplitter()
    assert splitter.split([]) == []
