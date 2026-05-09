from unittest.mock import MagicMock

import pytest

from src.application.use_cases import AskTutorUC, IndexDocumentsUC
from src.domain.entities import AnswerResponse, DocumentChunk, Query
from src.infrastructure.indexer import _chunk_text
from src.infrastructure.retriever import RAGRetriever


# --- Domain ---

def test_query_defaults():
    q = Query(text="¿Qué es una red neuronal?")
    assert q.top_k == 5


def test_document_chunk_default_metadata():
    chunk = DocumentChunk(id="1", content="texto", source="file.txt")
    assert chunk.metadata == {}


def test_answer_response_creation():
    chunk = DocumentChunk(id="1", content="texto", source="file.txt")
    resp = AnswerResponse(answer="respuesta", sources=[chunk], query="pregunta")
    assert resp.answer == "respuesta"
    assert len(resp.sources) == 1


# --- Chunking ---

def test_chunk_text_produces_multiple_chunks():
    text = "a" * 1200
    chunks = _chunk_text(text, size=500, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    chunks = _chunk_text("texto corto", size=500, overlap=50)
    assert len(chunks) == 1


# --- IndexDocumentsUC ---

def test_index_documents_uc_delegates_to_indexer():
    mock_indexer = MagicMock()
    mock_indexer.index.return_value = 42
    uc = IndexDocumentsUC(mock_indexer)
    result = uc.execute("docs/curso_ia")
    assert result == 42
    mock_indexer.index.assert_called_once_with("docs/curso_ia")


# --- AskTutorUC ---

def test_ask_tutor_uc_delegates_to_retriever():
    mock_retriever = MagicMock()
    expected = AnswerResponse(answer="La IA es...", sources=[], query="¿Qué es la IA?")
    mock_retriever.ask.return_value = expected

    uc = AskTutorUC(mock_retriever)
    query = Query(text="¿Qué es la IA?")
    result = uc.execute(query)

    assert result.answer == "La IA es..."
    mock_retriever.ask.assert_called_once_with(query)


# --- RAGRetriever ---

def test_rag_retriever_returns_answer_with_sources():
    mock_chroma = MagicMock()
    mock_openai = MagicMock()

    mock_openai.embed.return_value = [0.1] * 1536
    mock_chroma.query.return_value = {
        "ids": [["id-1", "id-2"]],
        "documents": [["chunk 1", "chunk 2"]],
        "metadatas": [[{"source": "file.txt"}, {"source": "file2.txt"}]],
    }
    mock_openai.complete.return_value = "Respuesta generada"

    retriever = RAGRetriever(mock_chroma, mock_openai)
    result = retriever.ask(Query(text="¿Qué es ML?", top_k=2))

    assert result.answer == "Respuesta generada"
    assert len(result.sources) == 2
    assert result.sources[0].source == "file.txt"
    assert result.query == "¿Qué es ML?"
