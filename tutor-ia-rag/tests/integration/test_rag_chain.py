"""
Tests de integración — requieren OPENAI_API_KEY válida.
Corré con: pytest tests/integration/ -v
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.domain.entities import Query
from src.infrastructure.rag_chain import RAGChain


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="El aprendizaje supervisado usa datos etiquetados.", metadata={"source": "ia.txt"}),
        Document(page_content="Requiere pares (entrada, salida) para entrenar.", metadata={"source": "ia.txt"}),
    ]
    store.as_retriever.return_value = retriever
    return store


def test_rag_chain_returns_answer(mock_vector_store):
    with patch("src.infrastructure.rag_chain.RetrievalQA") as MockQA:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "El aprendizaje supervisado usa datos etiquetados.",
            "source_documents": [
                Document(page_content="chunk 1", metadata={"source": "ia.txt"}),
            ],
        }
        MockQA.from_chain_type.return_value = mock_chain

        chain = RAGChain(mock_vector_store)
        result = chain.ask(Query(text="¿Qué es el aprendizaje supervisado?"))

    assert "aprendizaje supervisado" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].source == "ia.txt"
    assert result.query == "¿Qué es el aprendizaje supervisado?"


def test_rag_chain_empty_sources(mock_vector_store):
    with patch("src.infrastructure.rag_chain.RetrievalQA") as MockQA:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "No encontré información sobre ese tema.",
            "source_documents": [],
        }
        MockQA.from_chain_type.return_value = mock_chain

        chain = RAGChain(mock_vector_store)
        result = chain.ask(Query(text="¿Qué es la computación cuántica?"))

    assert result.sources == []
