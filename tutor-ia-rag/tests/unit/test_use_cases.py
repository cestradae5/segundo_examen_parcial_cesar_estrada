from unittest.mock import MagicMock

from src.application.use_cases import AskTutorUC, IndexDocumentsUC
from src.domain.entities import AnswerResponse, Query


def test_index_documents_delegates_to_indexer():
    mock_indexer = MagicMock()
    mock_indexer.index.return_value = 42

    uc = IndexDocumentsUC(mock_indexer)
    result = uc.execute("docs/curso_ia")

    assert result == 42
    mock_indexer.index.assert_called_once_with("docs/curso_ia")


def test_index_documents_returns_zero_when_no_files():
    mock_indexer = MagicMock()
    mock_indexer.index.return_value = 0

    uc = IndexDocumentsUC(mock_indexer)
    assert uc.execute("docs/vacio") == 0


def test_ask_tutor_delegates_to_retriever():
    mock_retriever = MagicMock()
    expected = AnswerResponse(answer="La IA es...", sources=[], query="¿Qué es la IA?")
    mock_retriever.ask.return_value = expected

    uc = AskTutorUC(mock_retriever)
    query = Query(text="¿Qué es la IA?")
    result = uc.execute(query)

    assert result.answer == "La IA es..."
    mock_retriever.ask.assert_called_once_with(query)
