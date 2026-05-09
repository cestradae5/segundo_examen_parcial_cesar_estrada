from src.domain.entities import AnswerResponse, DocumentChunk, Query


def test_query_default_top_k():
    q = Query(text="¿Qué es una red neuronal?")
    assert q.top_k == 5


def test_query_custom_top_k():
    q = Query(text="¿Qué es ML?", top_k=3)
    assert q.top_k == 3


def test_document_chunk_default_metadata():
    chunk = DocumentChunk(id="1", content="contenido", source="file.txt")
    assert chunk.metadata == {}


def test_document_chunk_with_metadata():
    chunk = DocumentChunk(id="1", content="texto", source="file.txt", metadata={"page": 1})
    assert chunk.metadata["page"] == 1


def test_answer_response_fields():
    chunk = DocumentChunk(id="1", content="texto", source="file.txt")
    resp = AnswerResponse(answer="respuesta", sources=[chunk], query="pregunta")
    assert resp.answer == "respuesta"
    assert len(resp.sources) == 1
    assert resp.query == "pregunta"
