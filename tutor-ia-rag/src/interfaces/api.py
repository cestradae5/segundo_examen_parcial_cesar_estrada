from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.application.use_cases import AskTutorUC
from src.domain.entities import Query
from src.infrastructure.rag_chain import RAGChain
from src.infrastructure.vector_store import ChromaVectorStore

app = FastAPI(title="Tutor IA RAG", version="1.0.0")


def _build_use_case() -> AskTutorUC:
    store = ChromaVectorStore()
    chain = RAGChain(store)
    return AskTutorUC(chain)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponseDTO(BaseModel):
    answer: str
    sources: list[str]
    query: str


@app.post("/ask", response_model=AskResponseDTO)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    uc = _build_use_case()
    query = Query(text=request.question, top_k=request.top_k)
    result = uc.execute(query)

    return AskResponseDTO(
        answer=result.answer,
        sources=[chunk.source for chunk in result.sources],
        query=result.query,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
