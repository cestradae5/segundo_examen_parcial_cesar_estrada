# Tutor IA RAG

Sistema de preguntas y respuestas sobre material del curso usando RAG (Retrieval-Augmented Generation).

## Stack

- **Embeddings + LLM**: OpenAI (`text-embedding-3-small` + `gpt-4o-mini`)
- **Vector store**: ChromaDB (persistido localmente en `vectorstore/`)
- **API**: FastAPI
- **Arquitectura**: Clean Architecture (domain / application / infrastructure / interfaces)

## Estructura

```
tutor-ia-rag/
├── docs/curso_ia/      ← colocá aquí los .txt, .md del curso
├── vectorstore/        ← generado automáticamente (no subir a git)
├── src/
│   ├── domain/         ← entidades puras
│   ├── application/    ← casos de uso
│   ├── infrastructure/ ← ChromaDB, OpenAI, indexer
│   └── interfaces/     ← FastAPI
├── tests/
├── main.py
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Editá `.env` con tu API key de OpenAI.

## Uso

```bash
# 1. Indexar documentos del curso
python main.py index --docs docs/curso_ia

# 2. Levantar la API
python main.py serve
```

La API queda disponible en `http://localhost:8000`.

### Endpoint `/ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es el aprendizaje supervisado?", "top_k": 5}'
```

## Tests

```bash
pytest tests/
```
