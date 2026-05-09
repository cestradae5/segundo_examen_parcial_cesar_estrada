import argparse

from src.config import settings


def _index(docs_path: str) -> None:
    from src.application.use_cases import IndexDocumentsUC
    from src.infrastructure.vector_store import ChromaVectorStore

    store = ChromaVectorStore()
    uc = IndexDocumentsUC(store)
    total = uc.execute(docs_path)
    print(f"Indexados {total} chunks desde '{docs_path}'")


def _serve() -> None:
    import uvicorn

    uvicorn.run(
        "src.interfaces.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Tutor IA RAG")
    subparsers = parser.add_subparsers(dest="command")

    idx = subparsers.add_parser("index", help="Indexar documentos del curso")
    idx.add_argument("--docs", default="docs/curso_ia", help="Ruta a los documentos")

    subparsers.add_parser("serve", help="Levantar la API en localhost:8000")

    args = parser.parse_args()

    if args.command == "index":
        _index(args.docs)
    elif args.command == "serve":
        _serve()
    else:
        parser.print_help()
