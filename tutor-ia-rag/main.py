"""Punto de entrada principal del Tutor IA RAG.

Uso:
    python main.py index
    python main.py ask "¿Qué es el aprendizaje supervisado?"
"""
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def cmd_index() -> None:
    """Indexa todos los documentos de docs/curso_ia/ en ChromaDB."""
    from src.application.use_cases import IndexDocumentsUC
    from src.infrastructure.indexer import DocumentIndexer

    indexer = DocumentIndexer()
    uc = IndexDocumentsUC(indexer)

    try:
        total = uc.execute()
        print(f"\nIndexados {total} chunks en ChromaDB.")
    except FileNotFoundError as exc:
        logger.error("Directorio no encontrado: %s", exc)
        print(f"\nError: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Fallo en la indexación: %s", exc)
        print(f"\nError: {exc}")
        sys.exit(1)


def cmd_ask(question: str) -> None:
    """Responde una pregunta usando el pipeline RAG.

    Args:
        question: Pregunta del alumno en lenguaje natural.
    """
    from src.application.use_cases import AskTutorUC
    from src.domain.entities import Query
    from src.infrastructure.retriever import QueryRetriever

    retriever = QueryRetriever()
    uc = AskTutorUC(retriever)

    try:
        result = uc.execute(Query(text=question))
    except RuntimeError as exc:
        logger.error("Fallo en el pipeline RAG: %s", exc)
        print(f"\nError: {exc}")
        sys.exit(1)

    print(f"\nRespuesta:\n{result.answer}")

    if result.sources:
        unique_sources = sorted({chunk.source for chunk in result.sources})
        print("\nFuentes:")
        for src in unique_sources:
            print(f"  · {src}")


def main() -> None:
    """Punto de entrada CLI. Parsea el comando y delega la ejecución."""
    if len(sys.argv) < 2:
        print("Tutor IA RAG\n")
        print("Comandos disponibles:")
        print("  python main.py index              Indexa los documentos del curso")
        print('  python main.py ask "<pregunta>"   Responde una pregunta')
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "index":
        cmd_index()

    elif command == "ask":
        if len(sys.argv) < 3:
            print('Error: falta la pregunta.')
            print('Ejemplo: python main.py ask "¿Qué es backpropagation?"')
            sys.exit(1)
        cmd_ask(sys.argv[2])

    else:
        logger.error("Comando desconocido: %r", command)
        print(f"Comando desconocido: '{command}'")
        print("Opciones: index, ask")
        sys.exit(1)


if __name__ == "__main__":
    main()
