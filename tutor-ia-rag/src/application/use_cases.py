"""Casos de uso de la aplicación Tutor IA RAG.

Esta capa contiene TODA la lógica de negocio. No depende de implementaciones
concretas de infraestructura — solo de las interfaces del dominio (ports).
"""
import logging

from src.config import settings
from src.domain.entities import AnswerResponse, DocumentChunk, Query
from src.domain.ports import IIndexer, IRetriever

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD: float = settings.similarity_threshold
_OUT_OF_SCOPE_MSG: str = "Esa pregunta está fuera del temario del curso."


class IndexDocumentsUC:
    """Caso de uso: indexar todos los documentos del curso.

    Args:
        indexer: Implementación de ``IIndexer`` inyectada por el llamador.
            No instancia adapters directamente.

    Example:
        >>> from src.infrastructure.indexer import DocumentIndexer
        >>> uc = IndexDocumentsUC(indexer=DocumentIndexer())
        >>> total = uc.execute()
    """

    def __init__(self, indexer: IIndexer) -> None:
        self.indexer = indexer

    def execute(self, docs_path: str = "docs/curso_ia") -> int:
        """Ejecuta la indexación de documentos del curso.

        Args:
            docs_path: Ruta al directorio de documentos.

        Returns:
            Número de chunks indexados en ChromaDB.
        """
        logger.info("[IndexDocumentsUC] Indexando desde '%s'", docs_path)
        total = self.indexer.index(docs_path)
        logger.info("[IndexDocumentsUC] Completado: %d chunks", total)
        return total


class AskTutorUC:
    """Caso de uso: responder preguntas sobre el curso usando RAG.

    Orquesta la recuperación de chunks y la generación de respuestas.
    Aplica la regla de negocio del umbral de similitud: si ningún chunk
    supera ``0.6``, la pregunta se considera fuera del temario y NO se
    llama al LLM (evita alucinaciones y ahorra tokens).

    Args:
        retriever: Implementación de ``IRetriever`` inyectada por el llamador.
            No instancia adapters directamente.

    Example:
        >>> from src.infrastructure.retriever import QueryRetriever
        >>> uc = AskTutorUC(retriever=QueryRetriever())
        >>> response = uc.execute(Query(text="¿Qué es una red neuronal?"))
    """

    def __init__(self, retriever: IRetriever) -> None:
        self.retriever = retriever

    def execute(self, query: Query) -> AnswerResponse:
        """Ejecuta el pipeline RAG con validación de umbral de similitud.

        Pasos:
            1. Recupera los top-k chunks con sus scores de similitud.
            2. Verifica que el score máximo supere el umbral (``0.6``).
            3. Si no lo supera, retorna mensaje de fuera de temario.
            4. Si lo supera, genera la respuesta con el LLM y cita las fuentes.

        Args:
            query: Objeto Query con el texto de la pregunta del alumno.

        Returns:
            AnswerResponse con la respuesta generada y los chunks fuente.
            Si la similitud máxima es menor a ``0.6``, retorna el mensaje
            de fuera de temario con ``sources`` vacío.
        """
        logger.info("[AskTutorUC] Query: '%s'", query.text)

        results: list[tuple[DocumentChunk, float]] = self.retriever.retrieve(query.text)

        if not results:
            logger.warning("[AskTutorUC] Sin resultados en el índice.")
            return AnswerResponse(
                answer=(
                    "No hay información indexada. "
                    "Ejecutá primero: python main.py index"
                ),
                sources=[],
                query=query.text,
            )

        max_score: float = max(score for _, score in results)
        logger.info(
            "[AskTutorUC] Score máximo: %.3f (umbral: %.1f)",
            max_score,
            _SIMILARITY_THRESHOLD,
        )

        if max_score < _SIMILARITY_THRESHOLD:
            logger.info("[AskTutorUC] Score insuficiente → fuera del temario")
            return AnswerResponse(
                answer=_OUT_OF_SCOPE_MSG,
                sources=[],
                query=query.text,
            )

        chunks: list[DocumentChunk] = [chunk for chunk, _ in results]
        answer = self.retriever.generate_answer(query.text, chunks)

        logger.info("[AskTutorUC] Respuesta generada con %d fuentes", len(chunks))
        return AnswerResponse(answer=answer, sources=chunks, query=query.text)
