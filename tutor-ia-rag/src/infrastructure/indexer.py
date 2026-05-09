"""Módulo de indexación de documentos para el pipeline RAG.

Pertenece a la capa de infraestructura. Carga, divide y persiste documentos
en ChromaDB. No contiene lógica de negocio.
"""
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import settings
from src.domain.ports import IIndexer

logger = logging.getLogger(__name__)

_LOADERS: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".pl": TextLoader,
}


def _build_default_store() -> Chroma:
    """Construye el vector store con embeddings de Ollama (nomic-embed-text)."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        ),
        persist_directory=settings.chroma_persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


class DocumentIndexer(IIndexer):
    """Indexa documentos del curso en ChromaDB usando LangChain + Ollama.

    Soporta PDFs (via PyMuPDFLoader) y archivos .txt/.pl (via TextLoader).
    Divide el contenido en chunks con overlap y persiste los embeddings
    en ChromaDB con distancia coseno.

    Args:
        docs_path: Ruta al directorio de documentos. Por defecto ``docs/curso_ia``.
        store: Vector store inyectado (opcional). Si es ``None`` se crea uno
            con Ollama embeddings. Inyectalo en tests para evitar I/O real.

    Example:
        >>> indexer = DocumentIndexer()
        >>> total = indexer.index_all()
        >>> print(f"Indexados {total} chunks")
    """

    def __init__(
        self,
        docs_path: str = "docs/curso_ia",
        store: Chroma | None = None,
    ) -> None:
        self._docs_path = Path(docs_path)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self._store = store or _build_default_store()

    def _load_documents(self, path: Path) -> list[Document]:
        """Carga todos los archivos soportados del directorio de forma recursiva.

        Args:
            path: Directorio raíz a escanear.

        Returns:
            Lista de objetos Document cargados.

        Raises:
            FileNotFoundError: Si el directorio no existe.
        """
        if not path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {path}")

        documents: list[Document] = []

        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue

            loader_cls = _LOADERS.get(file_path.suffix.lower())
            if loader_cls is None:
                logger.debug("Ignorado (extensión no soportada): %s", file_path.name)
                continue

            try:
                kwargs: dict[str, str] = (
                    {} if file_path.suffix.lower() == ".pdf" else {"encoding": "utf-8"}
                )
                loaded = loader_cls(str(file_path), **kwargs).load()
                documents.extend(loaded)
                logger.info(
                    "Cargado '%s' → %d sección(es)", file_path.name, len(loaded)
                )
            except (OSError, ValueError) as exc:
                logger.warning("Archivo ignorado '%s': %s", file_path.name, exc)
            except Exception as exc:
                logger.error(
                    "Error inesperado cargando '%s': %s",
                    file_path.name,
                    exc,
                    exc_info=True,
                )
                raise

        return documents

    def index(self, docs_path: str) -> int:
        """Implementa IIndexer. Indexa todos los documentos del directorio dado.

        Args:
            docs_path: Ruta al directorio con los documentos del curso.

        Returns:
            Número de chunks indexados. Retorna ``0`` si no hay archivos
            con extensiones soportadas.

        Raises:
            FileNotFoundError: Si el directorio no existe.
            RuntimeError: Si falla la carga o la persistencia en ChromaDB.
        """
        target = Path(docs_path)
        logger.info("Iniciando indexación desde: %s", target)

        try:
            raw_docs = self._load_documents(target)
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Error al cargar documentos: %s", exc, exc_info=True)
            raise RuntimeError(f"Fallo en la carga de documentos: {exc}") from exc

        if not raw_docs:
            logger.warning("Sin documentos soportados en: %s", target)
            return 0

        try:
            chunks = self._splitter.split_documents(raw_docs)
            logger.info(
                "Divididos en %d chunks (size=%d, overlap=%d)",
                len(chunks),
                settings.chunk_size,
                settings.chunk_overlap,
            )
            self._store.add_documents(chunks)
            logger.info(
                "Indexación completada: %d chunks persistidos en ChromaDB", len(chunks)
            )
            return len(chunks)
        except Exception as exc:
            logger.error("Error durante la indexación: %s", exc, exc_info=True)
            raise RuntimeError(f"Fallo en la indexación: {exc}") from exc

    def index_all(self) -> int:
        """Indexa todos los documentos usando la ruta configurada en ``__init__``.

        Método de conveniencia que delega a :meth:`index`.

        Returns:
            Número de chunks indexados.

        Raises:
            FileNotFoundError: Si el directorio configurado no existe.
            RuntimeError: Si falla la indexación.
        """
        return self.index(str(self._docs_path))
