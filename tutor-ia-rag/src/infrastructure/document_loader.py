from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


class LangChainDocumentLoader:
    """Carga archivos de texto desde un directorio usando LangChain loaders."""

    def load(self, docs_path: str) -> list[Document]:
        path = Path(docs_path)
        if not path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {docs_path}")

        loader = DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        return loader.load()
