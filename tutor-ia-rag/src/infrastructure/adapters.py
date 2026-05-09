import os
import chromadb
from openai import OpenAI


class ChromaAdapter:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("curso_ia")

    def add(self, ids: list, embeddings: list, documents: list, metadatas: list):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

    def count(self) -> int:
        return self.collection.count()


class OpenAIAdapter:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.embed_model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.embed_model)
        return [item.embedding for item in response.data]

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
