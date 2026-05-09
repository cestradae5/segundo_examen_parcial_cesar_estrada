from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    top_k: int = 5


@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AnswerResponse:
    answer: str
    sources: list[DocumentChunk]
    query: str
