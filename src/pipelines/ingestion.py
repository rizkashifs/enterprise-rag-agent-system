from src.core.models import Document, Chunk


def ingest(docs: list[Document], chunk_size: int = 100) -> list[Chunk]:
    """Split documents into overlapping word chunks. Embeddings are added by VectorIndex."""
    chunks = []
    for doc in docs:
        words = doc.content.split()
        if not words:
            continue

        if len(words) <= chunk_size:
            chunks.append(Chunk(
                id=f"{doc.id}-0",
                doc_id=doc.id,
                content=doc.content,
                source=doc.source,
            ))
            continue

        step = max(1, chunk_size // 2)
        for i in range(0, len(words), step):
            text = " ".join(words[i:i + chunk_size])
            chunks.append(Chunk(
                id=f"{doc.id}-{i}",
                doc_id=doc.id,
                content=text,
                source=doc.source,
            ))

    return chunks
