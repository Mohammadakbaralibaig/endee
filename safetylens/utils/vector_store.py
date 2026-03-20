from typing import List, Tuple, Dict, Any
from endee import Endee, Precision


def init_endee(base_url: str = "http://localhost:8080",
               dimension: int = 768,
               index_name: str = "safetylens_index"):
    """Connect to Endee and create a fresh index."""
    client = Endee()
    client.set_base_url(f"{base_url.rstrip('/')}/api/v1")

    # Force delete old index if exists
    try:
        client.delete_index(index_name)
    except:
        pass

    # Create new index
    client.create_index(
        name=index_name,
        dimension=dimension,
        space_type="cosine",
        precision=Precision.INT8,
    )

    index = client.get_index(name=index_name)
    return client, index


def upsert_chunks(index, chunks: List[str],
                  embeddings: List[List[float]],
                  doc_name: str = "document"):
    """Store text chunks and their vectors into Endee."""
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i:05d}",
            "vector": embedding,
            "meta": {
                "text": chunk,
                "chunk_id": f"chunk_{i:05d}",
                "doc_name": doc_name,
                "chunk_index": i,
            }
        })
        if len(vectors) == 50:
            index.upsert(vectors)
            vectors = []

    if vectors:
        index.upsert(vectors)


def search_similar(index, query_vector: List[float],
                   top_k: int = 5) -> List[Dict[str, Any]]:
    """Find the most similar chunks to a query vector."""
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        ef=128,
        include_vectors=False,
    )
    return results