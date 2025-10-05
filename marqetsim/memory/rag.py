"""
This code is part of a semantic knowledge management system using ChromaDB for vector
storage and retrieval. It includes functionality to chunk text documents,
add them to a knowledge base, and retrieve relevant documents based on semantic queries.
"""

import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class MarqKnowledge:
    """
    MarqKnowledge is a ChromaDB-powered vector knowledge store
    for semantic search and retrieval.
    """

    def __init__(self, collection_name="marq-knowledge", persist_directory=None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def add_document(self, doc_text: str, doc_id: str, metadata: dict = None):
        """
        Adds a document to MarqKnowledge.
        """
        chunks = chunk_text(doc_text)

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        embeddings = self.embedding_fn(chunks)
        metadatas = [metadata or {} for _ in chunks]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieves relevant documents based on a query.
        """
        query_embedding = self.embedding_fn([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results
