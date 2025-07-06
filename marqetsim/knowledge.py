import chromadb
from chromadb.utils import embedding_functions


class MarqKnowledge:
    """
    MarqKnowledge is a ChromaDB-powered vector knowledge store
    for semantic search and retrieval.
    """

    def __init__(self, collection_name="marq-knowledge"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def add_document(self, doc_text: str, doc_id: str, metadata: dict = None):
        """
        Adds a document to MarqKnowledge.
        """
        embedding = self.embedding_fn([doc_text])[0]
        self.collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata or {}],
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
