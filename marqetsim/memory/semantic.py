"""Semantic memory module."""

import os
import uuid
from typing import Any

import rich
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

from marqetsim.memory.base import TinyMemory
from marqetsim.memory.rag import MarqKnowledge
from marqetsim.utils import common


class SemanticMemory(TinyMemory):
    """
    Semantic memory is the memory of meanings, understandings, and other concept-based knowledge
    unrelated to specific experiences.
    It is not ordered temporally, and it is not about remembering specific events or episodes.
    This class provides a simple implementation
    of semantic memory, where the agent can store and retrieve semantic information.
    """

    def __init__(
        self,
        documents_paths: list = None,
        web_urls: list = None,
        knowledge_base=None,
        name=None,
        persistent_path=None,
    ) -> None:
        self.knowledge_base = knowledge_base or MarqKnowledge(
            collection_name=name, persist_directory=persistent_path
        )

        self.documents = []
        self.documents_paths = []
        self.documents_web_urls = []
        self.filename_to_document = {}

        self.add_documents_paths(documents_paths)
        if web_urls:
            self.add_web_urls(web_urls)

    def _preprocess_value_for_storage(self, value: dict) -> Any:
        engram = None
        ts = value["simulation_timestamp"]

        if value["type"] == "action":
            engram = (
                "# Fact\n"
                + f"I have performed the following action at date and time {ts}:\n\n"
                + f" {value['content']}"
            )

        elif value["type"] == "stimulus":
            engram = (
                "# Stimulus\n"
                + f"I have received the following stimulus at date and time {ts}:\n\n"
                + f" {value['content']}"
            )

        # else: # Anything else here?

        return engram

    def _store(self, value: Any) -> None:
        engram_text = str(value)
        doc_id = str(uuid.uuid4())
        self.knowledge_base.add_document(engram_text, doc_id)

    def retrieve_relevant(self, relevance_target: str, top_k=20) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        results = self.knowledge_base.retrieve(relevance_target, top_k)

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            content = f"SOURCE: {meta.get('file_name', '(unknown)')}\n"
            content += f"DISTANCE: {dist:.4f}\n"
            content += f"RELEVANT CONTENT:\n{doc}"
            retrieved.append(content)

        return retrieved

    def retrieve_document_content_by_name(self, document_name: str) -> str:
        """
        Retrieves a document by its name.
        """
        if self.filename_to_document is not None:
            doc = self.filename_to_document[document_name]
            if doc is not None:
                content = "SOURCE: " + document_name
                content += "\n" + "CONTENT: " + doc.text[:10000]
                return content
            else:
                return None
        else:
            return None

    def list_documents_names(self) -> list:
        """
        Lists the names of the documents in memory.
        """
        if self.filename_to_document is not None:
            return list(self.filename_to_document.keys())
        else:
            return []

    def add_documents_paths(self, documents_paths: list) -> None:
        """
        Adds a path to a folder with documents used for semantic memory.
        """

        if documents_paths is not None:
            for documents_path in documents_paths:
                try:
                    self.add_documents_path(documents_path)
                except (FileNotFoundError, ValueError) as e:
                    rich.print(f"Error: {e}")
                    rich.print(f"Current working directory: {os.getcwd()}")
                    rich.print(f"Provided path: {documents_path}")
                    rich.print("Please check if the path exists and is accessible.")

    def add_documents_path(self, documents_path: str) -> None:
        """
        Adds a path to a folder with documents used for semantic memory.
        """

        documents = SimpleDirectoryReader(input_dir=documents_path).load_data()
        for doc in documents:
            sanitized_text = common.sanitize_raw_string(doc.text)
            doc_id = str(uuid.uuid4())
            file_name = doc.metadata.get("file_name", "unknown")
            self.filename_to_document[file_name] = doc
            self.knowledge_base.add_document(
                sanitized_text, doc_id, {"file_name": file_name}
            )

    def add_web_urls(self, web_urls: list) -> None:
        """
        Adds the data retrieved from the specified URLs to documents used for semantic memory.
        """

        filtered_urls = [url for url in web_urls if url not in self.documents_web_urls]
        self.documents_web_urls += filtered_urls

        if filtered_urls:
            self.add_web_urls(filtered_urls)

    def add_web_url(self, web_urls: str) -> None:
        """
        Adds the data retrieved from the specified URL to documents used for semantic memory.
        """
        # we do it like this because the add_web_urls could run scrapes in parallel, so it is better
        # to implement this one in terms of the other

        # self.add_web_urls([web_url])
        documents = SimpleWebPageReader(html_to_text=True).load_data(web_urls)
        for doc in documents:
            doc.text = common.sanitize_raw_string(doc.text)
            doc_id = str(uuid.uuid4())
            self.knowledge_base.add_document(doc.text, doc_id, {"source": "web"})

    def _add_documents(self, new_documents, doc_to_name_func=None) -> list:
        """
        Adds multiple documents by calling _add_document on each.
        """
        for document in new_documents:
            self._add_document(document, doc_to_name_func)

    def _add_document(self, document, doc_to_name_func=None) -> None:
        """
        Adds a single document to the semantic memory.
        """
        # Sanitize text
        document.text = common.sanitize_raw_string(document.text)

        # Determine document name if function provided
        name = None
        if doc_to_name_func is not None:
            name = doc_to_name_func(document)
            self.filename_to_document[name] = document

        # Generate unique doc_id
        doc_id = str(uuid.uuid4())

        # Add to vector DB
        self.knowledge_base.add_document(
            document.text, doc_id, metadata={"file_name": name}
        )

        # Also keep in documents list
        self.documents.append(document)

    ###########################################################
    # IO
    ###########################################################

    def _post_deserialization_init(self):

        # Reset or recreate MarqKnowledge instance if needed
        self.knowledge_base = MarqKnowledge()

        # Reload documents and web URLs into MarqKnowledge
        self.add_documents_paths(self.documents_paths)
        self.add_web_urls(self.documents_web_urls)
