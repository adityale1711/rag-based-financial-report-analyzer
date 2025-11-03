import chromadb
from typing import Any
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ...domain.entities import DocumentChunk, RetrievalResult
from ...domain.repositories import IVectorStore, VectorStoreError


class ChromaDBStore(IVectorStore):
    """ChromaDB implementation of the vector store interface.

    This class handles storing document embeddings and retrieving relevant
    chunks using ChromaDB and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "financial_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the ChromaDB store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
            embedding_model: Name of the sentence-transformers model.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Initialize the embedding model
        self._embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create the collection
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get the existing collection
            collection = self._client.get_collection(self.collection_name)
        except Exception:
            # Create a new collection if it doesn't exist
            collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "Financial document embeddings"}
            )
        return collection
    
    def add_documents(
        self, 
        chunks: list[DocumentChunk]
    ) -> None: 
        """Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add.

        Raises:
            VectorStoreError: If adding documents fails.
        """
        if not chunks:
            return
        
        try:
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)

                # Prepare metadata
                metadata = {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number or 0,
                    "chunk_index": chunk.chunk_index or 0,
                    "content_length": len(chunk.content)
                }

                # Add any additional metadata
                if chunk.metadata:
                    metadata.update(chunk.metadata)

                metadatas.append(metadata)

            # Generate embeddings in batch
            embeddings = self._embedding_model.encode(
                documents,
                convert_to_tensor=False,
                show_progress_bar=False
            ).tolist()

            # Add to chromaDB
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to add documents to vector store: {str(e)}"
            ) from e
        
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> RetrievalResult:
        """Search for relevant document chunks.

        Args:
            query: The search query.
            n_results: Number of results to return.

        Returns:
            RetrievalResult with relevant document chunks.

        Raises:
            VectorStoreError: If search fails.
        """
        try:
            # Generate query embedding
            query_embedding = self._embedding_model.encode(
                [query],
                convert_to_tensor=False,
                show_progress_bar=False
            ).tolist()

            # Perform the search
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self._collection.count()), # Ensure we don't request more than available
                include=['documents', 'metadatas', 'distances']
            )

            # Parse results
            chunks = []
            retrieval_scores = []

            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    document_text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Convert distance to similarity score
                    similarity_score = 1 / (1 + distance)
                    retrieval_scores.append(similarity_score)

                    chunk = DocumentChunk(
                        chunk_id=doc_id,
                        document_name=metadata.get("document_name", "unknown"),
                        content=document_text,
                        page_number=metadata.get("page_number"),
                        chunk_index=metadata.get("chunk_index"),
                        metadata=metadata
                    )
                    chunks.append(chunk)

            # Calculate average retrieval score
            avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

            return RetrievalResult(
                chunks=chunks,
                query=query,
                retrieval_score=avg_score,
                total_retrieved=len(chunks)
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to search vector store: {str(e)}"
            ) from e
        
    def is_empty(self) -> bool:
        """Check if the vector store is empty.

        Returns:
            True if the store has no documents, False otherwise.
        """
        try:
            return self._collection.count() == 0
        except Exception:
            return True
        
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection.

        Returns:
            Dictionary with collection information.
        """
        try:
            count = self._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
                "is_empty": self.is_empty()
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "is_empty": True
            }
        
    def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it.

        Raises:
            VectorStoreError: If reset fails.
        """
        try:
            # Delete the collection
            self._client.delete_collection(self.collection_name)

            # Recreate it
            self._collection = self._get_or_create_collection()
        except Exception as e:
            raise VectorStoreError(
                f"Failed to reset vector store collection: {str(e)}"
            ) from e
        
    def get_document_names(self) -> list[str]:
        """Get a list of unique document names stored in the vector store.

        Returns:
            List of document names.
        """
        try:
            # Get all metadata to extract unique document names
            results = self._collection.get(
                include=['metadatas']
            )
            document_names = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    doc_name = metadata.get("document_name", "Unknown")
                    document_names.add(doc_name)

            return sorted(list(document_names))
        except Exception:
            return []
        
    def get_stats_by_document(self) -> dict[str, int]:
        """Get statistics of number of chunks per document.

        Returns:
            Dictionary mapping document names to chunk counts.
        """
        try:
            results = self._collection.get(
                include=['metadatas']
            )
            doc_counts = {}
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    doc_name = metadata.get("document_name", "Unknown")
                    doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            return doc_counts
        except Exception:
            return {}
