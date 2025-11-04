import chromadb
from typing import Any
from chromadb.config import Settings
from openai import OpenAI
from ... import logger
from ...domain.entities import DocumentChunk, RetrievalResult
from ...domain.repositories import IVectorStore, VectorStoreError


class ChromaDBStore(IVectorStore):
    """ChromaDB implementation of the vector store interface.

    This class handles storing document embeddings and retrieving relevant
    chunks using ChromaDB and OpenAI embeddings for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "financial_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str = None
    ):
        """Initialize the ChromaDB store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
            embedding_model: Name of the OpenAI embedding model.
            openai_api_key: OpenAI API key for embeddings.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Initialize the OpenAI client for embeddings
        if not openai_api_key:
            import os
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self._embedding_client = OpenAI(api_key=openai_api_key)

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

    def _generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """Generate embeddings using OpenAI's API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            response = self._embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise VectorStoreError(
                f"Failed to generate embeddings: {str(e)}"
            ) from e

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get the existing collection
            collection = self._client.get_collection(self.collection_name)

            # Check if the collection has the expected embedding dimension
            # We can detect dimension mismatch when we try to add embeddings
            return collection
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

                # Add financial data if available (serialized as JSON)
                if chunk.financial_data and chunk.financial_data.data_points:
                    import json
                    financial_data_dict = {
                        "data_points": [
                            {
                                "metric_type": dp.metric_type,
                                "value": dp.value,
                                "period": dp.period,
                                "currency": dp.currency,
                                "confidence": dp.confidence,
                                "raw_text": dp.raw_text
                            }
                            for dp in chunk.financial_data.data_points
                        ],
                        "document_name": chunk.financial_data.document_name,
                        "extraction_method": chunk.financial_data.extraction_method,
                        "confidence_score": chunk.financial_data.confidence_score,
                        "extraction_timestamp": chunk.financial_data.extraction_timestamp
                    }
                    metadata["financial_data"] = json.dumps(financial_data_dict)

                # Add any additional metadata
                if chunk.metadata:
                    metadata.update(chunk.metadata)

                metadatas.append(metadata)

            # Generate embeddings using OpenAI
            embeddings = self._generate_embeddings(documents)

            # Add to chromaDB
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            # Check if this is an embedding dimension mismatch error
            error_msg = str(e).lower()
            if ("expecting embedding with dimension" in error_msg and
                "got" in error_msg):
                # Reset the collection and retry
                logger.warning("Embedding dimension mismatch detected. Resetting vector store collection...")
                self.reset_collection()

                # Retry adding documents
                logger.info("Retrying document addition with new collection...")
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                logger.info("Documents added successfully after collection reset.")
            else:
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
            # Generate query embedding using OpenAI
            query_embedding = self._generate_embeddings([query])

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

                    # Reconstruct financial_data if available
                    financial_data = None
                    if "financial_data" in metadata:
                        import json
                        from ...domain.entities import FinancialDataPoint, StructuredFinancialData
                        financial_data_dict = json.loads(metadata["financial_data"])
                        data_points = [
                            FinancialDataPoint(
                                metric_type=dp["metric_type"],
                                value=dp["value"],
                                period=dp["period"],
                                currency=dp["currency"],
                                confidence=dp["confidence"],
                                raw_text=dp["raw_text"]
                            )
                            for dp in financial_data_dict["data_points"]
                        ]
                        financial_data = StructuredFinancialData(
                            data_points=data_points,
                            document_name=financial_data_dict["document_name"],
                            extraction_method=financial_data_dict["extraction_method"],
                            confidence_score=financial_data_dict["confidence_score"],
                            extraction_timestamp=financial_data_dict["extraction_timestamp"]
                        )

                    chunk = DocumentChunk(
                        chunk_id=doc_id,
                        document_name=metadata.get("document_name", "unknown"),
                        content=document_text,
                        page_number=metadata.get("page_number"),
                        chunk_index=metadata.get("chunk_index"),
                        financial_data=financial_data,
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
