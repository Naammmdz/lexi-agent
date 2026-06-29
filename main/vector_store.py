import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import numpy as np
from config import Config
from tqdm import tqdm


class QdrantVectorStore:
    """QDrant vector store for legal document embeddings"""

    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.collection_name = Config.COLLECTION_NAME
        self._initialize_client()
        self._initialize_embedding_model()

    def _initialize_client(self):
        """Initialize QDrant client"""
        try:
            if Config.QDRANT_URL and Config.QDRANT_API_KEY:
                # Cloud QDrant
                self.client = QdrantClient(
                    url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY
                )
            else:
                # Local QDrant (fallback)
                self.client = QdrantClient(host="localhost", port=6333)

            print("QDrant client initialized successfully")
        except Exception as e:
            print(f"Error initializing QDrant client: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize embedding model (local GPU/MPS or remote API)."""
        backend = Config.EMBEDDING_BACKEND
        if backend == "api":
            from main.api_embedder import ApiEmbeddingEncoder

            self.embedding_model = ApiEmbeddingEncoder()
            print(
                f"Embedding API {Config.EMBEDDING_MODEL} @ {Config.EMBED_API_BASE_URL}",
                flush=True,
            )
            return

        try:
            from sentence_transformers import SentenceTransformer

            import torch

            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self.embedding_model = SentenceTransformer(
                Config.EMBEDDING_MODEL, device=device
            )
            print(
                f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully on {device}",
                flush=True,
            )
        except UnicodeDecodeError as e:
            print(f"Encoding error loading embedding model: {e}")
            print("Trying to clear sentence-transformers cache...")
            try:
                import shutil
                import tempfile

                from sentence_transformers import SentenceTransformer

                cache_dir = os.path.join(tempfile.gettempdir(), "sentence_transformers")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("Cache cleared, retrying...")

                import torch

                device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                self.embedding_model = SentenceTransformer(
                    Config.EMBEDDING_MODEL, device=device
                )
                print(
                    f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully on {device} after cache clear",
                    flush=True,
                )
            except Exception as retry_e:
                print(
                    f"Failed to load embedding model even after cache clear: {retry_e}"
                )
                raise
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def create_collection(self, vector_size: int = 384, force_recreate: bool = False):
        """Create collection in QDrant"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )
            print(f"Collection exists: {collection_exists}")

            if collection_exists:
                if force_recreate:
                    print(f"Force recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                    # Create collection
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        ),
                    )
                    print(f"Successfully created collection: {self.collection_name}")
                else:
                    print(
                        f"Collection {self.collection_name} already exists - skipping creation"
                    )
                    return
            else:
                print(
                    f"Collection {self.collection_name} does not exist - creating new collection"
                )
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                print(f"Successfully created collection: {self.collection_name}")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            if hasattr(embedding, "tolist"):
                return embedding.tolist()
            return list(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def add_documents(self, documents: List[Dict[str, Any]], embed_batch_size: int | None = None):
        """Add documents to vector store (batched encode + upsert)."""
        if embed_batch_size is None:
            embed_batch_size = int(os.getenv("INDEX_EMBED_BATCH_SIZE", "64"))
        upsert_batch_size = int(os.getenv("INDEX_UPSERT_BATCH_SIZE", "256"))

        try:
            prepared: list[tuple[Dict[str, Any], str]] = []
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    prepared.append((doc, content))

            total = len(prepared)
            print(
                f"Embedding {total:,} articles "
                f"(embed_batch={embed_batch_size}, upsert_batch={upsert_batch_size})...",
                flush=True,
            )

            uploaded = 0
            for start in tqdm(range(0, total, embed_batch_size), desc="embed batches"):
                chunk = prepared[start : start + embed_batch_size]
                texts = [c[1] for c in chunk]
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=embed_batch_size,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                )

                points: list[PointStruct] = []
                for (doc, content), embedding in zip(chunk, embeddings):
                    vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                    if not vec:
                        continue
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "article_id": doc.get("id", ""),
                                "title": doc.get("title", ""),
                                "content": content,
                                "metadata": doc.get("metadata", {}),
                            },
                        )
                    )

                for i in range(0, len(points), upsert_batch_size):
                    batch = points[i : i + upsert_batch_size]
                    self.client.upsert(collection_name=self.collection_name, points=batch)
                    uploaded += len(batch)

            print(f"Successfully added {uploaded} documents to vector store")

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def search_similar_documents(
        self, query: str, top_k: int = None, score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        if score_threshold is None:
            score_threshold = Config.SIMILARITY_THRESHOLD

        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []

            # Search
            search_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )
            search_results = search_response.points

            # Format results
            scores = []
            for result in search_results:
                scores.append(result.score)
            
            max_score = max(scores)
            min_score = min(scores)
            
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.payload.get("article_id", ""),
                        "title": result.payload.get("title", ""),
                        "content": result.payload.get("content", ""),
                        "score": float((result.score - min_score) / (max_score - min_score)) if max_score > 0 and min_score > 0 and max_score != min_score else 0,
                        "metadata": result.payload.get("metadata", {}),
                    }
                )

            print(f"Vector DB found {len(results)} similar documents")
            return results

        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            # Support Qdrant python client version changes where vectors_count attribute might be absent
            v_count = getattr(info, 'vectors_count', None)
            if v_count is None:
                v_count = getattr(info, 'points_count', 0)
                
            idx_v_count = getattr(info, 'indexed_vectors_count', 0)
            p_count = getattr(info, 'points_count', 0)

            result = {
                "name": self.collection_name,
                "vectors_count": v_count,
                "indexed_vectors_count": idx_v_count,
                "points_count": p_count,
            }
            print(f"Collection info: {result}")
            return result
        except Exception as e:
            print(f"Collection '{self.collection_name}' does not exist: {e}")
            return {}

    def delete_collection(self):
        """Delete collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
