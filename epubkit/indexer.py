from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Protocol, Type, Optional
import faiss  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
import numpy.typing as npt
from sentence_transformers import CrossEncoder 

@dataclass
class IndexConfig:
    """Configuration for index construction"""
    metric: str = "cosine"  # cosine, euclidean, dot_product
    num_clusters: int = 100  # for cluster-based methods
    use_gpu: bool = False   # for FAISS
    extra_params: Dict[str, Any] = field(default_factory=dict)

class VectorIndex(Protocol):
    """Protocol for vector indexing implementations"""
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None: ...
    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> 'VectorIndex': ...

class BaseVectorIndex(ABC):
    """Abstract base class for vector indexing"""
    def __init__(self, config: IndexConfig):
        self.config = config
        self._index = None
        
    @abstractmethod
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None:
        """Build the index from vectors"""
        pass
        
    @abstractmethod
    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Search the index for nearest neighbors"""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk"""
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseVectorIndex':
        """Load the index from disk"""
        pass

class FlatIndex(BaseVectorIndex):
    """Simple brute-force index using numpy"""
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None:
        self._vectors = vectors
        
    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        if self.config.metric == "cosine":
            # Normalize vectors for cosine similarity
            query_norm = query / np.linalg.norm(query)
            vectors_norm = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            indices = np.argsort(similarities)[::-1][:k]
            return similarities[indices], indices
        else:
            # Euclidean distance
            distances = np.linalg.norm(self._vectors - query, axis=1)
            indices = np.argsort(distances)[:k]
            return distances[indices], indices
            
    def save(self, path: str) -> None:
        np.save(path, self._vectors)
        
    @classmethod
    def load(cls, path: str) -> 'FlatIndex':
        instance = cls(IndexConfig())
        instance._vectors = np.load(path)
        return instance

class FaissIndex(BaseVectorIndex):
    """FAISS-based index"""
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None:
        dim = vectors.shape[1]
        
        if self.config.metric == "cosine":
            # Normalize vectors and use inner product
            faiss.normalize_L2(vectors)
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)
            
        if self.config.use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            
        self._index.add(vectors)
        
    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        if self.config.metric == "cosine":
            faiss.normalize_L2(query.reshape(1, -1))
        return self._index.search(query.reshape(1, -1), k)
        
    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)
        
    @classmethod
    def load(cls, path: str) -> 'FaissIndex':
        instance = cls(IndexConfig())
        instance._index = faiss.read_index(path)
        return instance

class ClusterIndex(BaseVectorIndex):
    """K-means clustering based index"""
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None:
        self._vectors = vectors
        self._kmeans = KMeans(
            n_clusters=self.config.num_clusters,
            **self.config.extra_params
        )
        self._clusters = self._kmeans.fit_predict(vectors)
        self._centroids = self._kmeans.cluster_centers_
        
    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        # Find nearest clusters
        centroid_distances = np.linalg.norm(self._centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:2]  # Check top 2 clusters
        
        # Get points from nearest clusters
        candidate_indices = np.where(np.isin(self._clusters, nearest_clusters))[0]
        candidate_vectors = self._vectors[candidate_indices]
        
        # Find nearest points
        if self.config.metric == "cosine":
            query_norm = query / np.linalg.norm(query)
            vectors_norm = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            top_k = np.argsort(similarities)[::-1][:k]
        else:
            distances = np.linalg.norm(candidate_vectors - query, axis=1)
            top_k = np.argsort(distances)[:k]
            
        return (
            np.array([similarities[i] for i in top_k]) if self.config.metric == "cosine"
            else np.array([distances[i] for i in top_k]),
            candidate_indices[top_k]
        )
        
    def save(self, path: str) -> None:
        np.savez(
            path,
            vectors=self._vectors,
            clusters=self._clusters,
            centroids=self._centroids
        )
        
    @classmethod
    def load(cls, path: str) -> 'ClusterIndex':
        instance = cls(IndexConfig())
        data = np.load(path)
        instance._vectors = data['vectors']
        instance._clusters = data['clusters']
        instance._centroids = data['centroids']
        return instance

class CrossEncoderIndex(BaseVectorIndex):
    """HuggingFace cross-encoder based reranking"""
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self.model_name = config.extra_params.get(
            'model_name', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.cross_encoder = CrossEncoder(self.model_name)
        
    def build_index(self, vectors: npt.NDArray[np.float32]) -> None:
        self._vectors = vectors
        # Use basic index for initial retrieval
        self._base_index = FlatIndex(self.config)
        self._base_index.build_index(vectors)
        
    def search(
        self, 
        query: npt.NDArray[np.float32], 
        k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        # Get initial candidates
        n_candidates = min(k * 10, len(self._vectors))
        _, candidate_indices = self._base_index.search(query, n_candidates)
        
        # Rerank with cross-encoder
        candidate_vectors = self._vectors[candidate_indices]
        scores = self.cross_encoder.predict([
            [query, vec] for vec in candidate_vectors
        ])
        
        # Get top k after reranking
        top_k = np.argsort(scores)[-k:]
        return (
            scores[top_k],
            candidate_indices[top_k]
        )
        
    def save(self, path: str) -> None:
        np.savez(
            path,
            vectors=self._vectors,
            model_name=self.model_name
        )
        
    @classmethod
    def load(cls, path: str) -> 'CrossEncoderIndex':
        data = np.load(path)
        config = IndexConfig(
            extra_params={'model_name': str(data['model_name'])}
        )
        instance = cls(config)
        instance._vectors = data['vectors']
        instance.build_index(instance._vectors)
        return instance

class IndexFactory:
    """Factory for creating vector indices"""
    _index_types: Dict[str, Type[BaseVectorIndex]] = {
        'flat': FlatIndex,
        'faiss': FaissIndex,
        'cluster': ClusterIndex,
        'cross-encoder': CrossEncoderIndex
    }
    
    @classmethod
    def register_index(cls, name: str, index_cls: Type[BaseVectorIndex]) -> None:
        cls._index_types[name] = index_cls
        
    @classmethod
    def create(cls, index_type: str, config: Optional[IndexConfig] = None) -> BaseVectorIndex:
        if index_type not in cls._index_types:
            raise ValueError(f"Unknown index type: {index_type}")
            
        config = config or IndexConfig()
        return cls._index_types[index_type](config)
