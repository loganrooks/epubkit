import pytest
import numpy as np
from pathlib import Path
from epubkit.indexer import (
    IndexConfig,
    FlatIndex,
    ClusterIndex,
    IndexFactory
)

@pytest.fixture
def sample_vectors():
    # Create random vectors with known closest pairs
    vectors = np.random.randn(100, 10).astype(np.float32)
    # Make first two vectors very similar
    vectors[1] = vectors[0] + 0.01 * np.random.randn(10)
    return vectors

@pytest.fixture
def index_config():
    return IndexConfig(
        metric="cosine",
        num_clusters=10
    )

def test_flat_index_cosine(sample_vectors, index_config):
    index = FlatIndex(index_config)
    index.build_index(sample_vectors)
    
    # Search with first vector as query
    query = sample_vectors[0]
    scores, indices = index.search(query, k=2)
    
    # First vector should match itself best, then the similar second vector
    assert indices[0] == 0
    assert indices[1] == 1

def test_flat_index_euclidean(sample_vectors):
    config = IndexConfig(metric="euclidean")
    index = FlatIndex(config)
    index.build_index(sample_vectors)
    
    query = sample_vectors[0]
    scores, indices = index.search(query, k=2)
    
    assert indices[0] == 0  # Should match itself first
    assert indices[1] == 1  # Then the similar vector

def test_cluster_index(sample_vectors, index_config):
    index = ClusterIndex(index_config)
    index.build_index(sample_vectors)
    
    # Check cluster assignments
    assert len(index._clusters) == len(sample_vectors)
    assert len(index._centroids) == index_config.num_clusters
    
    # Test search
    query = sample_vectors[0]
    scores, indices = index.search(query, k=2)
    
    assert len(scores) == 2
    assert len(indices) == 2
    assert 0 in indices  # Should find the query vector

def test_index_save_load(sample_vectors, index_config, tmp_path):
    # Test FlatIndex
    flat_index = FlatIndex(index_config)
    flat_index.build_index(sample_vectors)
    
    flat_path = tmp_path / "flat_index.npy"
    flat_index.save(str(flat_path))
    
    loaded_flat = FlatIndex.load(str(flat_path))
    np.testing.assert_array_equal(
        flat_index._vectors,
        loaded_flat._vectors
    )
    
    # Test ClusterIndex
    cluster_index = ClusterIndex(index_config)
    cluster_index.build_index(sample_vectors)
    
    cluster_path = tmp_path / "cluster_index.npz"
    cluster_index.save(str(cluster_path))
    
    loaded_cluster = ClusterIndex.load(str(cluster_path))
    np.testing.assert_array_equal(
        cluster_index._vectors,
        loaded_cluster._vectors
    )
    np.testing.assert_array_equal(
        cluster_index._clusters,
        loaded_cluster._clusters
    )

def test_index_factory():
    # Test registration
    class DummyIndex(FlatIndex):
        pass
    
    IndexFactory.register_index('dummy', DummyIndex)
    assert 'dummy' in IndexFactory._index_types
    
    # Test creation
    config = IndexConfig()
    index = IndexFactory.create('dummy', config)
    assert isinstance(index, DummyIndex)
    
    # Test invalid type
    with pytest.raises(ValueError):
        IndexFactory.create('invalid')

@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_index_metrics(metric, sample_vectors):
    config = IndexConfig(metric=metric)
    
    # Test with different index types
    for index_type in ['flat', 'cluster']:
        index = IndexFactory.create(index_type, config)
        index.build_index(sample_vectors)
        
        query = sample_vectors[0]
        scores, indices = index.search(query, k=1)
        
        assert len(scores) == 1
        assert len(indices) == 1
        assert indices[0] == 0  # Should match query vector
