import pytest
import torch # type: ignore
import numpy as np
from pathlib import Path
from epubkit.embedder import (
    TrainingConfig,
    GloVeEmbedder,
    HuggingFaceEmbedder,
    EmbeddingFactory
)

@pytest.fixture
def sample_texts():
    return [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump"
    ]

@pytest.fixture
def small_glove():
    return GloVeEmbedder(
        embedding_dim=50,
        vocab_size=100
    )

def test_training_config_defaults():
    config = TrainingConfig()
    assert config.batch_size == 64
    assert config.learning_rate == 0.001
    assert config.num_epochs == 10
    assert config.context_window == 5

def test_glove_vocab_building(small_glove, sample_texts):
    small_glove._build_vocab(sample_texts, min_freq=1)
    
    # Check vocabulary
    assert len(small_glove.word2idx) > 0
    assert len(small_glove.word2idx) <= 100  # vocab_size limit
    assert 'the' in small_glove.word2idx
    assert small_glove.idx2word[small_glove.word2idx['the']] == 'the'

def test_glove_cooccurrence(small_glove, sample_texts):
    small_glove._build_vocab(sample_texts, min_freq=1)
    small_glove._build_cooccurrence(sample_texts, window_size=2)
    
    assert small_glove.cooccurrence is not None
    assert isinstance(small_glove.cooccurrence, torch.Tensor)
    assert small_glove.cooccurrence.shape == (len(small_glove.word2idx), len(small_glove.word2idx))

def test_glove_model_initialization(small_glove, sample_texts):
    small_glove._build_vocab(sample_texts, min_freq=1)
    small_glove._init_model()
    
    vocab_size = len(small_glove.word2idx)
    assert small_glove.word_embeddings.weight.shape == (vocab_size, 50)
    assert small_glove.context_embeddings.weight.shape == (vocab_size, 50)
    assert small_glove.word_biases.shape == (vocab_size,)

def test_glove_save_load(small_glove, sample_texts, tmp_path):
    # Train minimally
    small_glove.train(
        sample_texts,
        TrainingConfig(num_epochs=1, batch_size=2)
    )
    
    # Save
    save_path = tmp_path / "glove_test.pt"
    small_glove.save_model(save_path)
    assert save_path.exists()
    
    # Load
    loaded = GloVeEmbedder.load_model(save_path)
    assert loaded.embedding_dim == small_glove.embedding_dim
    assert loaded.word2idx == small_glove.word2idx
    
    # Test embeddings
    text = "the quick brown fox"
    original_emb = small_glove.embed_text(text)
    loaded_emb = loaded.embed_text(text)
    np.testing.assert_array_almost_equal(original_emb, loaded_emb)

def test_embedding_factory_registration():
    class DummyEmbedder(GloVeEmbedder):
        pass
    
    EmbeddingFactory.register_provider('dummy', DummyEmbedder)
    assert 'dummy' in EmbeddingFactory.list_providers()
    
    embedder = EmbeddingFactory.create('dummy', embedding_dim=50)
    assert isinstance(embedder, DummyEmbedder)
    assert embedder.embedding_dim == 50

def test_embedding_factory_config():
    config = {
        "provider": "glove",
        "embedding_dim": 100,
        "vocab_size": 1000
    }
    embedder = EmbeddingFactory.from_config(config)
    assert isinstance(embedder, GloVeEmbedder)
    assert embedder.embedding_dim == 100
    assert embedder.vocab_size == 1000

@pytest.mark.parametrize("pooling_strategy", ["mean", "cls"])
def test_huggingface_pooling_strategies(pooling_strategy):
    embedder = HuggingFaceEmbedder(
        model_name="prajjwal1/bert-tiny",  # Very small model for testing
        pooling_strategy=pooling_strategy
    )
    
    # Test single text
    text = "Test sentence"
    embedding = embedder.embed_text(text)
    assert embedding.shape == (1, embedder.embedding_dim)
    
    # Test multiple texts
    texts = ["First sentence", "Second sentence"]
    embeddings = embedder.embed_texts(texts)
    assert embeddings.shape == (2, embedder.embedding_dim)

def test_huggingface_save_load(tmp_path):
    embedder = HuggingFaceEmbedder(
        model_name="prajjwal1/bert-tiny"
    )
    
    # Save
    save_path = tmp_path / "hf_test"
    embedder.save_model(save_path)
    
    # Load
    loaded = HuggingFaceEmbedder.load_model(save_path)
    
    # Test both give same embeddings
    text = "Test sentence"
    original_emb = embedder.embed_text(text)
    loaded_emb = loaded.embed_text(text)
    np.testing.assert_array_almost_equal(original_emb, loaded_emb)
