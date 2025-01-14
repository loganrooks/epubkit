from __future__ import annotations
import abc
import torch  # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
from typing import Callable, Dict, List, Type, Any, Protocol, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
import numpy.typing as npt
from tqdm.auto import tqdm
from gensim.models import KeyedVectors 
from sentence_transformers import SentenceTransformer  # type: ignore
import nltk
from nltk.tokenize import word_tokenize
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer 
import torch.nn.functional as F
import transformers  # type: ignore

# Download NLTK data
nltk.download('punkt', quiet=True)

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""
    def embed_text(self, text: str) -> npt.NDArray[np.float32]: ...
    def embed_texts(self, texts: List[str]) -> npt.NDArray[np.float32]: ...
    @property
    def embedding_dim(self) -> int: ...

class BaseEmbeddingProvider(abc.ABC):
    """Abstract base class for embedding providers"""
    @abc.abstractmethod
    def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text"""
        pass
        
    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """Embed multiple texts"""
        pass
        
    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings"""
        pass

@dataclass
class TrainingConfig:
    """Configuration for embedder training"""
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 10
    validation_split: float = 0.1
    context_window: int = 5
    min_word_freq: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)

class TrainableEmbedder(Protocol):
    """Protocol for embedders that can be trained"""
    def train(self, texts: List[str], config: TrainingConfig) -> Dict[str, float]: ...
    def save_model(self, path: Path) -> None: ...
    @classmethod
    def load_model(cls, path: Path) -> 'TrainableEmbedder': ...

class BaseTrainableEmbedder(BaseEmbeddingProvider):
    """Base class for trainable embedders"""
    def train(self, texts: List[str], config: TrainingConfig) -> Dict[str, float]:
        """Train the embedder on texts"""
        raise NotImplementedError
        
    def save_model(self, path: Path) -> None:
        """Save model weights"""
        raise NotImplementedError
        
    @classmethod
    def load_model(cls, path: Path) -> 'BaseTrainableEmbedder':
        """Load model weights"""
        raise NotImplementedError

class GloVeEmbedder(BaseTrainableEmbedder):
    """GloVe embeddings with training capability"""
    def __init__(
        self, 
        embedding_dim: int = 300,
        vocab_size: Optional[int] = None,
        pretrained_path: Optional[Path] = None
    ):
        self._dim = embedding_dim
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.cooccurrence: Optional[torch.Tensor] = None
        
        # Model components
        self.word_embeddings: Optional[nn.Embedding] = None
        self.context_embeddings: Optional[nn.Embedding] = None
        self.word_biases: Optional[nn.Parameter] = None
        self.context_biases: Optional[nn.Parameter] = None
        
        if pretrained_path:
            self.load_model(pretrained_path)
    
    def _build_vocab(self, texts: List[str], min_freq: int) -> None:
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
            
        # Filter by frequency and limit vocab size
        vocab = [word for word, count in word_counts.most_common() 
                if count >= min_freq]
        if self.vocab_size:
            vocab = vocab[:self.vocab_size]
            
        # Create mappings
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        
    def _build_cooccurrence(self, texts: List[str], window_size: int) -> None:
        """Build cooccurrence matrix"""
        vocab_size = len(self.word2idx)
        cooccurrence = torch.zeros((vocab_size, vocab_size))
        
        for text in tqdm(texts, desc="Building cooccurrence matrix"):
            words = text.lower().split()
            word_indices = [self.word2idx.get(word) for word in words]
            word_indices = [idx for idx in word_indices if idx is not None]
            
            for center_idx, center_word_idx in enumerate(word_indices):
                context_start = max(0, center_idx - window_size)
                context_end = min(len(word_indices), center_idx + window_size + 1)
                
                for context_idx in range(context_start, context_end):
                    if context_idx != center_idx:
                        context_word_idx = word_indices[context_idx]
                        distance = abs(context_idx - center_idx)
                        weight = 1 / distance
                        cooccurrence[center_word_idx, context_word_idx] += weight
        
        self.cooccurrence = cooccurrence
        
    def _init_model(self) -> None:
        """Initialize model components"""
        vocab_size = len(self.word2idx)
        self.word_embeddings = nn.Embedding(vocab_size, self._dim)
        self.context_embeddings = nn.Embedding(vocab_size, self._dim)
        self.word_biases = nn.Parameter(torch.zeros(vocab_size))
        self.context_biases = nn.Parameter(torch.zeros(vocab_size))
    
    def train(self, texts: List[str], config: TrainingConfig) -> Dict[str, float]:
        """Train GloVe embeddings"""
        # Build vocabulary and cooccurrence matrix
        self._build_vocab(texts, config.min_word_freq)
        self._build_cooccurrence(texts, config.context_window)
        self._init_model()
        
        # Prepare for training
        optimizer = torch.optim.Adam(
            [
                self.word_embeddings.weight,
                self.context_embeddings.weight,
                self.word_biases,
                self.context_biases
            ],
            lr=config.learning_rate
        )
        
        # Training loop
        losses = []
        for epoch in range(config.num_epochs):
            total_loss = 0
            
            # Create batches
            nonzero = torch.nonzero(self.cooccurrence)
            num_nonzero = len(nonzero)
            indices = torch.randperm(num_nonzero)
            
            for start_idx in range(0, num_nonzero, config.batch_size):
                batch_indices = indices[start_idx:start_idx + config.batch_size]
                i = nonzero[batch_indices, 0]
                j = nonzero[batch_indices, 1]
                Xij = self.cooccurrence[i, j]
                
                # Forward pass
                wi = self.word_embeddings(i)
                wj = self.context_embeddings(j)
                bi = self.word_biases[i]
                bj = self.context_biases[j]
                
                # GloVe loss
                weight_factor = torch.minimum(
                    torch.pow(Xij/100, 0.75),
                    torch.ones_like(Xij)
                )
                loss = weight_factor * torch.pow(
                    wi @ wj.transpose(-1, -2) + bi + bj - torch.log(Xij + 1),
                    2
                ).sum()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_nonzero
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
            
        return {"loss": losses[-1], "all_losses": losses}
    
    def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        """Get embedding for a single text"""
        if not self.word_embeddings:
            raise ValueError("Model not trained or loaded")
            
        words = text.lower().split()
        word_indices = [self.word2idx.get(word, 0) for word in words]
        embeddings = self.word_embeddings(torch.tensor(word_indices))
        return embeddings.mean(dim=0).detach().numpy()
    
    def embed_texts(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """Get embeddings for multiple texts"""
        return np.stack([self.embed_text(text) for text in texts])
    
    @property
    def embedding_dim(self) -> int:
        return self._dim
        
    def save_model(self, path: Path) -> None:
        """Save model state"""
        state = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_embeddings': self.word_embeddings.state_dict(),
            'context_embeddings': self.context_embeddings.state_dict(),
            'word_biases': self.word_biases,
            'context_biases': self.context_biases,
            'embedding_dim': self._dim
        }
        torch.save(state, path)
        
    @classmethod
    def load_model(cls, path: Path) -> 'GloVeEmbedder':
        """Load model state"""
        state = torch.load(path)
        instance = cls(embedding_dim=state['embedding_dim'])
        instance.word2idx = state['word2idx']
        instance.idx2word = state['idx2word']
        
        # Initialize and load model components
        instance._init_model()
        instance.word_embeddings.load_state_dict(state['word_embeddings'])
        instance.context_embeddings.load_state_dict(state['context_embeddings'])
        instance.word_biases = state['word_biases']
        instance.context_biases = state['context_biases']
        
        return instance

class OpenAIEmbedder(BaseTrainableEmbedder):
    """OpenAI embeddings with fine-tuning support"""
    def __init__(self, model: str = "text-embedding-ada-002", fine_tuned_model: Optional[str] = None):
        self.client = OpenAI()
        self.base_model = model
        self.model = fine_tuned_model or model
        self._dim = 1536
        
    def train(self, texts: List[str], config: TrainingConfig) -> Dict[str, float]:
        """Fine-tune embeddings using OpenAI API"""
        # Prepare training data in OpenAI format
        training_data = [
            {"text": text} for text in texts
        ]
        
        # Create fine-tuning job
        try:
            response = self.client.fine_tunes.create(
                model=self.base_model,
                training_data=training_data,
                **config.extra_params
            )
            
            self.model = response.fine_tuned_model
            return {"model_id": self.model}
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            return {"error": str(e)}
            
    def save_model(self, path: Path) -> None:
        """Save model identifier"""
        with open(path, 'w') as f:
            f.write(self.model)
            
    @classmethod
    def load_model(cls, path: Path) -> 'OpenAIEmbedder':
        """Load fine-tuned model identifier"""
        with open(path) as f:
            model_id = f.read().strip()
        return cls(fine_tuned_model=model_id)

class HuggingFaceEmbedder(BaseTrainableEmbedder):
    """HuggingFace model-based embeddings with fine-tuning support"""
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        pooling_strategy: str = "mean"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dim = self.model.config.hidden_size
        
    def _pool_embeddings(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embedding"""
        if self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            return token_embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def embed_text(self, text: str) -> npt.NDArray[np.float32]:
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
            
        return F.normalize(embeddings, p=2, dim=1).numpy()
    
    def embed_texts(self, texts: List[str]) -> npt.NDArray[np.float32]:
        # Batch process texts
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
            
        return F.normalize(embeddings, p=2, dim=1).numpy()
    
    def train(self, texts: List[str], config: TrainingConfig) -> Dict[str, float]:
        """Fine-tune the model on texts"""
        # Setup training configurations
        training_args = transformers.TrainingArguments(
            output_dir="./results",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            **config.extra_params
        )
        
        # Prepare dataset
        dataset = self._prepare_dataset(texts)
        
        # Train
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        result = trainer.train()
        return {"loss": result.training_loss}
    
    def save_model(self, path: Path) -> None:
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def load_model(cls, path: Path) -> 'HuggingFaceEmbedder':
        """Load saved model"""
        instance = cls()
        instance.model = AutoModel.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        return instance
    
    @property
    def embedding_dim(self) -> int:
        return self._dim

class EmbeddingFactory:
    """Factory for creating embedding providers"""
    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {
        'openai': OpenAIEmbedder,
        'glove': GloVeEmbedder,
        'huggingface': HuggingFaceEmbedder
    }
    
    @classmethod
    def register_provider(
        cls, 
        name: str, 
        provider: Type[BaseEmbeddingProvider]
    ) -> None:
        """Register a new embedding provider"""
        cls._providers[name] = provider
    
    @classmethod
    def create(
        cls, 
        provider: str,
        **kwargs: Any
    ) -> BaseEmbeddingProvider:
        """Create embedding provider instance"""
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")
            
        return cls._providers[provider](**kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEmbeddingProvider:
        """Create provider from configuration dictionary"""
        provider = config.pop("provider")
        return cls.create(provider, **config)

# Example usage:
if __name__ == "__main__":
    texts = [
        "Here is some sample text",
        "More text for training",
        # ...more texts...
    ]
    
    # # Train GloVe embedder
    # glove = GloVeEmbedder(embedding_dim=100)
    # results = glove.train(texts, TrainingConfig(
    #     num_epochs=5,
    #     batch_size=32
    # ))
    # glove.save_model(Path("glove_model.pt"))
    
    # Fine-tune OpenAI embedder
    openai = OpenAIEmbedder()
    results = openai.train(texts, TrainingConfig(
        extra_params={
            "n_epochs": 3,
            "batch_size": 32
        }
    ))
    
    # Create HuggingFace embedder
    hf_embedder = EmbeddingFactory.create(
        'huggingface',
        model_name='bert-base-uncased',
        pooling_strategy='cls'
    )
    
    # Register custom provider
    class CustomEmbedder(BaseEmbeddingProvider):
        # Implementation...
        pass
    
    EmbeddingFactory.register_provider('custom', CustomEmbedder)
    
    # Create from config
    config = {
        "provider": "huggingface",
        "model_name": "bert-base-uncased",
        "pooling_strategy": "mean"
    }
    embedder = EmbeddingFactory.from_config(config)
