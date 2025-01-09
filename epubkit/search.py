from __future__ import annotations
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, TypedDict, Any
import numpy as np
from openai import OpenAI
from dataclasses import dataclass, field
from .parser import ExtractedText, TextBlock, CategoryType

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

class EmbeddingMetadata(TypedDict):
    category: CategoryType
    header_path: List[str]
    footnotes: List[str]

@dataclass
class DocumentEmbeddings:
    """Stores embeddings and metadata for a document"""
    embeddings: Dict[str, List[float]]  # Maps text to its embedding
    metadata: Dict[str, EmbeddingMetadata]  # Maps text to its metadata
    embedding_model: str
    embedding_dim: int
    
    def save(self, path: Path) -> None:
        """Save embeddings and metadata to files"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array for efficiency
        embeddings_arr = np.array(list(self.embeddings.values()))
        text_keys = list(self.embeddings.keys())
        
        np.save(path / "embeddings.npy", embeddings_arr)
        
        # Save text keys and metadata as JSON
        save_data = {
            "text_keys": text_keys,
            "metadata": self.metadata,
            "model": self.embedding_model,
            "dimensions": self.embedding_dim
        }
        
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> DocumentEmbeddings:
        """Load embeddings and metadata from files"""
        # Load embeddings array
        embeddings_arr = np.load(path / "embeddings.npy")
        
        # Load metadata and text keys
        with open(path / "metadata.json", 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # Reconstruct embeddings dictionary
        embeddings = {
            text: embeddings_arr[i].tolist()
            for i, text in enumerate(save_data["text_keys"])
        }
        
        return cls(
            embeddings=embeddings,
            metadata=save_data["metadata"],
            embedding_model=save_data["model"],
            embedding_dim=save_data["dimensions"]
        )

class TextEmbedder:
    """Handles embedding text using OpenAI API"""
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model
        self.embedding_dim = 1536  # Dimensionality of ada-002 embeddings
        
    def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]

    def embed_extracted_text(self, extracted: ExtractedText) -> DocumentEmbeddings:
        """Create embeddings for all text blocks"""
        # Collect all text blocks
        all_blocks: List[TextBlock] = []
        for category in ['headers', 'subheaders', 'body', 'footnotes', 'toc']:
            blocks = getattr(extracted, category)
            all_blocks.extend(blocks)
        
        # Get embeddings for all texts
        texts = [block.text for block in all_blocks]
        embeddings = self.embed_texts(texts)
        
        # Create metadata dictionary
        metadata = {
            block.text: {
                'category': block.category,
                'header_path': block.header_path,
                'footnotes': block.footnotes
            }
            for block in all_blocks
        }
        
        # Create embeddings dictionary
        embeddings_dict = {
            text: emb for text, emb in zip(texts, embeddings)
        }
        
        return DocumentEmbeddings(
            embeddings=embeddings_dict,
            metadata=metadata,
            embedding_model=self.model,
            embedding_dim=self.embedding_dim
        )

class SemanticSearcher:
    """Performs semantic search over embedded documents"""
    def __init__(self, embeddings: DocumentEmbeddings):
        self.embeddings = embeddings
        self.embedder = TextEmbedder(model=embeddings.embedding_model)
        
        # Convert embeddings to numpy array for efficient search
        self.texts = list(embeddings.embeddings.keys())
        self.embedding_array = np.array([
            embeddings.embeddings[text] for text in self.texts
        ])
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        category_filter: Optional[CategoryType] = None
    ) -> List[dict[str, Any]]:
        """
        Search for most similar texts to query
        
        Args:
            query: Search query
            n_results: Number of results to return
            category_filter: Only return results from this category
            
        Returns:
            List of dicts containing:
                - text: The matching text
                - similarity: Cosine similarity score
                - metadata: Associated metadata
        """
        # Get query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Calculate cosine similarities
        similarities = np.dot(
            self.embedding_array, 
            np.array(query_embedding)
        ) / (
            np.linalg.norm(self.embedding_array, axis=1) * 
            np.linalg.norm(query_embedding)
        )
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            text = self.texts[idx]
            metadata = self.embeddings.metadata[text]
            
            if category_filter and metadata['category'] != category_filter:
                continue
                
            results.append({
                'text': text,
                'similarity': float(similarities[idx]),
                'metadata': metadata
            })
            
            if len(results) >= n_results:
                break
                
        return results

def embed_and_save_extracted_text(
    extracted: ExtractedText,
    output_dir: Path,
    model: str = "text-embedding-ada-002"
) -> Path:
    """
    Create and save embeddings for extracted text
    
    Args:
        extracted: ExtractedText object containing text blocks
        output_dir: Directory to save embeddings
        model: OpenAI embedding model to use
        
    Returns:
        Path to saved embeddings
    """
    embedder = TextEmbedder(model=model)
    embeddings = embedder.embed_extracted_text(extracted)
    
    save_path = output_dir / "embeddings"
    embeddings.save(save_path)
    
    return save_path
