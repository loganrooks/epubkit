from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass

from epubkit.embedder import BaseTrainableEmbedder, EmbeddingProvider, EmbeddingFactory, TrainingConfig
from epubkit.indexer import IndexConfig, IndexFactory
from epubkit.parser import ExtractedText, CategoryType, TOCEntry

@dataclass
class SearchResult:
    text: str
    similarity: float
    metadata: Dict[str, Any]
    category: CategoryType
    header_path: List[str]

class SemanticSearch:
    """Coordinates embedding and indexing for semantic search"""
    def __init__(
        self,
        embedder: EmbeddingProvider,
        index_type: str = "flat",
        index_config: Optional[IndexConfig] = None
    ):
        self.embedder = embedder
        self.index = IndexFactory.create(index_type, index_config)
        self.texts: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
    @classmethod
    def from_extracted_text(
        cls,
        extracted: ExtractedText,
        embedding_provider: str = "glove",
        index_type: str = "flat",
        index_config: Optional[IndexConfig] = None,
        **kwargs: Any
    ) -> 'SemanticSearch':
        # Create embedder
        embedder = EmbeddingFactory.create(embedding_provider, **kwargs)
        instance = cls(embedder, index_type, index_config)
        
        # Collect texts and metadata
        for category in ['headers', 'subheaders', 'body', 'footnotes', 'toc']:
            blocks = getattr(extracted, category)
            for block in blocks:
                instance.texts.append(block.text)
                instance.metadata[block.text] = {
                    'category': block.category,
                    'header_path': block.header_path,
                    'footnotes': block.footnotes
                }
        
        # Build index
        embeddings = embedder.embed_texts(instance.texts)
        instance.index.build_index(embeddings)
        
        return instance
    
    @classmethod
    def from_extracted_toc(
        cls,
        toc_entries: List[TOCEntry],
        embedding_provider: str = "glove",
        index_type: str = "flat",
        index_config: Optional[IndexConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **kwargs: Any
    ) -> 'SemanticSearch':
        """Initialize search from TOC entries with progress tracking
        
        Args:
            toc_entries: List of top-level TOC entries
            embedding_provider: Name of embedding provider to use
            index_type: Type of index to create
            index_config: Optional index configuration
            progress_callback: Optional callback for progress updates
            **kwargs: Additional args passed to embedding provider
        """
        # Create embedder with progress tracking
        kwargs['progress_callback'] = progress_callback
        embedder = EmbeddingFactory.create(embedding_provider, **kwargs)
        instance = cls(embedder, index_type, index_config)
        
        def process_entry(entry: TOCEntry, path: List[str]) -> None:
            """Recursively process TOC entry and its children"""
            # Add entry's text blocks
            for text in entry.text_blocks:
                instance.texts.append(text)
                instance.metadata[text] = {
                    'category': 'body',
                    'header_path': path + [entry.title],
                    'footnotes': [],
                    'toc_entry': entry.title,
                    'href': entry.href,
                    'level': entry.level
                }
                
            # Process children recursively
            for child in entry.children:
                process_entry(child, path + [entry.title])
        
        # Process entries with progress updates
        if progress_callback:
            progress_callback("Processing TOC entries", 0.0)
            
        for entry in toc_entries:
            process_entry(entry, [])

        if instance.texts:
            # Train if using GloVe embedder
            if isinstance(embedder, BaseTrainableEmbedder) and not embedder.word_embeddings:
                if progress_callback:
                    progress_callback("Training embedder", 0.3)
                    
                config = TrainingConfig(
                    num_epochs=5,
                    batch_size=32,
                    min_word_freq=2
                )
                embedder.train(instance.texts, config)
                
            # Build search index with progress
            if progress_callback:
                progress_callback("Creating embeddings", 0.6)
                
            embeddings = embedder.embed_texts(instance.texts)
            
            if progress_callback:
                progress_callback("Building search index", 0.8)
                
            instance.index.build_index(embeddings)
            
            if progress_callback:
                progress_callback("Complete", 1.0)
        
        return instance
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[CategoryType] = None
    ) -> List[SearchResult]:
        # Get query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k=n_results)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):  # Handle batch dimension
            text = self.texts[idx]
            meta = self.metadata[text]
            
            if category_filter and meta['category'] != category_filter:
                continue
                
            results.append(SearchResult(
                text=text,
                similarity=float(score),
                metadata=meta,
                category=meta['category'],
                header_path=meta['header_path']
            ))
            
        return results
    
    def save(self, path: Path) -> None:
        """Save search state"""
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(str(path / "index"))
        np.save(str(path / "texts.npy"), np.array(self.texts))
        np.save(str(path / "metadata.npy"), self.metadata)
    
    @classmethod
    def load(
        cls,
        path: Path,
        embedding_provider: str = "glove",
        index_type: str = "flat",
        **kwargs: Any
    ) -> 'SemanticSearch':
        """Load saved search state"""
        embedder = EmbeddingFactory.create(embedding_provider, **kwargs)
        instance = cls(embedder, index_type)
        
        # Load saved state
        instance.index = IndexFactory.create(index_type).load(str(path / "index"))
        instance.texts = np.load(str(path / "texts.npy")).tolist()
        instance.metadata = np.load(str(path / "metadata.npy"), allow_pickle=True).item()
        
        return instance
