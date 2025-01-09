from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, TruncatedSVD
from umap import UMAP  
from typing import Dict, List, Optional, Tuple, Any, Protocol, Literal
import numpy.typing as npt
from pathlib import Path

DimReductionType = Literal['tsne', 'umap', 'pca', 'mds', 'svd']
ColorScheme = Literal['viridis', 'plasma', 'inferno', 'magma', 'category20']

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    n_components: int = 2  # 1, 2, or 3
    dim_reduction: DimReductionType = 'tsne'
    color_scheme: ColorScheme = 'viridis'
    point_size: int = 5
    opacity: float = 0.7
    show_labels: bool = True
    label_size: int = 8
    background_color: str = 'white'
    plot_title: str = ''
    extra_params: Dict[str, Any] = field(default_factory=dict)

class VectorVisualizer(ABC):
    """Base class for vector space visualizations"""
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._dim_reducers = {
            'tsne': lambda n: TSNE(n_components=n, **self.config.extra_params),
            'umap': lambda n: UMAP(n_components=n, **self.config.extra_params),
            'pca': lambda n: PCA(n_components=n),
            'mds': lambda n: MDS(n_components=n, **self.config.extra_params),
            'svd': lambda n: TruncatedSVD(n_components=n)
        }
        
    def reduce_dimensions(
        self, 
        vectors: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Reduce vectors to target dimensionality"""
        reducer = self._dim_reducers[self.config.dim_reduction](self.config.n_components)
        return reducer.fit_transform(vectors)
    
    @abstractmethod
    def visualize(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> go.Figure:
        """Create visualization"""
        pass
    
    def save(self, fig: go.Figure, path: Path) -> None:
        """Save visualization"""
        fig.write_html(str(path))

class EmbeddingVisualizer(VectorVisualizer):
    """Visualize embedding vector spaces"""
    def visualize(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> go.Figure:
        # Reduce dimensions
        reduced = self.reduce_dimensions(vectors)
        
        # Create figure based on dimensions
        if self.config.n_components == 1:
            return self._create_1d_plot(reduced, labels, colors, categories)
        elif self.config.n_components == 2:
            return self._create_2d_plot(reduced, labels, colors, categories)
        else:
            return self._create_3d_plot(reduced, labels, colors, categories)
    
    def _create_1d_plot(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]],
        colors: Optional[List[str]],
        categories: Optional[List[str]]
    ) -> go.Figure:
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter(
            x=vectors.flatten(),
            y=np.zeros_like(vectors.flatten()),
            mode='markers',
            marker=dict(
                size=self.config.point_size,
                color=colors or vectors.flatten(),
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity
            ),
            text=labels,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=self.config.plot_title,
            plot_bgcolor=self.config.background_color,
            showlegend=bool(categories)
        )
        
        return fig
    
    def _create_2d_plot(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]],
        colors: Optional[List[str]],
        categories: Optional[List[str]]
    ) -> go.Figure:
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=vectors[:, 0],
            y=vectors[:, 1],
            mode='markers+text' if self.config.show_labels else 'markers',
            marker=dict(
                size=self.config.point_size,
                color=colors or vectors[:, 0],
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=self.config.label_size),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=self.config.plot_title,
            plot_bgcolor=self.config.background_color,
            showlegend=bool(categories)
        )
        
        return fig
    
    def _create_3d_plot(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]],
        colors: Optional[List[str]],
        categories: Optional[List[str]]
    ) -> go.Figure:
        fig = go.Figure()
        
        # Add 3D scatter points
        fig.add_trace(go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode='markers+text' if self.config.show_labels else 'markers',
            marker=dict(
                size=self.config.point_size,
                color=colors or vectors[:, 0],
                colorscale=self.config.color_scheme,
                opacity=self.config.opacity
            ),
            text=labels,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=self.config.plot_title,
            scene=dict(
                bgcolor=self.config.background_color
            ),
            showlegend=bool(categories)
        )
        
        return fig

class IndexVisualizer(VectorVisualizer):
    """Visualize index structures and nearest neighbors"""
    def visualize(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        connections: Optional[List[Tuple[int, int]]] = None,
        highlighted_indices: Optional[List[int]] = None
    ) -> go.Figure:
        # Reduce dimensions
        reduced = self.reduce_dimensions(vectors)
        
        # Create base plot
        if self.config.n_components == 2:
            fig = self._create_2d_index_plot(
                reduced, labels, colors, categories, 
                connections, highlighted_indices
            )
        else:
            fig = self._create_3d_index_plot(
                reduced, labels, colors, categories,
                connections, highlighted_indices
            )
            
        return fig
    
    def _create_2d_index_plot(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]],
        colors: Optional[List[str]],
        categories: Optional[List[str]],
        connections: Optional[List[Tuple[int, int]]],
        highlighted_indices: Optional[List[int]]
    ) -> go.Figure:
        fig = go.Figure()
        
        # Add connections first (if any)
        if connections:
            for start_idx, end_idx in connections:
                fig.add_trace(go.Scatter(
                    x=[vectors[start_idx, 0], vectors[end_idx, 0]],
                    y=[vectors[start_idx, 1], vectors[end_idx, 1]],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.3)'),
                    showlegend=False
                ))
        
        # Add regular points
        mask = np.ones(len(vectors), dtype=bool)
        if highlighted_indices:
            mask[highlighted_indices] = False
            
        fig.add_trace(go.Scatter(
            x=vectors[mask, 0],
            y=vectors[mask, 1],
            mode='markers',
            marker=dict(
                size=self.config.point_size,
                color=colors[mask] if colors else 'blue',
                opacity=self.config.opacity
            ),
            text=[labels[i] if labels else None for i in range(len(vectors)) if mask[i]],
            name='Points'
        ))
        
        # Add highlighted points
        if highlighted_indices:
            fig.add_trace(go.Scatter(
                x=vectors[highlighted_indices, 0],
                y=vectors[highlighted_indices, 1],
                mode='markers',
                marker=dict(
                    size=self.config.point_size * 1.5,
                    color='red',
                    symbol='star'
                ),
                text=[labels[i] if labels else None for i in highlighted_indices],
                name='Highlighted'
            ))
        
        fig.update_layout(
            title=self.config.plot_title,
            plot_bgcolor=self.config.background_color,
            showlegend=True
        )
        
        return fig
    
    def _create_3d_index_plot(
        self,
        vectors: npt.NDArray[np.float32],
        labels: Optional[List[str]],
        colors: Optional[List[str]],
        categories: Optional[List[str]],
        connections: Optional[List[Tuple[int, int]]],
        highlighted_indices: Optional[List[int]]
    ) -> go.Figure:
        fig = go.Figure()
        
        # Add connections
        if connections:
            for start_idx, end_idx in connections:
                fig.add_trace(go.Scatter3d(
                    x=[vectors[start_idx, 0], vectors[end_idx, 0]],
                    y=[vectors[start_idx, 1], vectors[end_idx, 1]],
                    z=[vectors[start_idx, 2], vectors[end_idx, 2]],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.3)'),
                    showlegend=False
                ))
        
        # Add regular points
        mask = np.ones(len(vectors), dtype=bool)
        if highlighted_indices:
            mask[highlighted_indices] = False
            
        fig.add_trace(go.Scatter3d(
            x=vectors[mask, 0],
            y=vectors[mask, 1],
            z=vectors[mask, 2],
            mode='markers',
            marker=dict(
                size=self.config.point_size,
                color=colors[mask] if colors else 'blue',
                opacity=self.config.opacity
            ),
            text=[labels[i] if labels else None for i in range(len(vectors)) if mask[i]],
            name='Points'
        ))
        
        # Add highlighted points
        if highlighted_indices:
            fig.add_trace(go.Scatter3d(
                x=vectors[highlighted_indices, 0],
                y=vectors[highlighted_indices, 1],
                z=vectors[highlighted_indices, 2],
                mode='markers',
                marker=dict(
                    size=self.config.point_size * 1.5,
                    color='red',
                    symbol='diamond'
                ),
                text=[labels[i] if labels else None for i in highlighted_indices],
                name='Highlighted'
            ))
        
        fig.update_layout(
            title=self.config.plot_title,
            scene=dict(
                bgcolor=self.config.background_color
            ),
            showlegend=True
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    vectors = np.random.randn(100, 50)  # 100 50-dimensional vectors
    labels = [f"Point {i}" for i in range(100)]
    categories = ["A"] * 50 + ["B"] * 50
    
    # Visualize embeddings
    config = VisualizationConfig(
        n_components=3,
        dim_reduction='umap',
        color_scheme='plasma',
        point_size=8,
        plot_title='Sample Embeddings'
    )
    
    viz = EmbeddingVisualizer(config)
    fig = viz.visualize(vectors, labels=labels, categories=categories)
    viz.save(fig, Path("embeddings_viz.html"))
    
    # Visualize index structure
    config.n_components = 2
    viz = IndexVisualizer(config)
    
    # Simulate some connections and highlights
    connections = [(0, 1), (1, 2), (2, 3)]  # Example edges
    highlighted = [0, 3]  # Example points to highlight
    
    fig = viz.visualize(
        vectors,
        labels=labels,
        categories=categories,
        connections=connections,
        highlighted_indices=highlighted
    )
    viz.save(fig, Path("index_viz.html"))
