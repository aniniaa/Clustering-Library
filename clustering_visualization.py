"""
Visualization utilities for clustering results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
import warnings


class ClusterVisualizer:
    """Class for visualizing clustering results and analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
            'gray', 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime'
        ]
    
    def plot_silhouette_analysis(self, k_values: List[int], silhouette_scores: List[float],
                                title: str = "Silhouette Analysis", 
                                save_path: Optional[str] = None,
                                show_best: bool = True) -> None:
        """
        Plot silhouette scores vs number of clusters
        
        Args:
            k_values: List of k values
            silhouette_scores: Corresponding silhouette scores
            title: Plot title
            save_path: Path to save plot (optional)
            show_best: Whether to highlight best k value
        """
        plt.figure(figsize=self.figsize)
        plt.plot(k_values, silhouette_scores, 'o-', linewidth=2, markersize=8, 
                color='skyblue', markerfacecolor='darkblue')
        
        if show_best and silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            plt.axvline(x=k_values[best_idx], color='red', linestyle='--', alpha=0.7,
                       label=f'Best k = {k_values[best_idx]}')
            plt.legend()
        
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Coefficient')
        plt.title(title)
        plt.xticks(k_values)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_elbow_analysis(self, k_values: List[int], wcss_scores: List[float],
                           title: str = "Elbow Analysis (WCSS)",
                           save_path: Optional[str] = None) -> None:
        """
        Plot WCSS scores vs number of clusters (elbow method)
        
        Args:
            k_values: List of k values
            wcss_scores: Corresponding WCSS scores
            title: Plot title
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=self.figsize)
        plt.plot(k_values, wcss_scores, 'o-', linewidth=2, markersize=8,
                color='green', markerfacecolor='darkgreen')
        
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title(title)
        plt.xticks(k_values)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_metrics(self, analysis_results: Dict[str, Any],
                             title: str = "Clustering Analysis",
                             save_path: Optional[str] = None) -> None:
        """
        Plot multiple evaluation metrics in subplots
        
        Args:
            analysis_results: Results from SilhouetteAnalyzer
            title: Overall plot title
            save_path: Path to save plot (optional)
        """
        k_values = analysis_results['k_values']
        silhouette_scores = analysis_results['silhouette_scores']
        wcss_scores = analysis_results['wcss_scores']
        ch_scores = analysis_results['calinski_harabasz_scores']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Silhouette plot
        axes[0].plot(k_values, silhouette_scores, 'o-', color='skyblue', linewidth=2)
        if silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            axes[0].axvline(x=k_values[best_idx], color='red', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Number of clusters (k)')
        axes[0].set_ylabel('Silhouette Coefficient')
        axes[0].set_title('Silhouette Analysis')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(k_values)
        
        # WCSS plot
        axes[1].plot(k_values, wcss_scores, 'o-', color='green', linewidth=2)
        axes[1].set_xlabel('Number of clusters (k)')
        axes[1].set_ylabel('WCSS')
        axes[1].set_title('Elbow Analysis')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(k_values)
        
        # Calinski-Harabasz plot
        axes[2].plot(k_values, ch_scores, 'o-', color='orange', linewidth=2)
        if ch_scores:
            best_idx = np.argmax(ch_scores)
            axes[2].axvline(x=k_values[best_idx], color='red', linestyle='--', alpha=0.7)
        axes[2].set_xlabel('Number of clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Index')
        axes[2].set_title('Calinski-Harabasz Analysis')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(k_values)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_clusters_2d(self, data: np.ndarray, labels: np.ndarray,
                        centroids: Optional[np.ndarray] = None,
                        title: str = "Clustering Results",
                        save_path: Optional[str] = None,
                        show_centroids: bool = True) -> None:
        """
        Plot 2D clustering results
        
        Args:
            data: 2D data points
            labels: Cluster assignments
            centroids: Cluster centroids (optional)
            title: Plot title
            save_path: Path to save plot (optional)
            show_centroids: Whether to show centroids
        """
        if data.shape[1] != 2:
            warnings.warn("Data is not 2D. Only first two dimensions will be plotted.")
            data = data[:, :2]
            if centroids is not None:
                centroids = centroids[:, :2]
        
        plt.figure(figsize=self.figsize)
        
        # Plot points
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = self.colors[i % len(self.colors)]
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=color, label=f'Cluster {label}', alpha=0.7, s=50)
        
        # Plot centroids
        if show_centroids and centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='black', marker='x', s=200, linewidths=3, 
                       label='Centroids')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_clusters_3d(self, data: np.ndarray, labels: np.ndarray,
                        centroids: Optional[np.ndarray] = None,
                        title: str = "3D Clustering Results",
                        save_path: Optional[str] = None) -> None:
        """
        Plot 3D clustering results
        
        Args:
            data: 3D data points
            labels: Cluster assignments
            centroids: Cluster centroids (optional)
            title: Plot title
            save_path: Path to save plot (optional)
        """
        if data.shape[1] < 3:
            raise ValueError("Data must have at least 3 dimensions for 3D plotting")
        
        if data.shape[1] > 3:
            warnings.warn("Data has more than 3 dimensions. Only first three will be plotted.")
            data = data[:, :3]
            if centroids is not None:
                centroids = centroids[:, :3]
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = self.colors[i % len(self.colors)]
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                      c=color, label=f'Cluster {label}', alpha=0.7, s=50)
        
        # Plot centroids
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                      c='black', marker='x', s=200, linewidths=3, 
                      label='Centroids')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_cluster_comparison(self, data: np.ndarray, 
                               results_dict: Dict[str, Dict[str, np.ndarray]],
                               title: str = "Algorithm Comparison",
                               save_path: Optional[str] = None) -> None:
        """
        Compare clustering results from different algorithms
        
        Args:
            data: 2D data points
            results_dict: Dict with algorithm names as keys and results as values
            title: Plot title
            save_path: Path to save plot (optional)
        """
        if data.shape[1] != 2:
            warnings.warn("Data is not 2D. Only first two dimensions will be plotted.")
            data = data[:, :2]
        
        n_algorithms = len(results_dict)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(5*n_algorithms, 5))
        
        if n_algorithms == 1:
            axes = [axes]
        
        for i, (alg_name, results) in enumerate(results_dict.items()):
            labels = results['labels']
            centroids = results.get('centroids', None)
            
            # Plot points
            unique_labels = np.unique(labels)
            for j, label in enumerate(unique_labels):
                mask = labels == label
                color = self.colors[j % len(self.colors)]
                axes[i].scatter(data[mask, 0], data[mask, 1], 
                               c=color, alpha=0.7, s=50)
            
            # Plot centroids
            if centroids is not None and centroids.shape[1] >= 2:
                axes[i].scatter(centroids[:, 0], centroids[:, 1], 
                               c='black', marker='x', s=200, linewidths=3)
            
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
            axes[i].set_title(alg_name)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_dendrogram(self, hierarchy: List[Dict], max_levels: int = 5,
                       title: str = "Hierarchical Clustering Dendrogram",
                       save_path: Optional[str] = None) -> None:
        """
        Plot dendrogram for hierarchical clustering (simplified version)
        
        Args:
            hierarchy: Hierarchy from BisectingKMeans
            max_levels: Maximum levels to show
            title: Plot title
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        levels = min(len(hierarchy), max_levels)
        x_positions = range(1, levels + 1)
        n_clusters = [i + 1 for i in range(levels)]
        
        plt.plot(x_positions, n_clusters, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Hierarchy Level')
        plt.ylabel('Number of Clusters')
        plt.title(title)
        plt.xticks(x_positions)
        plt.yticks(n_clusters)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()