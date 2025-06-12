"""
Clustering evaluation metrics module
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .algorithms import ClusteringAlgorithm


class ClusterEvaluator:
    """Class for evaluating clustering performance"""
    
    @staticmethod
    def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    @staticmethod
    def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette coefficient for clustering
        
        Args:
            data: Dataset of shape (n_samples, n_features)
            labels: Cluster assignments for each data point
            
        Returns:
            Silhouette coefficient (higher is better)
        """
        n_points = len(data)
        
        if n_points <= 1:
            return 0.0
        
        # If only one cluster, silhouette is 0
        if len(np.unique(labels)) <= 1:
            return 0.0
        
        silhouette_values = np.zeros(n_points)
        
        for i in range(n_points):
            current_cluster = labels[i]
            cluster_points = [j for j in range(n_points) if labels[j] == current_cluster]
            
            # If point is the only one in its cluster
            if len(cluster_points) <= 1:
                silhouette_values[i] = 0
                continue
            
            # Calculate a_i (mean distance to points in same cluster)
            distances_same_cluster = [
                ClusterEvaluator.compute_distance(data[i], data[j]) 
                for j in cluster_points if j != i
            ]
            a_i = np.mean(distances_same_cluster) if distances_same_cluster else 0
            
            # Calculate b_i (minimum mean distance to points in different clusters)
            b_i = float("inf")
            for cluster_id in set(labels):
                if cluster_id != current_cluster:
                    other_cluster_points = [
                        j for j in range(n_points) if labels[j] == cluster_id
                    ]
                    if other_cluster_points:
                        distances_diff_cluster = [
                            ClusterEvaluator.compute_distance(data[i], data[j])
                            for j in other_cluster_points
                        ]
                        mean_distance = np.mean(distances_diff_cluster)
                        b_i = min(b_i, mean_distance)
            
            # Calculate silhouette value
            if a_i < b_i:
                silhouette_values[i] = 1 - (a_i / b_i)
            elif a_i > b_i:
                silhouette_values[i] = (b_i / a_i) - 1
            else:
                silhouette_values[i] = 0
        
        return np.mean(silhouette_values)
    
    @staticmethod
    def within_cluster_sum_of_squares(data: np.ndarray, labels: np.ndarray, 
                                    centroids: np.ndarray) -> float:
        """
        Calculate Within-Cluster Sum of Squares (WCSS)
        
        Args:
            data: Dataset of shape (n_samples, n_features)
            labels: Cluster assignments
            centroids: Cluster centroids
            
        Returns:
            WCSS value (lower is better)
        """
        wcss = 0.0
        for i, point in enumerate(data):
            cluster_id = labels[i]
            wcss += ClusterEvaluator.compute_distance(point, centroids[cluster_id]) ** 2
        return wcss
    
    @staticmethod
    def between_cluster_sum_of_squares(data: np.ndarray, labels: np.ndarray,
                                     centroids: np.ndarray) -> float:
        """
        Calculate Between-Cluster Sum of Squares (BCSS)
        
        Args:
            data: Dataset of shape (n_samples, n_features)
            labels: Cluster assignments
            centroids: Cluster centroids
            
        Returns:
            BCSS value (higher is better)
        """
        overall_centroid = np.mean(data, axis=0)
        bcss = 0.0
        
        for cluster_id in np.unique(labels):
            cluster_size = np.sum(labels == cluster_id)
            cluster_centroid = centroids[cluster_id]
            bcss += cluster_size * ClusterEvaluator.compute_distance(
                cluster_centroid, overall_centroid
            ) ** 2
        
        return bcss
    
    @staticmethod
    def calinski_harabasz_score(data: np.ndarray, labels: np.ndarray,
                              centroids: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz Index (Variance Ratio Criterion)
        
        Args:
            data: Dataset of shape (n_samples, n_features)
            labels: Cluster assignments
            centroids: Cluster centroids
            
        Returns:
            CH Index (higher is better)
        """
        n_samples = len(data)
        n_clusters = len(np.unique(labels))
        
        if n_clusters == 1:
            return 0.0
        
        bcss = ClusterEvaluator.between_cluster_sum_of_squares(data, labels, centroids)
        wcss = ClusterEvaluator.within_cluster_sum_of_squares(data, labels, centroids)
        
        return (bcss / (n_clusters - 1)) / (wcss / (n_samples - n_clusters))


class SilhouetteAnalyzer:
    """Class for performing silhouette analysis across multiple k values"""
    
    def __init__(self, algorithm_class: type = None, **algorithm_kwargs):
        """
        Initialize SilhouetteAnalyzer
        
        Args:
            algorithm_class: Clustering algorithm class to use
            **algorithm_kwargs: Additional arguments for the algorithm
        """
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs
        self.results_ = None
    
    def analyze(self, data: np.ndarray, k_range: range = None) -> Dict[str, Any]:
        """
        Perform silhouette analysis across different k values
        
        Args:
            data: Dataset of shape (n_samples, n_features)
            k_range: Range of k values to test (default: 2 to 10)
            
        Returns:
            Dictionary containing analysis results
        """
        if k_range is None:
            k_range = range(2, min(11, len(data)))
        
        k_values = []
        silhouette_scores = []
        wcss_scores = []
        ch_scores = []
        
        for k in k_range:
            if k > len(data):
                continue
                
            # Initialize and fit algorithm
            if self.algorithm_class is None:
                from .algorithms import KMeans
                algorithm = KMeans(n_clusters=k, **self.algorithm_kwargs)
            else:
                algorithm = self.algorithm_class(n_clusters=k, **self.algorithm_kwargs)
            
            algorithm.fit(data)
            
            # Calculate metrics
            silhouette = ClusterEvaluator.silhouette_score(data, algorithm.labels_)
            wcss = ClusterEvaluator.within_cluster_sum_of_squares(
                data, algorithm.labels_, algorithm.centroids_
            )
            ch_score = ClusterEvaluator.calinski_harabasz_score(
                data, algorithm.labels_, algorithm.centroids_
            )
            
            k_values.append(k)
            silhouette_scores.append(silhouette)
            wcss_scores.append(wcss)
            ch_scores.append(ch_score)
        
        self.results_ = {
            'k_values': k_values,
            'silhouette_scores': silhouette_scores,
            'wcss_scores': wcss_scores,
            'calinski_harabasz_scores': ch_scores,
            'best_k_silhouette': k_values[np.argmax(silhouette_scores)] if silhouette_scores else None,
            'best_k_calinski_harabasz': k_values[np.argmax(ch_scores)] if ch_scores else None
        }
        
        return self.results_
    
    def get_optimal_k(self, method: str = 'silhouette') -> Optional[int]:
        """
        Get optimal number of clusters based on specified method
        
        Args:
            method: Method to use ('silhouette', 'calinski_harabasz', 'elbow')
            
        Returns:
            Optimal k value
        """
        if self.results_ is None:
            raise ValueError("Must run analyze() first")
        
        if method == 'silhouette':
            return self.results_['best_k_silhouette']
        elif method == 'calinski_harabasz':
            return self.results_['best_k_calinski_harabasz']
        elif method == 'elbow':
            # Simple elbow method using WCSS
            wcss_scores = self.results_['wcss_scores']
            k_values = self.results_['k_values']
            
            if len(wcss_scores) < 3:
                return k_values[0] if k_values else None
            
            # Calculate differences
            diffs = np.diff(wcss_scores)
            second_diffs = np.diff(diffs)
            
            # Find elbow point (maximum second difference)
            elbow_idx = np.argmax(second_diffs) + 1
            return k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[-1]
        else:
            raise ValueError(f"Unknown method: {method}")
