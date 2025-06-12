"""
Clustering algorithms module
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms"""
    
    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize clustering algorithm
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'ClusteringAlgorithm':
        """Fit the clustering algorithm to data"""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for data"""
        pass
    
    def fit_predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fit the algorithm and return cluster assignments"""
        return self.fit(data, **kwargs).predict(data)
    
    @staticmethod
    def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two points
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))


class KMeans(ClusteringAlgorithm):
    """K-Means clustering algorithm"""
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, 
                 random_state: Optional[int] = 42):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids_ = None
        self.labels_ = None
        self.n_iterations_ = 0
    
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Randomly initialize centroids"""
        n_samples, _ = data.shape
        k = min(self.n_clusters, n_samples)
        indices = random.sample(range(n_samples), k)
        return data[indices].copy()
    
    def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each data point to nearest centroid"""
        n_points = len(data)
        assignments = np.zeros(n_points, dtype=int)
        
        for i in range(n_points):
            min_dist = float("inf")
            min_cluster = 0
            
            for j in range(len(centroids)):
                dist = self.compute_distance(data[i], centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = j
            
            assignments[i] = min_cluster
        
        return assignments
    
    def _update_centroids(self, data: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Update centroids based on current assignments"""
        n_features = data.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        counts = np.zeros(self.n_clusters)
        
        # Sum up all points in each cluster
        for i in range(len(data)):
            cluster_id = assignments[i]
            centroids[cluster_id] += data[i]
            counts[cluster_id] += 1
        
        # Compute average for each cluster
        for i in range(self.n_clusters):
            if counts[i] > 0:
                centroids[i] /= counts[i]
            else:
                # If cluster is empty, pick a random point
                centroids[i] = data[random.randint(0, len(data) - 1)]
        
        return centroids
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        """
        Fit K-Means to data
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        if len(data) < self.n_clusters:
            raise ValueError(f"Number of clusters ({self.n_clusters}) cannot be "
                           f"greater than number of samples ({len(data)})")
        
        # Initialize centroids
        centroids = self._initialize_centroids(data)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            assignments = self._assign_clusters(data, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(data, assignments)
            
            # Check for convergence
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                break
            
            centroids = new_centroids
            self.n_iterations_ = iteration + 1
        
        self.centroids_ = centroids
        self.labels_ = assignments
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for data
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return self._assign_clusters(data, self.centroids_)


class KMeansPlusPlus(KMeans):
    """K-Means++ clustering algorithm with improved initialization"""
    
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids using K-Means++ algorithm"""
        n_samples, n_features = data.shape
        k = min(self.n_clusters, n_samples)
        
        centroids = []
        
        # Choose first centroid randomly
        first_idx = random.randint(0, n_samples - 1)
        centroids.append(data[first_idx].copy())
        
        # Choose remaining centroids
        for _ in range(1, k):
            # Calculate squared distances to nearest centroid
            distances = np.array([
                min([self.compute_distance(point, centroid) ** 2 
                     for centroid in centroids])
                for point in data
            ])
            
            # Normalize to create probability distribution
            probabilities = distances / np.sum(distances)
            
            # Choose next centroid based on probability distribution
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(data[next_idx].copy())
        
        return np.array(centroids)


class BisectingKMeans(ClusteringAlgorithm):
    """Bisecting K-Means clustering algorithm"""
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100,
                 random_state: Optional[int] = 42):
        """
        Initialize Bisecting K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            max_iterations: Maximum iterations for each K-Means run
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids_ = None
        self.labels_ = None
        self.hierarchy_ = None
    
    def _compute_sum_square(self, data: np.ndarray, centroid: np.ndarray) -> float:
        """Compute sum of squared distances within a cluster"""
        return sum(self.compute_distance(point, centroid) ** 2 for point in data)
    
    def _kmeans_step(self, data: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Perform K-Means with k clusters on data"""
        kmeans = KMeans(n_clusters=k, max_iterations=self.max_iterations, 
                       random_state=self.random_state)
        kmeans.fit(data)
        return kmeans.labels_, kmeans.centroids_
    
    def fit(self, data: np.ndarray) -> 'BisectingKMeans':
        """
        Fit Bisecting K-Means to data
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        if len(data) < self.n_clusters:
            raise ValueError(f"Number of clusters ({self.n_clusters}) cannot be "
                           f"greater than number of samples ({len(data)})")
        
        # Start with one cluster containing all points
        current_clusters = [{
            "points": data, 
            "centroid": np.mean(data, axis=0)
        }]
        
        hierarchy = []
        
        # Initial clustering (1 cluster)
        hierarchy.append({
            "assignments": np.zeros(len(data), dtype=int),
            "centroids": np.array([np.mean(data, axis=0)])
        })
        
        # Bisect clusters until we reach n_clusters
        for _ in range(1, self.n_clusters):
            # Find cluster with largest sum of squared distances
            max_sum_squared = -1
            cluster_to_bisect = -1
            
            for i, cluster in enumerate(current_clusters):
                sum_squared = self._compute_sum_square(cluster["points"], cluster["centroid"])
                if sum_squared > max_sum_squared:
                    max_sum_squared = sum_squared
                    cluster_to_bisect = i
            
            if cluster_to_bisect == -1 or len(current_clusters[cluster_to_bisect]["points"]) <= 1:
                break
            
            # Bisect the selected cluster
            cluster = current_clusters[cluster_to_bisect]
            bisect_assignments, bisect_centroids = self._kmeans_step(cluster["points"], 2)
            
            # Create two new clusters
            cluster1_points = cluster["points"][bisect_assignments == 0]
            cluster2_points = cluster["points"][bisect_assignments == 1]
            
            if len(cluster1_points) == 0 or len(cluster2_points) == 0:
                continue
            
            # Replace selected cluster with two new clusters
            current_clusters.pop(cluster_to_bisect)
            current_clusters.append({
                "points": cluster1_points,
                "centroid": np.mean(cluster1_points, axis=0)
            })
            current_clusters.append({
                "points": cluster2_points,
                "centroid": np.mean(cluster2_points, axis=0)
            })
            
            # Update assignments and centroids
            assignments = np.zeros(len(data), dtype=int)
            centroids = np.zeros((len(current_clusters), data.shape[1]))
            
            point_index = 0
            for cluster_id, cluster in enumerate(current_clusters):
                for _ in range(len(cluster["points"])):
                    assignments[point_index] = cluster_id
                    point_index += 1
                centroids[cluster_id] = cluster["centroid"]
            
            hierarchy.append({
                "assignments": assignments.copy(),
                "centroids": centroids.copy()
            })
        
        # Set final results
        final_clustering = hierarchy[-1]
        self.labels_ = final_clustering["assignments"]
        self.centroids_ = final_clustering["centroids"]
        self.hierarchy_ = hierarchy
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for data
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet")
        
        n_points = len(data)
        assignments = np.zeros(n_points, dtype=int)
        
        for i in range(n_points):
            min_dist = float("inf")
            min_cluster = 0
            
            for j in range(len(self.centroids_)):
                dist = self.compute_distance(data[i], self.centroids_[j])
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = j
            
            assignments[i] = min_cluster
        
        return assignments