"""
Utility functions for data loading and generation
"""

import numpy as np
import os
from typing import Optional, Tuple, Union
import warnings


class DataLoader:
    """Class for loading datasets from various formats"""
    
    @staticmethod
    def load_from_file(file_path: str, delimiter: str = None, 
                      skip_labels: bool = True, header: bool = False) -> np.ndarray:
        """
        Load data from a file
        
        Args:
            file_path: Path to the dataset file
            delimiter: Delimiter for parsing (auto-detected if None)
            skip_labels: Whether to skip the first column (assumed to be labels)
            header: Whether file has header row
            
        Returns:
            Numpy array of data points
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            data = []
            
            with open(file_path, "r") as f:
                lines = f.readlines()
                
                # Skip header if present
                if header and lines:
                    lines = lines[1:]
                
                for line in lines:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Auto-detect delimiter if not specified
                    if delimiter is None:
                        if '\t' in line:
                            delimiter = '\t'
                        elif ',' in line:
                            delimiter = ','
                        else:
                            delimiter = None  # Use whitespace
                    
                    if delimiter:
                        values = line.split(delimiter)
                    else:
                        values = line.split()
                    
                    if values:
                        try:
                            # Skip first column if it contains labels
                            start_idx = 1 if skip_labels else 0
                            numeric_values = [float(x) for x in values[start_idx:]]
                            if numeric_values:  # Only add if we have numeric values
                                data.append(numeric_values)
                        except ValueError:
                            warnings.warn(f"Could not parse line: {line}")
                            continue
            
            if not data:
                raise ValueError("No valid data found in file")
            
            data = np.array(data)
            
            # Validate data
            if len(data) < 2:
                warnings.warn("Dataset contains less than 2 data points")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    @staticmethod
    def load_csv(file_path: str, skip_labels: bool = True, 
                header: bool = True) -> np.ndarray:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            skip_labels: Whether to skip the first column
            header: Whether CSV has header row
            
        Returns:
            Numpy array of data points
        """
        return DataLoader.load_from_file(file_path, delimiter=',', 
                                       skip_labels=skip_labels, header=header)
    
    @staticmethod
    def load_tsv(file_path: str, skip_labels: bool = True,
                header: bool = True) -> np.ndarray:
        """
        Load data from TSV file
        
        Args:
            file_path: Path to TSV file
            skip_labels: Whether to skip the first column
            header: Whether TSV has header row
            
        Returns:
            Numpy array of data points
        """
        return DataLoader.load_from_file(file_path, delimiter='\t',
                                       skip_labels=skip_labels, header=header)


class DataGenerator:
    """Class for generating synthetic datasets"""
    
    @staticmethod
    def make_blobs(n_samples: int = 100, n_features: int = 2, centers: int = 3,
                  cluster_std: float = 1.0, center_box: Tuple[float, float] = (-10, 10),
                  random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate isotropic Gaussian blobs for clustering
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            centers: Number of centers (clusters)
            cluster_std: Standard deviation of clusters
            center_box: Bounding box for cluster centers
            random_state: Random seed
            
        Returns:
            Tuple of (data, true_labels)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate cluster centers
        cluster_centers = np.random.uniform(
            center_box[0], center_box[1], (centers, n_features)
        )
        
        # Generate samples
        samples_per_center = n_samples // centers
        remainder = n_samples % centers
        
        data = []
        labels = []
        
        for i in range(centers):
            n_center_samples = samples_per_center + (1 if i < remainder else 0)
            
            # Generate samples around this center
            center_data = np.random.multivariate_normal(
                cluster_centers[i], 
                np.eye(n_features) * cluster_std**2,
                n_center_samples
            )
            
            data.append(center_data)
            labels.extend([i] * n_center_samples)
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # Shuffle data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return data, labels
    
    @staticmethod
    def make_circles(n_samples: int = 100, noise: float = 0.1, factor: float = 0.8,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D concentric circles dataset
        
        Args:
            n_samples: Number of samples
            noise: Standard deviation of Gaussian noise
            factor: Scale factor between inner and outer circle
            random_state: Random seed
            
        Returns:
            Tuple of (data, true_labels)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Outer circle
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)
        
        # Inner circle
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        inner_circ_x = np.cos(linspace_in) * factor
        inner_circ_y = np.sin(linspace_in) * factor
        
        # Combine circles
        data = np.vstack([
            np.column_stack([outer_circ_x, outer_circ_y]),
            np.column_stack([inner_circ_x, inner_circ_y])
        ])
        
        # Add noise
        if noise > 0:
            data += np.random.normal(0, noise, data.shape)
        
        # Create labels
        labels = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
        
        # Shuffle data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return data, labels.astype(int)
    
    @staticmethod
    def make_moons(n_samples: int = 100, noise: float = 0.1,
                  random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D moon-shaped dataset
        
        Args:
            n_samples: Number of samples
            noise: Standard deviation of Gaussian noise
            random_state: Random seed
            
        Returns:
            Tuple of (data, true_labels)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Outer moon
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        
        # Inner moon
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
        
        # Combine moons
        data = np.vstack([
            np.column_stack([outer_circ_x, outer_circ_y]),
            np.column_stack([inner_circ_x, inner_circ_y])
        ])
        
        # Add noise
        if noise > 0:
            data += np.random.normal(0, noise, data.shape)
        
        # Create labels
        labels = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
        
        # Shuffle data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return data, labels.astype(int)
    
    @staticmethod
    def make_random(n_samples: int = 100, n_features: int = 2, 
                   random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random data with standard normal distribution
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            random_state: Random seed
            
        Returns:
            Random data array
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        return np.random.normal(0, 1, size=(n_samples, n_features))
    
    @staticmethod
    def generate_like_dataset(reference_data: np.ndarray, 
                            random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic data with same dimensions as reference dataset
        
        Args:
            reference_data: Reference dataset to match dimensions
            random_state: Random seed
            
        Returns:
            Synthetic data with same shape as reference
        """
        n_samples, n_features = reference_data.shape
        return DataGenerator.make_random(n_samples, n_features, random_state)


class DataPreprocessor:
    """Class for preprocessing data"""
    
    @staticmethod
    def standardize(data: np.ndarray, copy: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Standardize data to have zero mean and unit variance
        
        Args:
            data: Input data
            copy: Whether to copy data or modify in-place
            
        Returns:
            Tuple of (standardized_data, scaling_params)
        """
        if copy:
            data = data.copy()
        
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1
        
        standardized_data = (data - means) / stds
        
        scaling_params = {'means': means, 'stds': stds}
        
        return standardized_data, scaling_params
    
    @staticmethod
    def normalize(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1),
                 copy: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Normalize data to specified range
        
        Args:
            data: Input data
            feature_range: Target range for normalization
            copy: Whether to copy data or modify in-place
            
        Returns:
            Tuple of (normalized_data, scaling_params)
        """
        if copy:
            data = data.copy()
        
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        
        # Scale to [0, 1] first
        scaled_data = (data - min_vals) / ranges
        
        # Scale to target range
        target_min, target_max = feature_range
        normalized_data = scaled_data * (target_max - target_min) + target_min
        
        scaling_params = {
            'min_vals': min_vals,
            'max_vals': max_vals,
            'feature_range': feature_range
        }
        
        return normalized_data, scaling_params
    
    @staticmethod
    def apply_standardization(data: np.ndarray, scaling_params: dict) -> np.ndarray:
        """
        Apply previously computed standardization parameters
        
        Args:
            data: Data to standardize
            scaling_params: Parameters from standardize()
            
        Returns:
            Standardized data
        """
        means = scaling_params['means']
        stds = scaling_params['stds']
        return (data - means) / stds
    
    @staticmethod
    def apply_normalization(data: np.ndarray, scaling_params: dict) -> np.ndarray:
        """
        Apply previously computed normalization parameters
        
        Args:
            data: Data to normalize
            scaling_params: Parameters from normalize()
            
        Returns:
            Normalized data
        """
        min_vals = scaling_params['min_vals']
        max_vals = scaling_params['max_vals']
        feature_range = scaling_params['feature_range']
        
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        
        scaled_data = (data - min_vals) / ranges
        target_min, target_max = feature_range
        return scaled_data * (target_max - target_min) + target_min