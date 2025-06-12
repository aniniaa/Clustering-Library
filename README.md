# ClusterPy 🎯

A comprehensive Python library for clustering algorithms with built-in evaluation metrics and visualization tools.

## Features

- **Multiple Clustering Algorithms**:
  - K-Means
  - K-Means++ (improved initialization)
  - Bisecting K-Means (hierarchical approach)

- **Evaluation Metrics**:
  - Silhouette Analysis
  - Within-Cluster Sum of Squares (WCSS)
  - Calinski-Harabasz Index
  - Between-Cluster Sum of Squares (BCSS)

- **Data Utilities**:
  - Multiple file format support (CSV, TSV, custom delimited)
  - Synthetic data generation (blobs, circles, moons, random)
  - Data preprocessing (standardization, normalization)

- **Visualization Tools**:
  - 2D and 3D cluster plots
  - Silhouette analysis plots
  - Elbow method plots
  - Algorithm comparison plots
  - Multi-metric analysis dashboards

## Installation

### From PyPI (when published)
```bash
pip install clusterpy
```

### From Source
```bash
git clone https://github.com/aniniaa/clusterpy.git
cd clusterpy
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/aniniaa/clusterpy.git
cd clusterpy
pip install -e .[dev]
```

## Quick Start

### Basic Clustering

```python
import numpy as np
from clustering_library import KMeans, DataGenerator, ClusterVisualizer

# Generate sample data
data, true_labels = DataGenerator.make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data)

# Get cluster assignments
labels = kmeans.labels_
centroids = kmeans.centroids_

# Visualize results
visualizer = ClusterVisualizer()
visualizer.plot_clusters_2d(data, labels, centroids, title="K-Means Clustering")
```

### Silhouette Analysis

```python
from clustering_library import SilhouetteAnalyzer, KMeansPlusPlus

# Perform silhouette analysis
analyzer = SilhouetteAnalyzer(algorithm_class=KMeansPlusPlus, random_state=42)
results = analyzer.analyze(data, k_range=range(2, 11))

# Get optimal number of clusters
optimal_k = analyzer.get_optimal_k(method='silhouette')
print(f"Optimal number of clusters: {optimal_k}")

# Visualize analysis
visualizer = ClusterVisualizer()
visualizer.plot_multiple_metrics(results, title="Clustering Analysis")
```

### Load Your Own Data

```python
from clustering_library import DataLoader, DataPreprocessor

# Load data from file
data = DataLoader.load_csv('your_data.csv', skip_labels=True, header=True)

# Preprocess data
standardized_data, scaling_params = DataPreprocessor.standardize(data)

# Apply clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(standardized_data)
```

### Compare Multiple Algorithms

```python
from clustering_library import KMeans, KMeansPlusPlus, BisectingKMeans

# Test different algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'K-Means++': KMeansPlusPlus(n_clusters=4, random_state=42),
    'Bisecting K-Means': BisectingKMeans(n_clusters=4, random_state=42)
}

# Compare results
results = {}
for name, algorithm in algorithms.items():
    algorithm.fit(data)
    results[name] = {
        'labels': algorithm.labels_,
        'centroids': algorithm.centroids_
    }

# Visualize comparison
visualizer.plot_cluster_comparison(data, results, title="Algorithm Comparison")
```

## API Reference

### Clustering Algorithms

#### KMeans
```python
KMeans(n_clusters=3, max_iterations=100, random_state=42)
```
- `n_clusters`: Number of clusters to form
- `max_iterations`: Maximum number of iterations
- `random_state`: Random seed for reproducibility

#### KMeansPlusPlus
```python
KMeansPlusPlus(n_clusters=3, max_iterations=100, random_state=42)
```
K-Means with improved initialization using K-Means++ algorithm.

#### BisectingKMeans
```python
BisectingKMeans(n_clusters=3, max_iterations=100, random_state=42)
```
Hierarchical clustering that recursively bisects clusters.

### Evaluation

#### SilhouetteAnalyzer
```python
analyzer = SilhouetteAnalyzer(algorithm_class=KMeans, **algorithm_kwargs)
results = analyzer.analyze(data, k_range=range(2, 11))
optimal_k = analyzer.get_optimal_k(method='silhouette')  # 'silhouette', 'calinski_harabasz', 'elbow'
```

#### ClusterEvaluator
```python
# Individual metrics
silhouette = ClusterEvaluator.silhouette_score(data, labels)
wcss = ClusterEvaluator.within_cluster_sum_of_squares(data, labels, centroids)
bcss = ClusterEvaluator.between_cluster_sum_of_squares(data, labels, centroids)
ch_score = ClusterEvaluator.calinski_harabasz_score(data, labels, centroids)
```

### Data Utilities

#### DataLoader
```python
# Load from various formats
data = DataLoader.load_from_file('data.txt', delimiter=None, skip_labels=True)
data = DataLoader.load_csv('data.csv', skip_labels=True, header=True)
data = DataLoader.load_tsv('data.tsv', skip_labels=True, header=True)
```

#### DataGenerator
```python
# Generate synthetic datasets
data, labels = DataGenerator.make_blobs(n_samples=300, centers=4, random_state=42)
data, labels = DataGenerator.make_circles(n_samples=300, noise=0.1, random_state=42)
data, labels = DataGenerator.make_moons(n_samples=300, noise=0.1, random_state=42)
data = DataGenerator.make_random(n_samples=300, n_features=2, random_state=42)
```

#### DataPreprocessor
```python
# Standardize data (zero mean, unit variance)
standardized_data, params = DataPreprocessor.standardize(data)

# Normalize data to specific range
normalized_data, params = DataPreprocessor.normalize(data, feature_range=(0, 1))

# Apply saved parameters to new data
new_standardized = DataPreprocessor.apply_standardization(new_data, params)
```

### Visualization

#### ClusterVisualizer
```python
visualizer = ClusterVisualizer(figsize=(10, 6))

# Plot clustering results
visualizer.plot_clusters_2d(data, labels, centroids)
visualizer.plot_clusters_3d(data, labels, centroids)

# Plot analysis results
visualizer.plot_silhouette_analysis(k_values, silhouette_scores)
visualizer.plot_elbow_analysis(k_values, wcss_scores)
visualizer.plot_multiple_metrics(analysis_results)

# Compare algorithms
visualizer.plot_cluster_comparison(data, results_dict)
```

## Examples

### Complete Analysis Pipeline

```python
import numpy as np
from clustering_library import *

# 1. Load and preprocess data
data = DataLoader.load_csv('dataset.csv', skip_labels=True)
data, scaling_params = DataPreprocessor.standardize(data)

# 2. Find optimal number of clusters
analyzer = SilhouetteAnalyzer(algorithm_class=KMeansPlusPlus, random_state=42)
results = analyzer.analyze(data, k_range=range(2, 11))
optimal_k = analyzer.get_optimal_k(method='silhouette')

print(f"Optimal k: {optimal_k}")

# 3. Apply best clustering
best_kmeans = KMeansPlusPlus(n_clusters=optimal_k, random_state=42)
best_kmeans.fit(data)

# 4. Evaluate results
silhouette = ClusterEvaluator.silhouette_score(data, best_kmeans.labels_)
print(f"Silhouette score: {silhouette:.3f}")

# 5. Visualize everything
visualizer = ClusterVisualizer()

# Analysis plots
visualizer.plot_multiple_metrics(results, save_path='analysis.png')

# Clustering results (if 2D data)
if data.shape[1] == 2:
    visualizer.plot_clusters_2d(
        data, best_kmeans.labels_, best_kmeans.centroids_,
        title=f"Optimal Clustering (k={optimal_k})",
        save_path='clusters.png'
    )
```

### Working with Different Data Types

```python
# High-dimensional data
data = DataGenerator.make_random(n_samples=1000, n_features=50, random_state=42)

# Apply clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(data)

# Evaluate (works with any dimensionality)
silhouette = ClusterEvaluator.silhouette_score(data, labels)
print(f"Silhouette score for {data.shape[1]}D data: {silhouette:.3f}")

# For visualization, you'd need dimensionality reduction first
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

visualizer = ClusterVisualizer()
visualizer.plot_clusters_2d(data_2d, labels, title="PCA-reduced Clustering")
```

## File Structure

```
clusterpy/
├── clustering_library/
│   ├── __init__.py
│   ├── algorithms.py      # Clustering algorithms
│   ├── evaluation.py      # Evaluation metrics
│   ├── utils.py           # Data utilities
│   └── visualization.py   # Plotting functions
├── examples/
│   ├── basic_usage.py
│   ├── analysis_pipeline.py
│   └── synthetic_data_demo.py
├── tests/
│   ├── test_algorithms.py
│   ├── test_evaluation.py
│   ├── test_utils.py
│   └── test_visualization.py
├── docs/
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/aniniaa/clusterpy.git
cd clusterpy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 clustering_library/
black clustering_library/
mypy clustering_library/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=clustering_library --cov-report=html

# Run specific test file
pytest tests/test_algorithms.py

# Run specific test
pytest tests/test_algorithms.py::test_kmeans_basic
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ClusterPy in your research, please cite:

```bibtex
@software{clusterpy,
  title={ClusterPy: A Comprehensive Clustering Library for Python},
  author={Aniya Bagheri},
  year={2024},
  url={https://github.com/yourusername/clusterpy},
  version={1.0.0}
}
```

## Changelog

### Version 1.0.0
- Initial release
- K-Means, K-Means++, and Bisecting K-Means algorithms
- Comprehensive evaluation metrics
- Data loading and preprocessing utilities
- Visualization tools
- Synthetic data generation

## Roadmap

- [ ] Add DBSCAN algorithm
- [ ] Add Hierarchical clustering (Agglomerative)
- [ ] Add Gaussian Mixture Models
- [ ] Add dimensionality reduction integration
- [ ] Add streaming/online clustering algorithms
- [ ] Add GPU acceleration support
- [ ] Add more distance metrics
- [ ] Add cluster validation indices

