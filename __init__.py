"""
ClusterPy: A comprehensive clustering library
============================================

A Python library for various clustering algorithms including K-Means, K-Means++, 
Bisecting K-Means, and evaluation metrics like Silhouette analysis.

Author: Aniya Bagheri 
Version: 1.0.0
"""

from .algorithms import KMeans, KMeansPlusPlus, BisectingKMeans
from .evaluation import SilhouetteAnalyzer, ClusterEvaluator
from .utils import DataLoader, DataGenerator
from .visualization import ClusterVisualizer

__version__ = "1.0.0"
__author__ = "Aniya bagheri"
__email__ = "aniyabagherii@gmail.com"

__all__ = [
    'KMeans',
    'KMeansPlusPlus', 
    'BisectingKMeans',
    'SilhouetteAnalyzer',
    'ClusterEvaluator',
    'DataLoader',
    'DataGenerator',
    'ClusterVisualizer'
]
