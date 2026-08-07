"""
Distance-based similarity metrics for Pokemon embedding comparison.

Replaces cosine similarity with Mahalanobis distance and other discriminative
metrics that preserve the covariance structure of the embedding space.

Mathematical foundation:
- Mahalanobis distance: d² = (x-μ)ᵀΣ⁻¹(x-μ) accounts for class spread
- Euclidean distance: d² = ||x-y||² preserves magnitude information
- Gaussian kernel similarity: sim = exp(-d²/2σ²) converts distance to similarity
"""

import numpy as np
import json
import logging
import os
from typing import Optional, Dict, Tuple
from scipy.linalg import pinvh

logger = logging.getLogger(__name__)

# API-specific paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVARIANCE_PATH = os.path.join(PROJECT_ROOT, "models", "class_covariances.json")
DEFAULT_SIGMA = 1.0


class DistanceMetrics:
    """Distance-based similarity metrics with per-class statistics."""
    
    def __init__(self, covariance_path: str = COVARIANCE_PATH):
        self._covariance_path = covariance_path
        self._class_means: Dict[str, np.ndarray] = {}
        self._class_covariances: Dict[str, np.ndarray] = {}
        self._class_counts: Dict[str, int] = {}
        self._sigma = DEFAULT_SIGMA
        self._fitted = False
        
        self._load_statistics()
    
    def _load_statistics(self):
        """Load pre-computed class statistics from disk."""
        if os.path.exists(self._covariance_path):
            try:
                with open(self._covariance_path, "r") as f:
                    data = json.load(f)
                
                for label, stats in data.items():
                    if "mean" in stats and "covariance" in stats:
                        self._class_means[label] = np.array(stats["mean"], dtype=np.float32)
                        self._class_covariances[label] = np.array(stats["covariance"], dtype=np.float32)
                        self._class_counts[label] = stats.get("count", 0)
                
                self._sigma = data.get("sigma", DEFAULT_SIGMA)
                self._fitted = len(self._class_means) > 0
                
                logger.info(
                    f"Loaded distance metrics for {len(self._class_means)} classes, "
                    f"σ={self._sigma:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to load class statistics: {e}")
    
    def _save_statistics(self):
        """Save class statistics to disk."""
        try:
            os.makedirs(os.path.dirname(self._covariance_path), exist_ok=True)
            
            data = {
                "sigma": self._sigma,
                "classes": {}
            }
            
            for label in self._class_means:
                data["classes"][label] = {
                    "mean": self._class_means[label].tolist(),
                    "covariance": self._class_covariances[label].tolist(),
                    "count": self._class_counts[label]
                }
            
            with open(self._covariance_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved distance metrics for {len(self._class_means)} classes")
        except Exception as e:
            logger.error(f"Failed to save class statistics: {e}")
    
    def fit(self, embeddings: np.ndarray, labels: np.ndarray, min_samples: int = 5) -> Dict:
        """
        Compute per-class mean and covariance statistics.
        
        Args:
            embeddings: Embedding vectors (N, D)
            labels: Class labels (N,)
            min_samples: Minimum samples per class to compute statistics
        
        Returns:
            Dictionary with fitting metrics
        """
        if len(embeddings) == 0 or len(labels) == 0:
            return {"status": "skipped", "reason": "empty_data"}
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        self._class_means = {}
        self._class_covariances = {}
        self._class_counts = {}
        
        valid_classes = 0
        
        for label, count in zip(unique_labels, counts):
            if count < min_samples:
                logger.debug(f"Skipping {label}: insufficient samples ({count} < {min_samples})")
                continue
            
            mask = labels == label
            class_embeddings = embeddings[mask]
            
            # Compute mean
            mean = np.mean(class_embeddings, axis=0)
            
            # Compute covariance with regularization
            centered = class_embeddings - mean
            cov = np.cov(centered, rowvar=False)
            
            # Add regularization to prevent singular matrices
            reg = 1e-6 * np.eye(cov.shape[0])
            cov_reg = cov + reg
            
            self._class_means[str(label)] = mean.astype(np.float32)
            self._class_covariances[str(label)] = cov_reg.astype(np.float32)
            self._class_counts[str(label)] = int(count)
            valid_classes += 1
        
        # Learn optimal sigma for Gaussian kernel using median heuristic
        if valid_classes > 1:
            all_means = np.array(list(self._class_means.values()))
            pairwise_dists = []
            for i in range(len(all_means)):
                for j in range(i + 1, len(all_means)):
                    dist = np.linalg.norm(all_means[i] - all_means[j])
                    pairwise_dists.append(dist)
            
            if pairwise_dists:
                self._sigma = float(np.median(pairwise_dists))
                logger.info(f"Learned σ={self._sigma:.3f} from {len(pairwise_dists)} pairwise distances")
        
        self._fitted = valid_classes > 0
        self._save_statistics()
        
        return {
            "status": "success",
            "valid_classes": valid_classes,
            "total_classes": len(unique_labels),
            "sigma": self._sigma
        }
    
    def mahalanobis_distance(
        self, 
        query: np.ndarray, 
        class_label: str,
        covariance: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Mahalanobis distance to a class.
        
        Args:
            query: Query embedding vector (D,)
            class_label: Target class label
            covariance: Optional pre-computed covariance matrix
        
        Returns:
            Mahalanobis distance (higher = more dissimilar)
        """
        if class_label not in self._class_means:
            # Fall back to Euclidean distance if class statistics not available
            return float(np.linalg.norm(query))
        
        mean = self._class_means[class_label]
        cov = covariance or self._class_covariances[class_label]
        
        # Compute centered vector
        diff = query - mean
        
        # Compute Mahalanobis distance: sqrt((x-μ)ᵀΣ⁻¹(x-μ))
        try:
            # Use pseudo-inverse for numerical stability
            inv_cov = pinvh(cov)
            mahal_sq = diff @ inv_cov @ diff
            return float(np.sqrt(max(0, mahal_sq)))
        except Exception as e:
            logger.debug(f"Mahalanobis distance failed for {class_label}: {e}, using Euclidean")
            return float(np.linalg.norm(diff))
    
    def gaussian_kernel_similarity(
        self,
        query: np.ndarray,
        class_label: str,
        sigma: Optional[float] = None
    ) -> float:
        """
        Convert Mahalanobis distance to similarity via Gaussian kernel.
        
        Args:
            query: Query embedding vector (D,)
            class_label: Target class label
            sigma: Bandwidth parameter (uses learned σ if None)
        
        Returns:
            Similarity in [0, 1] range (higher = more similar)
        """
        sigma = sigma or self._sigma
        distance = self.mahalanobis_distance(query, class_label)
        
        # Gaussian kernel: exp(-d²/2σ²)
        similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        return float(similarity)
    
    def euclidean_distance(self, query: np.ndarray, reference: np.ndarray) -> float:
        """
        Compute Euclidean distance (preserves magnitude information).
        
        Args:
            query: Query embedding vector (D,)
            reference: Reference embedding vector (D,)
        
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(query - reference))
    
    def euclidean_similarity(
        self, 
        query: np.ndarray, 
        reference: np.ndarray,
        sigma: Optional[float] = None
    ) -> float:
        """
        Convert Euclidean distance to similarity via Gaussian kernel.
        
        Args:
            query: Query embedding vector (D,)
            reference: Reference embedding vector (D,)
            sigma: Bandwidth parameter
        
        Returns:
            Similarity in [0, 1] range
        """
        sigma = sigma or self._sigma
        distance = self.euclidean_distance(query, reference)
        similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        return float(similarity)
    
    def query_nearest_classes(
        self,
        query: np.ndarray,
        top_k: int = 5,
        metric: str = "mahalanobis"
    ) -> list[Tuple[str, float]]:
        """
        Find nearest classes using distance-based metric.
        
        Args:
            query: Query embedding vector (D,)
            top_k: Number of results to return
            metric: Distance metric ("mahalanobis" or "euclidean")
        
        Returns:
            List of (label, similarity) tuples sorted by similarity descending
        """
        if not self._fitted or len(self._class_means) == 0:
            return []
        
        similarities = []
        
        for label in self._class_means:
            if metric == "mahalanobis":
                sim = self.gaussian_kernel_similarity(query, label)
            else:  # euclidean
                mean = self._class_means[label]
                sim = self.euclidean_similarity(query, mean)
            
            similarities.append((label, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Global distance metrics instance
_distance_metrics: Optional[DistanceMetrics] = None
_distance_metrics_lock = None

try:
    import threading
    _distance_metrics_lock = threading.Lock()
except ImportError:
    _distance_metrics_lock = None


def get_distance_metrics() -> DistanceMetrics:
    """Get the global distance metrics instance."""
    global _distance_metrics
    if _distance_metrics is None:
        if _distance_metrics_lock is not None:
            with _distance_metrics_lock:
                if _distance_metrics is None:
                    _distance_metrics = DistanceMetrics()
        else:
            _distance_metrics = DistanceMetrics()
    return _distance_metrics
