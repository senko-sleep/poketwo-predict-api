"""
Probabilistic calibration for embedding similarity scores.

Converts raw cosine similarity scores into properly calibrated probabilities
using temperature scaling. This addresses the fundamental issue where similarity
scores (e.g., 0.92) are treated as probabilities but lack statistical meaning.

Mathematical foundation:
- Temperature scaling: P calibrated = sigmoid((sim - μ) / (σ * T))
- Temperature T is learned on validation data to maximize likelihood
- Calibrated scores reflect true prediction confidence
"""

import numpy as np
import json
import logging
import os
from typing import Optional, Dict, Tuple
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# API-specific paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "models", "similarity_calibration.json")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MEAN = 0.0
DEFAULT_STD = 1.0


class SimilarityCalibrator:
    """Calibrates raw similarity scores to probabilities using temperature scaling."""
    
    def __init__(self, calibration_path: str = CALIBRATION_PATH):
        self._calibration_path = calibration_path
        self._temperature = DEFAULT_TEMPERATURE
        self._mean = DEFAULT_MEAN
        self._std = DEFAULT_STD
        self._fitted = False
        
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration parameters from disk if available."""
        if os.path.exists(self._calibration_path):
            try:
                with open(self._calibration_path, "r") as f:
                    data = json.load(f)
                self._temperature = data.get("temperature", DEFAULT_TEMPERATURE)
                self._mean = data.get("mean", DEFAULT_MEAN)
                self._std = data.get("std", DEFAULT_STD)
                self._fitted = True
                logger.info(
                    f"Loaded similarity calibration: T={self._temperature:.3f}, "
                    f"μ={self._mean:.3f}, σ={self._std:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}, using defaults")
    
    def _save_calibration(self):
        """Save calibration parameters to disk."""
        try:
            os.makedirs(os.path.dirname(self._calibration_path), exist_ok=True)
            with open(self._calibration_path, "w") as f:
                json.dump({
                    "temperature": self._temperature,
                    "mean": self._mean,
                    "std": self._std,
                    "fitted": self._fitted
                }, f, indent=2)
            logger.info(f"Saved similarity calibration: T={self._temperature:.3f}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def fit(self, similarities: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Fit temperature scaling parameters using validation data.
        
        Args:
            similarities: Raw similarity scores (N,)
            labels: Binary labels (1 = correct match, 0 = incorrect match)
        
        Returns:
            Dictionary with fitting metrics
        """
        if len(similarities) == 0 or len(labels) == 0:
            logger.warning("Empty calibration data, using defaults")
            return {"status": "skipped", "reason": "empty_data"}
        
        # Normalize similarities for stable optimization
        self._mean = float(np.mean(similarities))
        self._std = float(np.std(similarities))
        if self._std < 1e-6:
            self._std = 1.0
        
        normalized_sims = (similarities - self._mean) / self._std
        
        # Optimize temperature using maximum likelihood
        def negative_log_likelihood(temp):
            temp = max(0.1, min(10.0, temp[0]))  # Constrain temperature
            logits = normalized_sims / temp
            # Clip logits to prevent numerical issues
            logits = np.clip(logits, -10, 10)
            probs = 1 / (1 + np.exp(-logits))
            # Add small epsilon to prevent log(0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return nll
        
        result = minimize(
            negative_log_likelihood,
            x0=[DEFAULT_TEMPERATURE],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)]
        )
        
        if result.success:
            self._temperature = float(result.x[0])
            self._fitted = True
            self._save_calibration()
            
            # Calculate calibration metrics
            calibrated = self.calibrate(similarities)
            metrics = self._calculate_metrics(calibrated, labels)
            
            logger.info(
                f"Fitted similarity calibration: T={self._temperature:.3f}, "
                f"NLL={result.fun:.4f}, metrics={metrics}"
            )
            return {
                "status": "success",
                "temperature": self._temperature,
                "nll": float(result.fun),
                "metrics": metrics
            }
        else:
            logger.warning(f"Calibration optimization failed: {result.message}")
            return {"status": "failed", "reason": result.message}
    
    def _calculate_metrics(self, probs: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate calibration metrics."""
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                acc = np.mean(labels[mask])
                conf = np.mean(probs[mask])
                ece += np.abs(acc - conf) * np.sum(mask) / len(probs)
        
        # Brier score (lower is better)
        brier = np.mean((probs - labels) ** 2)
        
        return {"ece": float(ece), "brier": float(brier)}
    
    def calibrate(self, similarities: np.ndarray) -> np.ndarray:
        """
        Calibrate raw similarity scores to probabilities.
        
        Args:
            similarities: Raw similarity scores (can be scalar or array)
        
        Returns:
            Calibrated probabilities in [0, 1] range
        """
        similarities = np.asarray(similarities)
        
        if not self._fitted:
            # If not fitted, apply sigmoid with default parameters
            # but clip to prevent extreme values
            normalized = (similarities - 0.5) / 0.5  # Assume 0.5 is neutral
        else:
            normalized = (similarities - self._mean) / self._std
        
        # Apply temperature scaling
        scaled = normalized / self._temperature
        
        # Convert to probability via sigmoid
        probs = 1 / (1 + np.exp(-scaled))
        
        # Clip to valid probability range
        probs = np.clip(probs, 0.01, 0.99)
        
        return probs
    
    def get_uncertainty(self, similarities: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty based on calibration curve slope.
        
        Higher uncertainty when similarity is in the "uncertain region" where
        the calibration curve is steep (small changes in similarity cause large
        changes in probability).
        
        Args:
            similarities: Raw similarity scores
        
        Returns:
            Uncertainty scores in [0, 1] range (higher = more uncertain)
        """
        probs = self.calibrate(similarities)
        # Uncertainty is highest at p=0.5 (maximum entropy)
        uncertainty = 4 * probs * (1 - probs)  # Peaks at 0.5 with value 1.0
        return uncertainty


# Global calibrator instance
_calibrator: Optional[SimilarityCalibrator] = None
_calibrator_lock = None

try:
    import threading
    _calibrator_lock = threading.Lock()
except ImportError:
    _calibrator_lock = None


def get_calibrator() -> SimilarityCalibrator:
    """Get the global similarity calibrator instance."""
    global _calibrator
    if _calibrator is None:
        if _calibrator_lock is not None:
            with _calibrator_lock:
                if _calibrator is None:
                    _calibrator = SimilarityCalibrator()
        else:
            _calibrator = SimilarityCalibrator()
    return _calibrator


def calibrate_similarity(similarity: float) -> float:
    """Convenience function to calibrate a single similarity score."""
    calibrator = get_calibrator()
    return float(calibrator.calibrate(similarity))


def get_uncertainty_score(similarity: float) -> float:
    """Convenience function to get uncertainty for a single similarity score."""
    calibrator = get_calibrator()
    return float(calibrator.get_uncertainty(similarity))
