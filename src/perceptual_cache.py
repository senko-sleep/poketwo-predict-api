"""
Perceptual Caching Layer for Pokemon Prediction
Provides sprite-invariant caching for repeated Pokemon spawns using
foreground segmentation, canonicalization, and perceptual hashing.
"""
import os
import sys
import json
import pickle
import hashlib
import threading
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import imagehash

# Try to import imagehash, provide fallback if not available
try:
    from imagehash import phash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("Warning: imagehash not available, using fallback hashing")


class PerceptualCache:
    """
    Perceptual caching layer for Pokemon sprite prediction.
    Invariant to position, scale, and flip while sensitive to Pokemon identity.
    """
    
    def __init__(self, 
                 cache_file: str = "perceptual_cache.pkl",
                 canonical_size: int = 64,
                 hamming_threshold: int = 5,
                 max_cache_size: int = 10000):
        """
        Initialize perceptual cache.
        
        Args:
            cache_file: Path to cache file for persistence
            canonical_size: Size to resize sprites to (default 64x64)
            hamming_threshold: Maximum hamming distance for cache match
            max_cache_size: Maximum number of entries in cache
        """
        self.cache_file = cache_file
        self.canonical_size = canonical_size
        self.hamming_threshold = hamming_threshold
        self.max_cache_size = max_cache_size
        
        # Cache structure: {hash: {'pokemon': str, 'confidence': float, 'timestamp': str}}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.cache_hits = data.get('cache_hits', 0)
                    self.cache_misses = data.get('cache_misses', 0)
                    self.total_requests = data.get('total_requests', 0)
                print(f"Loaded perceptual cache: {len(self.cache)} entries")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with self.lock:
                data = {
                    'cache': self.cache,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'total_requests': self.total_requests,
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _extract_sprite_bounding_box(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract Pokemon sprite bounding box from image using foreground segmentation.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of (x1, y1, x2, y2) bounding box or None if extraction fails
        """
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply adaptive thresholding to separate foreground from background
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (assuming Pokemon is the main object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding)
            y2 = min(image.height, y + h + padding)
            
            # Ensure minimum size
            if x2 - x1 < 10 or y2 - y1 < 10:
                return None
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            print(f"Failed to extract bounding box: {e}")
            return None
    
    def _canonicalize_sprite(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Extract and canonicalize sprite to fixed size.
        
        Args:
            image: Original image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Canonicalized sprite image
        """
        try:
            # Crop to bounding box
            sprite = image.crop(bbox)
            
            # Convert to RGBA to handle transparency
            if sprite.mode != 'RGBA':
                sprite = sprite.convert('RGBA')
            
            # Resize to canonical size
            sprite = sprite.resize((self.canonical_size, self.canonical_size), Image.LANCZOS)
            
            return sprite
            
        except Exception as e:
            print(f"Failed to canonicalize sprite: {e}")
            # Fallback: resize entire image
            return image.resize((self.canonical_size, self.canonical_size), Image.LANCZOS)
    
    def _compute_perceptual_hash(self, image: Image.Image) -> str:
        """
        Compute perceptual hash of image.
        
        Args:
            image: PIL Image to hash
            
        Returns:
            Hash string
        """
        try:
            if IMAGEHASH_AVAILABLE:
                # Use imagehash library for better perceptual hashing
                hash_obj = phash(image)
                return str(hash_obj)
            else:
                # Fallback: simple hash
                # Convert to grayscale and downsample
                small = image.resize((8, 8), Image.LANCZOS).convert('L')
                pixels = list(small.getdata())
                avg = sum(pixels) / len(pixels)
                bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
                return bits
        except Exception as e:
            print(f"Failed to compute hash: {e}")
            # Final fallback: MD5 of image data
            return hashlib.md5(image.tobytes()).hexdigest()
    
    def _find_cache_match(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        Find cache entry within hamming distance threshold.
        
        Args:
            hash_value: Hash to search for
            
        Returns:
            Cache entry if match found, None otherwise
        """
        if not IMAGEHASH_AVAILABLE:
            # For fallback hash, use exact match
            return self.cache.get(hash_value)
        
        # For perceptual hash, use hamming distance
        for cached_hash, entry in self.cache.items():
            try:
                # Compute hamming distance
                hash1 = imagehash.hex_to_hash(hash_value)
                hash2 = imagehash.hex_to_hash(cached_hash)
                distance = hash1 - hash2
                
                if distance <= self.hamming_threshold:
                    return entry
            except:
                # Fallback to exact match if hash comparison fails
                if hash_value == cached_hash:
                    return entry
        
        return None
    
    def predict(self, 
               image: Image.Image, 
               onnx_predict_func: callable,
               force_model: bool = False) -> Dict[str, Any]:
        """
        Predict Pokemon with perceptual caching.
        
        Args:
            image: PIL Image to predict
            onnx_predict_func: Function to call for ONNX prediction fallback
            force_model: Force model inference even if cache hit
            
        Returns:
            Prediction dict with 'pokemon', 'confidence', 'cache_hit' keys
        """
        self.total_requests += 1
        
        # Step 1: Extract bounding box
        bbox = self._extract_sprite_bounding_box(image)
        if bbox is None:
            # Fallback: use entire image
            bbox = (0, 0, image.width, image.height)
        
        # Step 2: Canonicalize sprite
        canonical_sprite = self._canonicalize_sprite(image, bbox)
        
        # Step 3: Compute perceptual hashes
        normal_hash = self._compute_perceptual_hash(canonical_sprite)
        flipped_sprite = ImageOps.mirror(canonical_sprite)
        flipped_hash = self._compute_perceptual_hash(flipped_sprite)
        
        # Step 4: Check cache
        if not force_model:
            for hash_value in [normal_hash, flipped_hash]:
                cache_entry = self._find_cache_match(hash_value)
                if cache_entry:
                    self.cache_hits += 1
                    cache_entry['cache_hit'] = True
                    cache_entry['hash_used'] = hash_value
                    return cache_entry
        
        # Step 5: Cache miss - run ONNX model
        self.cache_misses += 1
        
        # Call the ONNX prediction function
        prediction = onnx_predict_func(image)
        
        # Step 6: Store in cache
        cache_entry = {
            'pokemon': prediction.get('pokemon', 'unknown'),
            'confidence': prediction.get('confidence_raw', 0.0),
            'cache_hit': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store under both hashes for flip invariance
        with self.lock:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[normal_hash] = cache_entry.copy()
            self.cache[flipped_hash] = cache_entry.copy()
        
        # Periodically save cache
        if self.total_requests % 10 == 0:
            self._save_cache()
        
        return cache_entry
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0
            return {
                'cache_entries': len(self.cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'total_requests': self.total_requests,
                'hit_rate': hit_rate,
                'canonical_size': self.canonical_size,
                'hamming_threshold': self.hamming_threshold
            }
    
    def clear_cache(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_requests = 0
            self._save_cache()
    
    def save_cache(self):
        """Manually save cache to file."""
        self._save_cache()