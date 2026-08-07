"""
Supabase client for Pokemon prediction caching and storage.
"""
import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manage Supabase connections and Pokemon prediction caching."""
    
    def __init__(self):
        """Initialize Supabase client with environment variables."""
        self.supabase_url = os.environ.get(
            "SUPABASE_URL", 
            "https://fkjizoetccgfmthotttm.supabase.co"
        )
        self.supabase_key = os.environ.get(
            "SUPABASE_KEY", 
            "sb_publishable_5Wwz7UrMYkPP1RSuP-orZg_zsPjv23w"
        )
        
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Supabase client."""
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None
    
    def _generate_image_hash(self, image_bytes: bytes) -> str:
        """Generate a hash for image bytes to use as cache key."""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def cache_prediction(self, image_bytes: bytes, pokemon_name: str, 
                        confidence: float, prediction_time_ms: float) -> bool:
        """Cache a prediction result in Supabase."""
        if not self.client:
            return False
        
        try:
            image_hash = self._generate_image_hash(image_bytes)
            
            prediction_data = {
                "image_hash": image_hash,
                "pokemon_name": pokemon_name,
                "confidence": confidence,
                "prediction_time_ms": prediction_time_ms
            }
            
            result = self.client.table("pokemon_predictions").insert(prediction_data).execute()
            
            if result.data:
                logger.info(f"Cached prediction for {pokemon_name} with hash {image_hash[:8]}...")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
            # Don't fail completely - table might not exist yet
            return False
    
    def get_cached_prediction(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Retrieve a cached prediction from Supabase."""
        if not self.client:
            return None
        
        try:
            image_hash = self._generate_image_hash(image_bytes)
            
            result = self.client.table("pokemon_predictions").select("*").eq("image_hash", image_hash).execute()
            
            if result.data and len(result.data) > 0:
                cached = result.data[0]
                logger.info(f"Cache hit for {cached['pokemon_name']} with hash {image_hash[:8]}...")
                return {
                    "pokemon": cached["pokemon_name"],
                    "confidence": cached["confidence"],
                    "confidence_raw": cached["confidence"],
                    "cached": True,
                    "prediction_time_ms": cached.get("prediction_time_ms", 0)
                }
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached prediction: {e}")
            # Don't fail completely - table might not exist yet
            return None
    
    def store_prediction_stats(self, stats: Dict[str, Any]) -> bool:
        """Store prediction statistics in Supabase."""
        if not self.client:
            return False
        
        try:
            stats_data = {
                "total_predictions": stats.get("total_predictions", 0),
                "average_prediction_time_ms": stats.get("average_prediction_time_ms", 0),
                "cache_hit_rate": stats.get("cache_hit_rate", 0),
                "created_at": "now()"
            }
            
            result = self.client.table("prediction_stats").insert(stats_data).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.warning(f"Failed to store prediction stats: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Supabase connection is healthy."""
        if not self.client:
            return False
        
        try:
            # Simple health check - try to query a table
            result = self.client.table("pokemon_predictions").select("*").limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {e}")
            # Don't fail completely - table might not exist yet
            return False


# Global instance
supabase_manager = SupabaseManager()