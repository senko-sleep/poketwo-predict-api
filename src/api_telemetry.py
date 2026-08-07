"""
Simple telemetry collection for the Pokemon prediction API.

Tracks prediction decisions and performance metrics for monitoring
the effectiveness of the prediction fix.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

TELEMETRY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "api_telemetry.jsonl")


class APITelemetry:
    """Simple telemetry collector for API predictions."""
    
    def __init__(self, telemetry_path: str = TELEMETRY_PATH):
        self._telemetry_path = telemetry_path
        self._buffer: List[Dict] = []
        self._buffer_size = 100
        self._metrics: Dict = defaultdict(int)
        
    def record_prediction(
        self,
        pokemon: str,
        confidence: float,
        event_override: bool,
        override_decision: str,
        processing_time_ms: float,
        endpoint: str = "unknown"
    ):
        """Record a prediction with its metadata."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pokemon": pokemon,
            "confidence": float(confidence),
            "event_override": event_override,
            "override_decision": override_decision,
            "processing_time_ms": float(processing_time_ms),
            "endpoint": endpoint
        }
        
        self._buffer.append(entry)
        self._metrics["total_predictions"] += 1
        self._metrics[f"override_{override_decision}"] += 1
        self._metrics["event_overrides"] += 1 if event_override else 0
        
        if len(self._buffer) >= self._buffer_size:
            self._flush()
    
    def _flush(self):
        """Flush telemetry buffer to disk."""
        if not self._buffer:
            return
        
        try:
            os.makedirs(os.path.dirname(self._telemetry_path), exist_ok=True)
            
            with open(self._telemetry_path, "a") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry) + "\n")
            
            logger.debug(f"Flushed {len(self._buffer)} telemetry entries")
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush telemetry: {e}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics summary."""
        return dict(self._metrics)
    
    def flush(self):
        """Manually flush any remaining telemetry."""
        self._flush()


# Global telemetry instance
_api_telemetry: Optional[APITelemetry] = None


def get_api_telemetry() -> APITelemetry:
    """Get the global API telemetry instance."""
    global _api_telemetry
    if _api_telemetry is None:
        _api_telemetry = APITelemetry()
    return _api_telemetry
