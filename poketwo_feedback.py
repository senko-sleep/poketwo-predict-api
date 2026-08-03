"""
Poketwo Feedback System
Parses Poketwo catch messages and provides reinforcement learning feedback
to improve prediction accuracy without degrading the model.
"""
import re
import json
import threading
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List

class PoketwoFeedback:
    """Handles Poketwo catch message parsing and feedback processing"""
    
    def __init__(self, feedback_file: str = "poketwo_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
        self.lock = threading.Lock()
        
        # Pattern for Poketwo catch messages
        self.catch_pattern = re.compile(
            r'Congratulations <@(\d+)>! You caught a Level (\d+) ([^<\n]+)(?:<:[^:]+:\d+>)?(?:\s*\((\d+\.?\d*)%\))?',
            re.IGNORECASE
        )
        
        # Shiny chain pattern
        self.shiny_chain_pattern = re.compile(
            r'\+1 Shiny chain! \(\*\*(\d+)\*\*\)',
            re.IGNORECASE
        )
        
    def _load_feedback(self) -> dict:
        """Load existing feedback data"""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "correct_predictions": defaultdict(int),
                "incorrect_predictions": defaultdict(int),
                "confidence_adjustments": defaultdict(float),
                "pattern_frequency": defaultdict(int),
                "total_feedback": 0,
                "last_updated": None
            }
    
    def _save_feedback(self):
        """Save feedback data to file"""
        with self.lock:
            self.feedback_data["last_updated"] = datetime.now().isoformat()
            with open(self.feedback_file, 'w') as f:
                # Convert defaultdicts to regular dicts for JSON serialization
                save_data = {
                    "correct_predictions": dict(self.feedback_data["correct_predictions"]),
                    "incorrect_predictions": dict(self.feedback_data["incorrect_predictions"]),
                    "confidence_adjustments": dict(self.feedback_data["confidence_adjustments"]),
                    "pattern_frequency": dict(self.feedback_data["pattern_frequency"]),
                    "total_feedback": self.feedback_data["total_feedback"],
                    "last_updated": self.feedback_data["last_updated"]
                }
                json.dump(save_data, f, indent=2)
    
    def parse_catch_message(self, message: str) -> Optional[Dict]:
        """Parse Poketwo catch message and extract relevant information"""
        match = self.catch_pattern.search(message)
        if not match:
            return None
        
        user_id, level, pokemon_name, confidence = match.groups()
        
        # Extract shiny chain if present
        shiny_match = self.shiny_chain_pattern.search(message)
        shiny_chain = int(shiny_match.group(1)) if shiny_match else None
        
        # Clean up pokemon name (remove flags and other text)
        pokemon_name = pokemon_name.strip()
        
        # Handle missing confidence
        if confidence:
            confidence = float(confidence)
        else:
            confidence = None
        
        return {
            "user_id": user_id,
            "level": int(level),
            "pokemon_name": pokemon_name,
            "confidence": confidence,
            "shiny_chain": shiny_chain,
            "raw_message": message
        }
    
    def record_prediction(self, predicted_pokemon: str, actual_pokemon: str, 
                         confidence: float, is_correct: bool):
        """Record prediction feedback"""
        with self.lock:
            self.feedback_data["total_feedback"] += 1
            
            if is_correct:
                self.feedback_data["correct_predictions"][predicted_pokemon] += 1
                # Increase confidence for correct predictions
                current_adjustment = self.feedback_data["confidence_adjustments"].get(predicted_pokemon, 0.0)
                self.feedback_data["confidence_adjustments"][predicted_pokemon] = min(0.2, current_adjustment + 0.01)
            else:
                self.feedback_data["incorrect_predictions"][predicted_pokemon] += 1
                # Decrease confidence for incorrect predictions
                current_adjustment = self.feedback_data["confidence_adjustments"].get(predicted_pokemon, 0.0)
                self.feedback_data["confidence_adjustments"][predicted_pokemon] = max(-0.2, current_adjustment - 0.02)
            
            self._save_feedback()
    
    def get_confidence_adjustment(self, pokemon_name: str) -> float:
        """Get confidence adjustment for a specific pokemon based on feedback"""
        with self.lock:
            return self.feedback_data["confidence_adjustments"].get(pokemon_name, 0.0)
    
    def get_pokemon_accuracy(self, pokemon_name: str) -> float:
        """Get accuracy for a specific pokemon"""
        with self.lock:
            correct = self.feedback_data["correct_predictions"].get(pokemon_name, 0)
            incorrect = self.feedback_data["incorrect_predictions"].get(pokemon_name, 0)
            total = correct + incorrect
            if total == 0:
                return 0.0
            return correct / total
    
    def get_feedback_stats(self) -> dict:
        """Get overall feedback statistics"""
        with self.lock:
            return {
                "total_feedback": self.feedback_data["total_feedback"],
                "unique_pokemon": len(self.feedback_data["correct_predictions"]),
                "last_updated": self.feedback_data["last_updated"],
                "top_correct": dict(sorted(self.feedback_data["correct_predictions"].items(), 
                                          key=lambda x: x[1], reverse=True)[:10]),
                "top_incorrect": dict(sorted(self.feedback_data["incorrect_predictions"].items(), 
                                            key=lambda x: x[1], reverse=True)[:10])
            }
    
    def process_catch_message(self, message: str, predicted_pokemon: str, 
                            predicted_confidence: float) -> Dict:
        """Process a Poketwo catch message and provide feedback"""
        parsed = self.parse_catch_message(message)
        if not parsed:
            return {"error": "Could not parse catch message"}
        
        actual_pokemon = parsed["pokemon_name"]
        is_correct = predicted_pokemon.lower() == actual_pokemon.lower()
        
        # Record the feedback
        self.record_prediction(predicted_pokemon, actual_pokemon, 
                             predicted_confidence, is_correct)
        
        return {
            "parsed": parsed,
            "predicted": predicted_pokemon,
            "actual": actual_pokemon,
            "is_correct": is_correct,
            "confidence_adjustment": self.get_confidence_adjustment(predicted_pokemon),
            "accuracy": self.get_pokemon_accuracy(predicted_pokemon)
        }