# Poketwo Reinforcement Learning System

## Overview

The Pokemon Prediction API now includes a reinforcement learning system that learns from Poketwo catch messages to improve prediction accuracy without degrading the base model.

## How It Works

1. **Message Parsing**: The system parses Poketwo catch messages to extract the actual Pokemon that was caught
2. **Feedback Processing**: When you send feedback about predictions, the system compares predicted vs actual results
3. **Confidence Adjustment**: The system adjusts confidence scores for specific Pokemon based on their historical accuracy
4. **Non-Destructive Learning**: The base ONNX model is never modified - only confidence scores are adjusted

## API Endpoints

### `/feedback` (POST)
Submit Poketwo catch message feedback

**Request:**
```json
{
  "message": "Congratulations <@123>! You caught a Level 23 Queer Flag Vivillon (45.70%)!",
  "predicted_pokemon": "Queer Flag Vivillon",
  "predicted_confidence": 0.89
}
```

**Response:**
```json
{
  "status": "success",
  "feedback": {
    "parsed": {
      "pokemon_name": "Queer Flag Vivillon",
      "level": 23,
      "confidence": 45.7
    },
    "predicted": "Queer Flag Vivillon",
    "actual": "Queer Flag Vivillon",
    "is_correct": true,
    "confidence_adjustment": 0.01,
    "accuracy": 1.0
  }
}
```

### `/feedback/stats` (GET)
Get feedback system statistics

**Response:**
```json
{
  "total_feedback": 150,
  "unique_pokemon": 45,
  "last_updated": "2026-08-03T13:45:00",
  "top_correct": {
    "Pikachu": 25,
    "Charizard": 20
  },
  "top_incorrect": {
    "Eevee": 5,
    "Snorlax": 3
  }
}
```

## Integration with Discord Bot

To integrate with your Discord bot:

```python
import requests

def on_poketwo_catch_message(message, predicted_pokemon, confidence):
    """Called when Poketwo sends a catch message"""
    
    feedback_data = {
        "message": message.content,
        "predicted_pokemon": predicted_pokemon,
        "predicted_confidence": confidence
    }
    
    response = requests.post(
        "http://localhost:8080/feedback",
        json=feedback_data,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Feedback processed: {result['feedback']['is_correct']}")

# Example usage in your Discord bot
@bot.event
async def on_message(message):
    if "Congratulations" in message.content and "caught" in message.content:
        # Get your prediction from your prediction API
        prediction = get_prediction(message.attachments[0].url)
        
        # Send feedback
        on_poketwo_catch_message(message, prediction['pokemon'], prediction['confidence'])
```

## Message Format Support

The system supports various Poketwo catch message formats:

- Standard: `Congratulations <@user>! You caught a Level 23 Pokemon (45.70%)!`
- With emoji: `Congratulations <@user>! You caught a Level 20 Pokemon<:male:123> (39.78%)!`
- With shiny chain: `Congratulations <@user>! You caught a Level 14 Pokemon! +1 Shiny chain! (**186**)`
- Without confidence: `Congratulations <@user>! You caught a Level 14 Pokemon<:unknown:123>`

## Learning Behavior

### Correct Predictions
- Increases confidence adjustment by +0.01 (max +0.20)
- Improves future confidence for that Pokemon
- Tracks in correct_predictions counter

### Incorrect Predictions  
- Decreases confidence adjustment by -0.02 (max -0.20)
- Reduces future confidence for that Pokemon
- Tracks in incorrect_predictions counter

### Confidence Adjustment
The adjustment is applied to model predictions:
```python
final_confidence = model_confidence + feedback_adjustment
final_confidence = max(0.0, min(1.0, final_confidence))
```

## Configuration

Enable/disable the feedback system:

```bash
# Enable (default)
FEEDBACK_ENABLED=true python app.py

# Disable
FEEDBACK_ENABLED=false python app.py
```

## Benefits

1. **Adaptive Learning**: System improves over time based on real Poketwo results
2. **Non-Destructive**: Base model remains unchanged, only confidence scores adjust
3. **Poketwo-Specific**: Learns Poketwo's specific patterns and quirks
4. **Instant Feedback**: No retraining required - learning happens in real-time
5. **Accuracy Tracking**: Monitor which Pokemon are predicted correctly/incorrectly

## Monitoring

Check the health endpoint for feedback system status:

```bash
curl http://localhost:8080/health
```

Response includes:
```json
{
  "feedback": {
    "enabled": true,
    "total_feedback": 150
  }
}
```

## Data Storage

Feedback data is stored in `poketwo_feedback.json`:
- `correct_predictions`: Count of correct predictions per Pokemon
- `incorrect_predictions`: Count of incorrect predictions per Pokemon  
- `confidence_adjustments`: Current confidence adjustment per Pokemon
- `total_feedback`: Total number of feedback submissions
- `last_updated`: Timestamp of last feedback

## Important Notes

- The system does NOT modify the ONNX model
- Learning is purely through confidence score adjustments
- Incorrect predictions won't make the model "dumber"
- The system requires Poketwo catch messages to learn
- Best results come from consistent feedback over time