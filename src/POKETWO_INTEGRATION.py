"""
Example Discord Bot Integration for Poketwo Feedback System
This shows how to integrate the feedback system with a Discord bot
"""
import requests
import asyncio
import discord
from discord.ext import commands

class PoketwoFeedbackIntegration:
    """Handles integration with Poketwo feedback system"""
    
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.feedback_enabled = True
        
    async def send_feedback(self, message: str, predicted_pokemon: str, 
                          predicted_confidence: float) -> dict:
        """Send feedback to the prediction API"""
        if not self.feedback_enabled:
            return {"status": "disabled"}
        
        feedback_data = {
            "message": message,
            "predicted_pokemon": predicted_pokemon,
            "predicted_confidence": predicted_confidence
        }
        
        try:
            # Run the request in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.api_url}/feedback",
                    json=feedback_data,
                    timeout=5
                )
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def get_feedback_stats(self) -> dict:
        """Get feedback statistics"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/feedback/stats", timeout=5)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}


# Example Discord Bot Implementation
class PoketwoBot(commands.Bot):
    """Example Discord bot with Poketwo feedback integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback = PoketwoFeedbackIntegration()
        self.predictions = {}  # Store predictions by message ID
        
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        
    async def on_message(self, message):
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Check if this is a Poketwo spawn message
        if self._is_poketwo_spawn(message):
            # Get prediction (you would call your prediction API here)
            prediction = await self._get_prediction(message)
            
            # Store the prediction for later feedback
            self.predictions[message.id] = prediction
            
            # Send the prediction to Discord
            await message.channel.send(
                f"Prediction: {prediction['pokemon']} ({prediction['confidence']:.2f}%)"
            )
        
        # Check if this is a Poketwo catch message
        elif self._is_poketwo_catch(message):
            # Look for the corresponding spawn message
            spawn_message_id = self._find_spawn_message(message)
            
            if spawn_message_id and spawn_message_id in self.predictions:
                prediction = self.predictions[spawn_message_id]
                
                # Send feedback
                feedback_result = await self.feedback.send_feedback(
                    message.content,
                    prediction['pokemon'],
                    prediction['confidence']
                )
                
                if feedback_result.get('status') == 'success':
                    feedback = feedback_result['feedback']
                    if feedback['is_correct']:
                        await message.channel.add_reaction('✅')
                    else:
                        await message.channel.add_reaction('❌')
    
    def _is_poketwo_spawn(self, message) -> bool:
        """Check if message is a Poketwo spawn"""
        # Implement your spawn detection logic
        return False
    
    def _is_poketwo_catch(self, message) -> bool:
        """Check if message is a Poketwo catch message"""
        return "Congratulations" in message.content and "caught" in message.content
    
    async def _get_prediction(self, message) -> dict:
        """Get prediction from your prediction API"""
        # Implement your prediction logic
        return {
            "pokemon": "Pikachu",
            "confidence": 0.95
        }
    
    def _find_spawn_message(self, catch_message) -> str:
        """Find the spawn message corresponding to a catch message"""
        # Implement logic to find the spawn message
        return None


# Standalone function for manual feedback submission
async def submit_manual_feedback(api_url: str, message: str, 
                               predicted_pokemon: str, confidence: float):
    """Manually submit feedback (useful for testing)"""
    feedback = PoketwoFeedbackIntegration(api_url)
    result = await feedback.send_feedback(message, predicted_pokemon, confidence)
    print(f"Feedback result: {result}")


# Example usage
if __name__ == "__main__":
    # Test the feedback system manually
    import asyncio
    
    test_message = "Congratulations <@123>! You caught a Level 23 Queer Flag Vivillon (45.70%)!"
    
    asyncio.run(submit_manual_feedback(
        "http://localhost:8080",
        test_message,
        "Queer Flag Vivillon",
        0.89
    ))