"""
Test the Poketwo feedback system with example catch messages
"""
import requests
import json

# Example Poketwo catch messages from the user
test_messages = [
    "Congratulations <@1496286737939300494>! You caught a Level 23 Queer Flag Vivillon<:male:1207734081585152101> (45.70%)!\n\n+1 Shiny chain! (**186**)",
    "Congratulations <@702185600018546720>! You caught a Level 20 Wugtrio<:female:1207734084210532483> (39.78%)!",
    "Congratulations <@415841273438666763>! You caught a Level 14 Shaymin<:unknown:1207995667788726272>!"
]

def test_feedback_system():
    """Test the feedback system with example messages"""
    
    # Test the feedback endpoint
    base_url = "http://localhost:8080"
    
    print("Testing Poketwo Feedback System")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Message: {message[:80]}...")
        
        # Simulate a prediction (in real usage, this would come from the prediction API)
        predicted_pokemon = "test_pokemon"  # This would be the actual prediction
        predicted_confidence = 0.8
        
        # Send feedback
        feedback_data = {
            "message": message,
            "predicted_pokemon": predicted_pokemon,
            "predicted_confidence": predicted_confidence
        }
        
        try:
            response = requests.post(f"{base_url}/feedback", json=feedback_data, timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Feedback processed: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.json()}")
        except Exception as e:
            print(f"Request failed: {e}")
    
    # Get feedback statistics
    print("\n" + "=" * 50)
    print("Feedback Statistics:")
    try:
        response = requests.get(f"{base_url}/feedback/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(json.dumps(stats, indent=2))
        else:
            print(f"Error getting stats: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_message_parser():
    """Test the message parser directly"""
    from poketwo_feedback import PoketwoFeedback
    
    print("\n" + "=" * 50)
    print("Testing Message Parser Directly")
    print("=" * 50)
    
    feedback = PoketwoFeedback()
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Message: {message[:80]}...")
        
        parsed = feedback.parse_catch_message(message)
        if parsed:
            print(f"Parsed successfully:")
            print(f"  Pokemon: {parsed['pokemon_name']}")
            print(f"  Level: {parsed['level']}")
            print(f"  Confidence: {parsed['confidence']}%")
            print(f"  Shiny Chain: {parsed['shiny_chain']}")
        else:
            print("Failed to parse message")

if __name__ == "__main__":
    test_message_parser()
    print("\n\nMake sure the API server is running to test the full feedback system:")
    print("python app.py")
    print("\nThen run:")
    print("python test_feedback.py")