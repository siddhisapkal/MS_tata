# Test Gemini Models
# =================

import google.generativeai as genai

def test_gemini_models():
    """Test available Gemini models"""
    api_key = "AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac"
    genai.configure(api_key=api_key)
    
    print("Testing Gemini API models...")
    
    # List available models
    try:
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"- {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Test different model names
    model_names = [
        'gemini-1.5-flash',
        'gemini-pro',
        'models/gemini-1.5-flash',
        'models/gemini-pro',
        'gemini-1.5-pro',
        'models/gemini-1.5-pro'
    ]
    
    for model_name in model_names:
        try:
            print(f"\nTesting model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello, test message")
            print(f"Success with {model_name}: {response.text[:50]}...")
            break
        except Exception as e:
            print(f"Failed with {model_name}: {e}")

if __name__ == "__main__":
    test_gemini_models()
