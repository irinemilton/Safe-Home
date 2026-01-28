"""
Simple test script to verify Gemini API is working
Run this to test if Gemini can analyze images
"""

from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Key (first 10 chars): {api_key[:10] if api_key else 'None'}...")

if not api_key:
    print("ERROR: No API key found in .env file!")
    exit(1)

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    print("✓ Gemini API configured successfully")
except Exception as e:
    print(f"✗ Failed to configure Gemini: {e}")
    exit(1)

# Test with a simple text prompt first
try:
    print("\n--- Testing Text Generation ---")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say 'Hello, Gemini is working!'")
    print(f"Response: {response.text}")
    print("✓ Text generation works!")
except Exception as e:
    print(f"✗ Text generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test with an image if available
print("\n--- Testing Image Analysis ---")
test_image_path = "static/uploads"  # Check if there are any uploaded images

import glob
images = glob.glob(f"{test_image_path}/*.*")

if images:
    test_img = images[0]
    print(f"Testing with: {test_img}")
    
    try:
        img = Image.open(test_img)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = "Describe this image in one sentence."
        response = model.generate_content([prompt, img])
        
        print(f"Response: {response.text}")
        print("✓ Image analysis works!")
        
    except Exception as e:
        print(f"✗ Image analysis failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No test images found in static/uploads/")

print("\n=== Test Complete ===")
