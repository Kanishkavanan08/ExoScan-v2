import os
import requests
import json

def generate_discovery_brief(target_name, probability_score):
    # 1. Fetch the API key from your system environment securely
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set in your terminal environment.")
        print("Run: set GEMINI_API_KEY=your_actual_key_here")
        return
        
    # 2. Determine the classification status
    status = "HIGH-CONFIDENCE PLANET CANDIDATE" if probability_score >= 0.5 else "STELLAR NOISE / FALSE POSITIVE"
    
    # 3. Construct the prompt
    prompt = f"""
    You are ExoScan, an autonomous NASA astrophysics AI. 
    Write a short, professional 'Discovery Brief' for the target star {target_name}.
    
    Here is the data from our 1D Convolutional Neural Network:
    - Target: {target_name}
    - AI Transit Confidence Score: {probability_score * 100:.2f}%
    - Pipeline Assessment: {status}
    
    Keep the brief under 4 sentences. Sound like a professional astrophysics report. 
    Explain what this confidence score means for the target system.
    """
    
    print(f"Sending telemetry to Gemini API for {target_name}...\n")
    
    # 4. Construct the Direct REST API Web Request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    # 5. Send the request and parse the response
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Check for web errors
        
        result = response.json()
        brief = result['candidates'][0]['content']['parts'][0]['text']
        
        print("="*60)
        print(" 🚀 EXOSCAN AUTOMATED DISCOVERY BRIEF")
        print("="*60)
        print(brief.strip())
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Failed to reach Gemini API. Error: {e}")
        if 'response' in locals():
            print(f"API Details: {response.text}")

if __name__ == "__main__":
    # Passing in a mock prediction score (0.94) to test the generation
    generate_discovery_brief("TOI-700", 0.9412)