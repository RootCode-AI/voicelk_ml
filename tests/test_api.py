import requests
import json

# Local development server endpoint for IPA generation
url = "http://localhost:5000/api/generate-ipa"

# Test payload containing a complex code-switched (Sinhala-English) sentence with numbers
payload = {
    "text": "පරිගණකයේ මොළය CPU එකයි. RAM එක 1024 MB වේ."
}

# Set appropriate headers for the JSON payload
headers = {
    "Content-Type": "application/json"
}

try:
    print(f"Sending text: {payload['text']}")
    
    # Execute the POST request to the local API
    response = requests.post(url, json=payload, headers=headers)
    
    # Parse and pretty-print the JSON response with proper Unicode decoding
    print("\n--- Response from API ---")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    
except requests.exceptions.RequestException as e:
    # Handle network or connection-related exceptions
    print(f"Error connecting to the API: {e}")