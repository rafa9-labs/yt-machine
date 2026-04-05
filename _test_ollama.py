import requests
import json

# Test what endpoints and models are available
base_url = "http://localhost:11434"

print("Testing Ollama connection...")
print("=" * 50)

# Test 1: Check if Ollama is running
try:
    response = requests.get(f"{base_url}/api/version", timeout=5)
    print(f"✓ Ollama version: {response.text}")
except Exception as e:
    print(f"✗ Cannot connect to Ollama: {e}")
    exit(1)

# Test 2: List available models
try:
    response = requests.get(f"{base_url}/api/tags", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Available models:")
        for model in data.get('models', []):
            print(f"  - {model.get('name', 'unknown')}")
    else:
        print(f"✗ /api/tags returned status {response.status_code}")
except Exception as e:
    print(f"✗ /api/tags failed: {e}")

# Test 3: Try the /api/generate endpoint
print(f"\nTesting /api/generate endpoint...")
try:
    payload = {
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": True,
        "options": {"num_predict": 10}
    }
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=10, stream=True)
    response.raise_for_status()
    print("✓ /api/generate works")
    for line in response.iter_lines():
        if line:
            print(f"  Response: {line.decode()[:100]}")
            break
except Exception as e:
    print(f"✗ /api/generate failed: {e}")

# Test 4: Try the /v1/completions endpoint  
print(f"\nTesting /v1/completions endpoint...")
try:
    payload = {
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": True,
        "options": {"max_tokens": 10}
    }
    response = requests.post(f"{base_url}/v1/completions", json=payload, timeout=10, stream=True)
    response.raise_for_status()
    print("✓ /v1/completions works")
    for line in response.iter_lines():
        if line:
            print(f"  Response: {line.decode()[:100]}")
            break
except Exception as e:
    print(f"✗ /v1/completions failed: {e}")

print("\n" + "=" * 50)
print("If all tests fail, check if Ollama is actually running:")
print("  ollama list")
print("  ollama serve")