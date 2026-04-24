import requests
import json
import sys

def pull_model(model_name):
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name}
    
    print(f"Pulling model: {model_name}")
    print("This may take several minutes...")
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=600)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                
                if "total" in data and "completed" in data:
                    total = data["total"]
                    completed = data["completed"]
                    percent = (completed / total * 100) if total > 0 else 0
                    print(f"\r{status}: {percent:.1f}%", end="", flush=True)
                else:
                    print(f"\r{status}", end="", flush=True)
        
        print("\n✓ Model pulled successfully!")
        return True
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Is it running?")
        print("Check if Ollama service is running in Task Manager")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    model = "llama3.2:latest"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    success = pull_model(model)
    sys.exit(0 if success else 1)
