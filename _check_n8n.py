"""Quick check: verify n8n workflow status"""
import requests
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("N8N_KEY", "")
if not key:
    print("N8N_KEY not set in .env")
    exit(1)

r = requests.get("http://localhost:5678/api/v1/workflows", headers={"X-N8N-API-KEY": key})
print(f"n8n status: {r.status_code}")

if r.status_code == 200:
    workflows = r.json().get("data", [])
    print(f"Total workflows: {len(workflows)}")
    for w in workflows:
        print(f"  - {w['name']} | active={w['active']} | id={w['id']}")
else:
    print(f"Error: {r.text[:300]}")