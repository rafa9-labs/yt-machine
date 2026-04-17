"""
Push the daily video pipeline workflow to n8n.
Run once: python setup_n8n.py
"""
import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

N8N_URL = "http://localhost:5678"
N8N_KEY = os.getenv("N8N_KEY", "")

if not N8N_KEY:
    print("❌ N8N_KEY not found in .env")
    exit(1)

headers = {
    "X-N8N-API-KEY": N8N_KEY,
    "Content-Type": "application/json",
}

# Workflow with correct URLs for Docker→Host communication
workflow = {
    "name": "Geopolitical Sentinel — Daily Video Pipeline",
    "nodes": [
        {
            "parameters": {
                "rule": {
                    "interval": [
                        {
                            "field": "cronExpression",
                            "expression": "0 8 * * *"
                        }
                    ]
                }
            },
            "id": "cron-trigger",
            "name": "Daily 8AM Trigger",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.1,
            "position": [0, 0]
        },
        {
            "parameters": {
                "url": "http://host.docker.internal:8000/generate",
                "method": "POST",
                "options": {
                    "timeout": 600000
                }
            },
            "id": "generate-video",
            "name": "Generate Video",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.1,
            "position": [220, 0],
        },
        {
            "parameters": {
                "url": "http://host.docker.internal:8000/publish",
                "method": "POST",
                "options": {
                    "timeout": 600000
                }
            },
            "id": "publish-video",
            "name": "Publish to Platforms",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.1,
            "position": [440, 0],
        },
        {
            "parameters": {
                "conditions": {
                    "options": {
                        "caseSensitive": True,
                        "leftValue": "",
                        "typeValidation": "strict"
                    },
                    "conditions": [
                        {
                            "id": "success-check",
                            "leftValue": "={{ $json.status }}",
                            "rightValue": "published",
                            "operator": {
                                "type": "string",
                                "operation": "equals"
                            }
                        }
                    ],
                    "combinator": "and"
                }
            },
            "id": "check-success",
            "name": "Check Publish Status",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2,
            "position": [660, 0]
        },
        {
            "parameters": {
                "keepOnlySet": True,
                "values": {
                    "string": [
                        {"name": "status", "value": "=✅ Video published successfully"},
                        {"name": "result", "value": "={{ JSON.stringify($json) }}"}
                    ]
                }
            },
            "id": "notify-success",
            "name": "Log Success",
            "type": "n8n-nodes-base.set",
            "typeVersion": 1,
            "position": [880, -100],
        },
        {
            "parameters": {
                "keepOnlySet": True,
                "values": {
                    "string": [
                        {"name": "status", "value": "=❌ Video publishing FAILED"},
                        {"name": "error", "value": "={{ JSON.stringify($json) }}"}
                    ]
                }
            },
            "id": "notify-failure",
            "name": "Log Failure",
            "type": "n8n-nodes-base.set",
            "typeVersion": 1,
            "position": [880, 100],
        }
    ],
    "connections": {
        "Daily 8AM Trigger": {
            "main": [
                [{"node": "Generate Video", "type": "main", "index": 0}]
            ]
        },
        "Generate Video": {
            "main": [
                [{"node": "Publish to Platforms", "type": "main", "index": 0}]
            ]
        },
        "Publish to Platforms": {
            "main": [
                [{"node": "Check Publish Status", "type": "main", "index": 0}]
            ]
        },
        "Check Publish Status": {
            "main": [
                [{"node": "Log Success", "type": "main", "index": 0}],
                [{"node": "Log Failure", "type": "main", "index": 0}]
            ]
        }
    },
    "settings": {
        "executionOrder": "v1"
    },
}

# Step 1: Create the workflow
print("📤 Creating workflow in n8n...")
resp = requests.post(
    f"{N8N_URL}/api/v1/workflows",
    headers=headers,
    json=workflow,
)

if resp.status_code in (200, 201):
    data = resp.json()
    wf_id = data.get("id")
    print(f"✅ Workflow created! ID: {wf_id}")
    print(f"   Name: {data.get('name')}")
    print(f"   Active: {data.get('active')}")
    print(f"\n📋 Workflow: {N8N_URL}/workflow/{wf_id}")
    print(f"\n⚠️  Workflow is INACTIVE. Open n8n to review and activate it.")
    print(f"   The cron trigger will fire daily at 8:00 AM London time.")
else:
    print(f"❌ Failed to create workflow: {resp.status_code}")
    print(f"   {resp.text[:500]}")
    exit(1)