"""
Ollama Model Benchmark — Qwen3 Fast-Slow Architecture Validation
Tests all three model tiers: Worker (4B), Brain (30B MoE), Fallback (Gemma 4).
"""
import requests
import json
import time

BASE_URL = "http://localhost:11434"
TEST_MODELS = [
    {
        "name": "Worker (Qwen3 4B)",
        "tag": "huihui_ai/qwen3-abliterated:4b",
        "timeout": 60,
    },
    {
        "name": "Brain (Qwen3 30B MoE)",
        "tag": "huihui_ai/qwen3-abliterated:30b-a3b",
        "timeout": 120,
    },
    {
        "name": "Fallback (Gemma 4 Heretic)",
        "tag": "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:latest",
        "timeout": 180,
    },
]

JSON_PROMPT = 'Output ONLY valid JSON with keys "topic" and "impact_score" (1-10). Topic: Iran sanctions on oil exports.'
SCRIPT_PROMPT = 'Create a 2-sentence news hook about: "US deploys warships to Strait of Hormuz." Output plain text only.'
THINKING_PROMPT = 'Analyze: Is this headline geopolitically significant? "China and Russia sign new trade deal." Answer in 1 sentence.'


def _generate(model_tag, prompt, timeout_s, max_tokens=200, num_ctx=4096):
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "model": model_tag,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "num_ctx": num_ctx},
            },
            timeout=(10, timeout_s),
            stream=True,
        )
        r.raise_for_status()
        full = ""
        tokens = 0
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full += chunk["response"]
                    tokens += 1
                if chunk.get("done"):
                    break
        elapsed = time.time() - t0
        speed = tokens / elapsed if elapsed > 0 else 0
        return full.strip(), elapsed, speed, tokens
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, 0, 0, str(e)


def _strip_thinking(text):
    import re
    if not text:
        return text
    think_match = re.search(r'<think\b', text)
    if think_match:
        close = re.search(r'</think\s*>', text)
        if close:
            text = text[close.end():]
        else:
            js = re.search(r'[{]', text[think_match.start():])
            if js:
                text = text[think_match.start() + js.start():]
    text = re.sub(r'</?think[^>]*>', '', text)
    text = re.sub(r'<\|?channel\|?>output<\|?channel\|?>', '', text)
    text = re.sub(r'<\|[^>]*\>', '', text)
    return text.strip()


print("=" * 70)
print("OLLAMA MODEL BENCHMARK — Fast-Slow Architecture")
print("=" * 70)

print("\n[1] Connection test...")
try:
    r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
    data = r.json()
    models = [m["name"] for m in data.get("models", [])]
    print(f"    Available models ({len(models)}):")
    for m in models:
        print(f"      - {m}")
except Exception as e:
    print(f"    FAILED: {e}")
    exit(1)

for model_info in TEST_MODELS:
    print(f"\n{'=' * 70}")
    print(f"[TEST] {model_info['name']}")
    print(f"       Tag: {model_info['tag']}")
    print(f"{'=' * 70}")

    # Test 1: JSON output
    print("\n  Test 1: JSON output (news analysis simulation)")
    result = _generate(model_info["tag"], JSON_PROMPT, model_info["timeout"])
    if len(result) == 5:
        print(f"    FAILED after {result[1]:.1f}s: {result[4]}")
        continue
    text, elapsed, speed, tokens = result[:4]
    cleaned = _strip_thinking(text) if text else ""
    try:
        parsed = json.loads(cleaned)
        json_ok = True
        print(f"    JSON valid: topic={parsed.get('topic', 'N/A')}, impact={parsed.get('impact_score', 'N/A')}")
    except Exception:
        json_ok = False
        print(f"    JSON parse FAILED (cleaned: {cleaned[:100]})")
    print(f"    Speed: {speed:.1f} tok/s | Time: {elapsed:.1f}s | Tokens: {tokens}")
    if text and len(text) > len(cleaned) + 10:
        thinking_ratio = (len(text) - len(cleaned)) / len(text) * 100
        print(f"    Thinking tokens stripped: {thinking_ratio:.0f}% of output")

    # Test 2: Script generation
    print("\n  Test 2: Script generation (text output)")
    result = _generate(model_info["tag"], SCRIPT_PROMPT, model_info["timeout"], max_tokens=150)
    if len(result) == 5:
        print(f"    FAILED after {result[1]:.1f}s: {result[4]}")
        continue
    text, elapsed, speed, tokens = result[:4]
    cleaned = _strip_thinking(text) if text else ""
    print(f"    Output: {cleaned[:150]}")
    print(f"    Speed: {speed:.1f} tok/s | Time: {elapsed:.1f}s | Tokens: {tokens}")

    # Test 3: Thinking/reasoning
    print("\n  Test 3: Reasoning (thinking token handling)")
    result = _generate(model_info["tag"], THINKING_PROMPT, model_info["timeout"], max_tokens=300)
    if len(result) == 5:
        print(f"    FAILED after {result[1]:.1f}s: {result[4]}")
        continue
    text, elapsed, speed, tokens = result[:4]
    cleaned = _strip_thinking(text) if text else ""
    has_thinking = text and len(text) > len(cleaned) + 20
    print(f"    Thinking detected: {'Yes' if has_thinking else 'No'}")
    print(f"    Cleaned output: {cleaned[:150]}")
    print(f"    Speed: {speed:.1f} tok/s | Time: {elapsed:.1f}s | Tokens: {tokens}")

print(f"\n{'=' * 70}")
print("BENCHMARK COMPLETE")
print("=" * 70)
print("\nExpected results on RTX 3090:")
print("  Worker (4B):        80-150+ tok/s")
print("  Brain (30B MoE):    40-70+ tok/s")
print("  Fallback (Gemma 4): 10-20 tok/s")
print("=" * 70)
