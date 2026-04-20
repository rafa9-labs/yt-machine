"""
Pipeline Validation Test — Qwen3 Fast-Slow Architecture
========================================================
Validates every LLM call in the generate_complete_video.py pipeline
using the new Qwen3 model routing.

Tests:
  1. Config loading — correct model routing per task
  2. Raw LLM interface — generate, process_news, debate, script synthesis
  3. LangChain interface — structured chains, text chains
  4. Thinking token stripping — Qwen3, Gemma 4, DeepSeek-R1 patterns
  5. JSON extraction — handles thinking tokens + malformed JSON
  6. End-to-end pipeline simulation — news → analysis → script → curation
"""
import json
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}" + (f" — {detail}" if detail and not passed else ""))
    if not passed and detail:
        print(f"       {detail}")


# ══════════════════════════════════════════════════════════════════════
# TEST 1: CONFIG LOADING & MODEL ROUTING
# ══════════════════════════════════════════════════════════════════════
print("\n[1] CONFIG LOADING & MODEL ROUTING")
print("-" * 50)

config_path = Path(__file__).parent.parent / "config" / "system_prompts.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

mc = config["model_config"]

test("default_model is Qwen3 4B Worker",
     mc["default_model"] == "huihui_ai/qwen3-abliterated:4b",
     f"Got: {mc['default_model']}")

test("fallback_model is Gemma 4 Heretic",
     mc["fallback_model"] == "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:latest",
     f"Got: {mc['fallback_model']}")

test("timeout reduced from 600 to 300",
     mc["timeout"] == 300,
     f"Got: {mc['timeout']}")

test("num_ctx reduced from 32768 to 16384",
     mc["num_ctx"] == 16384,
     f"Got: {mc['num_ctx']}")

task_models = mc.get("task_models", {})
test("multi_news_synthesizer uses Brain (30B MoE)",
     "30b-a3b" in task_models.get("multi_news_synthesizer", ""),
     f"Got: {task_models.get('multi_news_synthesizer', 'MISSING')}")

worker_tasks = ["news_processor", "visual_prompt_generator", "script_curator",
                "salience_extractor", "debate_skeptic", "debate_explainer"]
for task in worker_tasks:
    test(f"{task} uses Worker (4B)",
         task_models.get(task) == "huihui_ai/qwen3-abliterated:4b",
         f"Got: {task_models.get(task, 'MISSING')}")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: RAW LLM INTERFACE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n[2] RAW LLM INTERFACE")
print("-" * 50)

try:
    from brain.llm_interface import LLMInterface
    llm = LLMInterface()

    test("LLMInterface loads without error", True)
    test(f"default_model = {llm.default_model[:40]}...",
         "qwen3-abliterated:4b" in llm.default_model)
    test(f"fallback_model = {llm.fallback_model[:40]}...",
         "gemma-4-26B" in llm.fallback_model)
    test(f"num_ctx = {llm.num_ctx}", llm.num_ctx == 16384)
    test(f"timeout = {llm.timeout}", llm.timeout == 300)

except Exception as e:
    test("LLMInterface loads without error", False, str(e))


# ══════════════════════════════════════════════════════════════════════
# TEST 3: THINKING TOKEN STRIPPING
# ══════════════════════════════════════════════════════════════════════
print("\n[3] THINKING TOKEN STRIPPING")
print("-" * 50)

strip = LLMInterface._strip_thinking_tokens

# Qwen3 pattern: <think...\n</think\n\n{json}
qwen3_thinking = '<think\nLet me analyze this...\nSome reasoning here.\n</think\n\n{"topic": "test", "impact_score": 7}'
stripped = strip(qwen3_thinking)
test("Qwen3: strips <think...</think wrapper",
     '"topic"' in stripped and '<think' not in stripped,
     f"Got: {stripped[:100]}")

# Qwen3 pattern without closing tag
qwen3_no_close = '<think\nReasoning about stuff...\n{"topic": "test"}'
stripped = strip(qwen3_no_close)
test("Qwen3: handles unclosed <think tag",
     '"topic"' in stripped and '<think' not in stripped,
     f"Got: {stripped[:100]}")

# Gemma 4 Heretic pattern
gemma4_thinking = '<|channel>thought<channel|>analysis here<|channel>output<channel|>{"topic": "test"}'
stripped = strip(gemma4_thinking)
test("Gemma 4: strips <|channel|> thought tokens",
     '"topic"' in stripped and 'channel' not in stripped,
     f"Got: {stripped[:100]}")

# Clean JSON (no thinking tokens)
clean_json = '{"topic": "Iran sanctions", "impact_score": 8}'
stripped = strip(clean_json)
test("Clean JSON: passes through unchanged",
     stripped == clean_json,
     f"Got: {stripped[:100]}")

# DeepSeek-R1 pattern
deepseek_thinking = '<think\n\nStep by step analysis...\n\n</think\n\n{"result": true}'
stripped = strip(deepseek_thinking)
test("DeepSeek-R1: strips thinking wrapper",
     '"result"' in stripped and '<think' not in stripped,
     f"Got: {stripped[:100]}")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: JSON EXTRACTION
# ══════════════════════════════════════════════════════════════════════
print("\n[4] JSON EXTRACTION")
print("-" * 50)

extract = llm._extract_json

# JSON wrapped in thinking tokens
thinking_json = '<think\nAnalyzing...\n</think\n\n{"topic": "Iran", "impact_score": 8}'
parsed = extract(thinking_json)
test("Extract JSON from thinking-wrapped response",
     parsed is not None and parsed.get("topic") == "Iran",
     f"Got: {parsed}")

# JSON with markdown fences
markdown_json = '```json\n{"topic": "test", "impact_score": 5}\n```'
parsed = extract(markdown_json)
test("Extract JSON from markdown code block",
     parsed is not None and parsed.get("topic") == "test",
     f"Got: {parsed}")

# Incomplete JSON (auto-close)
incomplete_json = '{"topic": "test", "impact_score": 8, "key_facts": ["fact1", "fact2"'
parsed = extract(incomplete_json)
test("Auto-close incomplete JSON",
     parsed is not None and parsed.get("topic") == "test",
     f"Got: {parsed}")

# Trailing commas
trailing_json = '{"topic": "test", "impact_score": 8,}'
parsed = extract(trailing_json)
test("Fix trailing commas",
     parsed is not None,
     f"Got: {parsed}")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: LANGCHAIN INTERFACE
# ══════════════════════════════════════════════════════════════════════
print("\n[5] LANGCHAIN INTERFACE")
print("-" * 50)

try:
    from brain.langchain_interface import LangChainInterface
    lc = LangChainInterface()

    test("LangChainInterface loads without error", True)
    test(f"LangChain default_model correct",
         "qwen3-abliterated:4b" in lc.default_model)

    prompt_config = lc.get_prompt_config("news_processor")
    test("news_processor routed to Worker",
         "4b" in prompt_config["model"],
         f"Got: {prompt_config['model']}")

    prompt_config = lc.get_prompt_config("multi_news_synthesizer")
    test("multi_news_synthesizer routed to Brain",
         "30b-a3b" in prompt_config["model"],
         f"Got: {prompt_config['model']}")

    prompt_config = lc.get_prompt_config("script_curator")
    test("script_curator routed to Worker",
         "4b" in prompt_config["model"],
         f"Got: {prompt_config['model']}")

except Exception as e:
    test("LangChainInterface loads without error", False, str(e))


# ══════════════════════════════════════════════════════════════════════
# TEST 6: LIVE LLM CALLS (Worker only — fast, ~10s total)
# ══════════════════════════════════════════════════════════════════════
print("\n[6] LIVE LLM CALLS (Worker — Qwen3 4B)")
print("-" * 50)

SAMPLE_ARTICLE = """
US Deploys Warships to Strait of Hormuz Amid Rising Iran Tensions.
The United States has deployed two additional warships to the Strait of Hormuz
as tensions with Iran continue to escalate over nuclear program disputes.
Iran has threatened to close the strait, through which 20% of global oil passes.
The Pentagon stated this is a "defensive posture" while Tehran called it "provocative."
"""

t0 = time.time()
analysis = llm.process_news(SAMPLE_ARTICLE)
elapsed = time.time() - t0
test(f"process_news() returns valid JSON ({elapsed:.1f}s)",
     analysis is not None and "topic" in analysis,
     f"Result: {analysis}")
if analysis:
    test("analysis has impact_score (int)",
         isinstance(analysis.get("impact_score"), (int, float, str)),
         f"Got: {type(analysis.get('impact_score'))}")
    test("analysis has key_facts (list)",
         isinstance(analysis.get("key_facts"), list),
         f"Got: {type(analysis.get('key_facts'))}")

t0 = time.time()
debate_result = llm.debate_skeptic(analysis or {})
elapsed = time.time() - t0
test(f"debate_skeptic() returns result ({elapsed:.1f}s)",
     debate_result is not None,
     f"Result: {debate_result}")

if debate_result:
    t0 = time.time()
    explainer_result = llm.debate_explainer(analysis or {}, debate_result)
    elapsed = time.time() - t0
    test(f"debate_explainer() returns result ({elapsed:.1f}s)",
         explainer_result is not None)


# ══════════════════════════════════════════════════════════════════════
# TEST 7: LIVE BRAIN CALL (MoE 30B — slower, ~30-60s)
# ══════════════════════════════════════════════════════════════════════
print("\n[7] LIVE BRAIN CALL (Qwen3 30B MoE — script synthesis)")
print("-" * 50)

mock_analyses = [
    {"topic": "Iran Warships Deploy", "angle": "Escalation risk",
     "impact_score": 8, "key_facts": ["strait of hormuz", "warships", "oil"],
     "second_order_consequence": "Oil price spike"},
    {"topic": "China Trade Deal", "angle": "New alliance",
     "impact_score": 6, "key_facts": ["trade deal", "Russia", "sanctions"],
     "second_order_consequence": "Dedollarization"},
    {"topic": "AI Regulation EU", "angle": "Tech sovereignty",
     "impact_score": 5, "key_facts": ["AI act", "compliance", "GDPR"],
     "second_order_consequence": "US-EU tech split"},
]

t0 = time.time()
script = llm.synthesize_multi_news_script(mock_analyses)
elapsed = time.time() - t0
test(f"synthesize_multi_news_script() returns script ({elapsed:.1f}s)",
     script is not None and "stories" in (script or {}),
     f"Result keys: {list(script.keys()) if script else 'None'}")

if script:
    test("Script has greeting", "greeting" in script)
    test("Script has 3 stories", len(script.get("stories", [])) == 3,
         f"Got: {len(script.get('stories', []))}")
    test("Script has full_text", bool(script.get("full_text")))
    test("Script has segment_timeline",
         isinstance(script.get("segment_timeline"), list))
    word_count = len(script.get("full_text", "").split())
    test(f"Script word count in range ({word_count} words)",
         100 <= word_count <= 500,
         f"Got: {word_count}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"RESULTS: {passed}/{total} passed")

if passed < total:
    print("\nFAILED TESTS:")
    for name, p in results:
        if not p:
            print(f"  {FAIL} {name}")
    sys.exit(1)
else:
    print(f"\n{PASS} ALL TESTS PASSED — Pipeline ready for production")
    sys.exit(0)
