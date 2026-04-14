"""
═══════════════════════════════════════════════════════════════════════════════
Phase 2 Integration Test — Verify LangChain chains can be built
═══════════════════════════════════════════════════════════════════════════════

This test verifies that:
1. LangChainInterface loads config correctly
2. All chains build without errors
3. PydanticOutputParser generates correct format instructions
4. Models are cached and reused properly

Run:  python -c "import sys; sys.path.insert(0,'.'); exec(open('tests/test_langchain_chains.py').read())"
"""

print("=" * 60)
print("Phase 2: LangChain Chain Construction Tests")
print("=" * 60)

# ── TEST 1: Interface loads config ──
print("\nTEST 1: LangChainInterface config loading")
from brain.langchain_interface import LangChainInterface
interface = LangChainInterface()
print(f"  Base URL: {interface.base_url}")
print(f"  Default model: {interface.default_model[:40]}...")
print(f"  Fallback model: {interface.fallback_model}")
print(f"  Task models: {list(interface.task_models.keys())}")
assert interface.base_url == "http://localhost:11434", "Wrong base URL"
assert "gemma" in interface.default_model.lower() or "gemma" in str(interface.task_models), "Expected gemma model"
print("  PASS ✅")

# ── TEST 2: LLM instances are created and cached ──
print("\nTEST 2: LLM instance creation + caching")
llm1 = interface.get_llm(temperature=0.7, max_tokens=500)
llm2 = interface.get_llm(temperature=0.7, max_tokens=500)  # Same params = cached
llm3 = interface.get_llm(temperature=0.5, max_tokens=1000)  # Different params = new
assert llm1 is llm2, "Expected same cached instance"
assert llm1 is not llm3, "Expected different instance for different params"
print(f"  Cache size: {len(interface._llm_cache)} instances")
print("  PASS ✅")

# ── TEST 3: Fallback chain builds ──
print("\nTEST 3: LLM with fallback chain")
llm_fb = interface.get_llm_with_fallback()
print(f"  Type: {type(llm_fb).__name__}")
print("  PASS ✅")

# ── TEST 4: News analysis chain builds ──
print("\nTEST 4: News analysis chain construction")
from brain.chains.news_analysis import NewsAnalysisChain
news_chain = NewsAnalysisChain(interface)
print(f"  Chain type: {type(news_chain._chain).__name__}")
print("  PASS ✅")

# ── TEST 5: Debate chain builds ──
print("\nTEST 5: Debate chain construction (skeptic + explainer)")
from brain.chains.debate import DebateChain, SkepticResponse, ExplainerResponse
debate = DebateChain(interface)
print(f"  Skeptic chain: {type(debate._skeptic_chain).__name__}")
print(f"  Explainer chain: {type(debate._explainer_chain).__name__}")
# Verify Pydantic models work
s = SkepticResponse(critique="Test critique", key_question="Test question?")
assert s.critique == "Test critique"
print("  PASS ✅")

# ── TEST 6: Curation chain builds (text output) ──
print("\nTEST 6: Curation chain construction (text output)")
from brain.chains.curation import CurationChain
curation = CurationChain(interface)
print(f"  Chain type: {type(curation._chain).__name__}")
print("  PASS ✅")

# ── TEST 7: PydanticOutputParser generates format instructions ──
print("\nTEST 7: Format instructions generated for NewsAnalysis")
from models.schemas import NewsAnalysis
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=NewsAnalysis)
instructions = parser.get_format_instructions()
assert "topic" in instructions.lower(), "Format instructions should mention 'topic'"
assert "impact_score" in instructions, "Format instructions should mention 'impact_score'"
print(f"  Instructions length: {len(instructions)} chars")
print(f"  Contains 'topic': {'topic' in instructions.lower()}")
print(f"  Contains 'impact_score': {'impact_score' in instructions}")
print("  PASS ✅")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED ✅")
print("Phase 2 chains are ready for use with Ollama.")
print("=" * 60)