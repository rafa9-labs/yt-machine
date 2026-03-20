from llm_interface import LLMInterface
import json

def test_llm_connection():
    print("=" * 60)
    print("LLM INTERFACE TEST")
    print("=" * 60)
    
    llm = LLMInterface()
    
    print("\n[TEST 1] Checking Ollama connection...")
    if llm.check_connection():
        print("  ✓ Ollama is running and accessible")
    else:
        print("  ✗ Cannot connect to Ollama")
        print("  Please ensure Ollama is running on http://localhost:11434")
        print("  Run: ollama serve")
        return False
    
    print("\n[TEST 2] Warming up model (first load takes 30-60s)...")
    if not llm.warmup_model():
        print("  ✗ Model warmup failed")
        print("  The model may still be loading. Try running the test again.")
        return False
    print("  ✓ Model loaded and ready")
    
    print("\n[TEST 3] Testing basic generation...")
    response = llm.generate(
        prompt="Explain quantum computing in exactly one sentence.",
        temperature=0.7,
        max_tokens=100
    )
    if response:
        print(f"  ✓ Response received: {response[:100]}...")
    else:
        print("  ✗ No response received")
        return False
    
    print("\n[TEST 4] Testing news processing...")
    sample_article = """
    Scientists at MIT have developed a new quantum computer that can solve complex 
    optimization problems 1000x faster than classical supercomputers. The breakthrough 
    uses a novel error-correction technique that maintains quantum coherence for up to 
    10 minutes, a significant improvement over previous systems. However, the technology 
    still requires temperatures near absolute zero to function, limiting practical applications.
    """
    
    news_analysis = llm.process_news(sample_article)
    if news_analysis:
        print("  ✓ News analysis completed:")
        print(f"    Topic: {news_analysis.get('topic', 'N/A')}")
        print(f"    Virality Score: {news_analysis.get('virality_score', 'N/A')}/10")
        print(f"    Keywords: {', '.join(news_analysis.get('keywords', []))}")
    else:
        print("  ✗ News processing failed")
        return False
    
    print("\n[TEST 5] Testing debate system (Skeptic)...")
    skeptic = llm.debate_skeptic(news_analysis)
    if skeptic:
        print("  ✓ Skeptic response:")
        print(f"    Critique: {skeptic.get('critique', 'N/A')[:100]}...")
        print(f"    Question: {skeptic.get('key_question', 'N/A')[:100]}...")
    else:
        print("  ✗ Skeptic debate failed")
        return False
    
    print("\n[TEST 6] Testing debate system (Explainer)...")
    explainer = llm.debate_explainer(news_analysis, skeptic)
    if explainer:
        print("  ✓ Explainer response:")
        print(f"    Explanation: {explainer.get('explanation', 'N/A')[:100]}...")
        print(f"    Analogy: {explainer.get('analogy', 'N/A')[:100]}...")
    else:
        print("  ✗ Explainer debate failed")
        return False
    
    print("\n[TEST 7] Testing script synthesis...")
    script = llm.synthesize_script(news_analysis, skeptic, explainer)
    if script:
        print("  ✓ Script generated:")
        print(f"    Hook: {script.get('hook', 'N/A')}")
        print(f"    Body: {script.get('body', 'N/A')[:80]}...")
        print(f"    Twist: {script.get('twist', 'N/A')}")
        print(f"    CTA: {script.get('cta', 'N/A')}")
        print(f"    Word Count: {script.get('word_count', 'N/A')}")
        print(f"    Estimated Duration: {script.get('estimated_duration', 'N/A')}s")
    else:
        print("  ✗ Script synthesis failed")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - LLM interface operational")
    print("=" * 60)
    print("\nFull script output:")
    print(json.dumps(script, indent=2))
    
    return True

if __name__ == "__main__":
    success = test_llm_connection()
    if not success:
        print("\n⚠ Some tests failed. Check Ollama installation and model availability.")
        print("Run: ollama pull deepseek-r1:latest")
