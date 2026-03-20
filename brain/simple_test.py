from llm_interface import LLMInterface

llm = LLMInterface()

print("Testing script synthesis with simpler input...")

simple_analysis = {
    "topic": "AI Breakthrough",
    "key_facts": ["Fast", "Revolutionary", "Game-changing"],
    "angle": "But is it safe?",
    "keywords": ["AI", "tech"]
}

simple_skeptic = {
    "critique": "Seems too good to be true",
    "key_question": "What are the risks?"
}

simple_explainer = {
    "explanation": "It's actually well-tested",
    "analogy": "Like a car with airbags"
}

print("\nGenerating script...")
script = llm.synthesize_script(simple_analysis, simple_skeptic, simple_explainer)

if script:
    print("\n✓ SUCCESS! Script generated:")
    print(f"  Hook: {script.get('hook', 'N/A')}")
    print(f"  Body: {script.get('body', 'N/A')[:100]}...")
    print(f"  Twist: {script.get('twist', 'N/A')}")
    print(f"  CTA: {script.get('cta', 'N/A')}")
    print(f"  Word Count: {script.get('word_count', 'N/A')}")
    print(f"  Duration: {script.get('estimated_duration', 'N/A')}s")
else:
    print("\n✗ FAILED")
