from llm_interface import LLMInterface

llm = LLMInterface()

print("Testing with available model...")
print("Checking connection...")
if not llm.check_connection():
    print("ERROR: Ollama not running")
    exit(1)

print("Attempting generation with deepseek-r1...")
response = llm.generate(
    prompt="Say 'hello' and nothing else.",
    model="deepseek-r1:latest",
    temperature=0.1,
    max_tokens=20
)

if response:
    print(f"SUCCESS: {response}")
else:
    print("FAILED: No response from deepseek-r1")
    print("\nTrying to pull qwen2.5 (faster alternative)...")
    print("Run this command manually:")
    print("  ollama pull qwen2.5:latest")
    print("\nOr use a smaller model:")
    print("  ollama pull phi3:latest")
