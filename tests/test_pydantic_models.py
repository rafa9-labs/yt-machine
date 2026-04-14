"""Quick verification that Pydantic models validate correctly."""
from models.schemas import NewsAnalysis
from pydantic import ValidationError

# Test 1: Valid data should work
print("TEST 1: Valid data")
a = NewsAnalysis(topic="Iran sanctions", impact_score=8, key_facts=["uranium enrichment"])
print(f"  PASS: {a.topic} (impact={a.impact_score})")

# Test 2: Bad type should be caught
print("\nTEST 2: Bad type (string instead of int)")
try:
    NewsAnalysis(topic="Test", impact_score="high")
    print("  FAIL: Should have raised ValidationError")
except ValidationError as e:
    print(f"  PASS: Validation caught {e.error_count()} error(s)")
    print(f"  Detail: {e.errors()[0]['msg']}")

# Test 3: Out of range should be caught
print("\nTEST 3: Out of range (impact_score=99)")
try:
    NewsAnalysis(topic="Test", impact_score=99)
    print("  FAIL: Should have raised ValidationError")
except ValidationError as e:
    print(f"  PASS: Validation caught {e.error_count()} error(s)")
    print(f"  Detail: {e.errors()[0]['msg']}")

print("\nAll validation tests complete!")