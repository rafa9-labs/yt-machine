"""Quick smoke test for all 6 improvement areas."""
import sys
sys.path.insert(0, 'c:/Users/rafa/yt-machine')

errors = []

# ── Test 1: LLM Truncation — num_ctx loaded from config ──
from brain.llm_interface import LLMInterface
llm = LLMInterface()
assert llm.num_ctx == 8192, f'Expected 8192, got {llm.num_ctx}'
print(f'[1] LLM num_ctx: {llm.num_ctx}  OK')

# ── Test 2: Geo Accuracy — partial match passes, zero match fails ──
from redfish.geopolitical_accuracy import validate_geopolitical_accuracy, get_theater_countries
r = validate_geopolitical_accuracy('IRGC patrol boats at Strait of Hormuz')
print(f'[2] Geo score IRGC+Hormuz (expect ~100): {r["accuracy_score"]}')
assert r['accuracy_score'] >= 85, f'Expected >=85 got {r["accuracy_score"]}'

r2 = validate_geopolitical_accuracy('some aircraft flew somewhere')
print(f'[2] Geo score generic (expect ~100, no countries detected): {r2["accuracy_score"]}')

tc = get_theater_countries('attack near the strait of hormuz')
print(f'[2] Theater countries for hormuz: {tc}')
assert 'iran' in tc, f'Expected iran in theater countries, got {tc}'

# ── Test 3: Action Extraction — phrasal verbs + irregular pasts ──
from redfish.script_parser import ScriptParser
sp = ScriptParser()

test_cases = [
    ('Iran came under fire from Israeli jets', 'strike'),
    ('Forces were poised to strike the facility', 'strike'),
    ('The US struck three missile sites yesterday', 'strike'),
    ('Prices surged to 112 dollars per barrel', 'surge'),
    ('Troops stepped up their operations in the region', 'escalate'),
    ('The carrier group sailed into the Persian Gulf', 'patrol'),  # sailed -> patrol
    ('Leaders signed a ceasefire agreement today', 'sign'),
]
for text, expected in test_cases:
    action = sp._extract_primary_action(text)
    ok = 'OK' if action == expected else f'FAIL (got {action})'
    print(f'[3] "{text[:45]}" -> {action}  {ok}')
    if action != expected:
        errors.append(f'Action test failed: "{text}" -> {action} != {expected}')

# ── Test 4: Script Relevance — stem matching ──
from redfish.prompt_validator import calculate_prompt_relevance
prompt = 'IRGC patrol boats conducting naval blockade in strait of Hormuz, Iranian Revolutionary Guard'
article = 'Iran deployed patrol boats near Hormuz blocking oil tankers'
score = calculate_prompt_relevance(prompt, article)
print(f'[4] Relevance score: {score}  (target >35)')
assert score >= 30, f'Expected >=30 got {score}'

# Also test alias table hit
prompt2 = 'Iranian Revolutionary Guard green and amber camouflage uniforms at Hormuz'
article2 = 'Iran IRGC forces deployed at the strait'
score2 = calculate_prompt_relevance(prompt2, article2)
print(f'[4] Alias relevance score: {score2}  (alias-only baseline: 15pts)')
assert score2 >= 15, f'Expected alias hit >=15 got {score2}'

# ── Test 5: Prompt Specificity — named entity extraction + quality check ──
from redfish.prompt_validator import validate_prompt_quality
prompt3 = '(F-35I Adir:1.4), (IRGC:1.3), Iron Dome interception, over Tel Aviv, isometric pixel art, (dramatic angle:1.2)'
result = validate_prompt_quality(prompt3)
print(f'[5] Quality score: {result["score"]}  checks: {result["checks"]}')
assert result['score'] >= 57, f'Expected >=57 got {result["score"]}'

# Named entity extraction
entities = sp._extract_named_entities('The F-35I Adir intercepted IRGC drones over the Strait of Hormuz. Iron Dome activated.')
print(f'[5] Named entities: {entities}')
assert any('F-35' in e or 'F-35I' in e for e in entities), f'Expected F-35 in {entities}'

# Specificity modifiers
mods = sp._extract_specificity_modifiers('Oil prices hit $112 per barrel. 3 carrier groups deployed.', [])
print(f'[5] Specificity modifiers: {mods}')
assert len(mods) >= 1, f'Expected at least 1 modifier, got {mods}'

# ── Test 6: Trending Boost — get_top_terms + injection ──
from redfish.trending_analyzer import TrendingAnalyzer
ta = TrendingAnalyzer()
mock_ctx = {
    'iran': {'score': 0.97, 'category': 'military'},
    'israel': {'score': 0.65, 'category': 'military'},
    'oil': {'score': 0.31, 'category': 'economic'},
    'noise': {'score': 0.1, 'category': 'general'},
}
top = ta.get_top_terms(mock_ctx, n=3, min_score=0.3)
print(f'[6] Top trending terms: {top}')
assert 'iran' in top, f'Expected iran in {top}'
assert 'noise' not in top, f'noise should be filtered (score<0.3)'

concepts = sp.extract_visual_concepts(
    'Iran fired missiles at Israel. Oil prices surged to 112 dollars.',
    trending_context=mock_ctx
)
print(f'[6] top_trending_terms: {concepts.get("top_trending_terms")}')
print(f'[6] named_entities:     {concepts.get("named_entities")}')
print(f'[6] script_anchors:     {concepts.get("script_anchors")}')
print(f'[6] specificity_mods:   {concepts.get("specificity_modifiers")}')

# top_trending_terms should contain iran and/or israel since they appear in text
ttt = concepts.get('top_trending_terms', [])
assert len(ttt) >= 1, f'Expected at least 1 trending term, got {ttt}'

if errors:
    print(f'\nFAILED: {len(errors)} error(s):')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('\n=== ALL 6 IMPROVEMENT TESTS PASSED ===')
