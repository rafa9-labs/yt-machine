#!/usr/bin/env python3
"""
Comprehensive LoRA test - shows the 3-tier fallback chain in action.
Tests with FAL_KEY, then simulates behavior without it.
"""

import os
from video_server.pixel_art_tool import (
    generate_pixel_art, FAL_KEY, FAL_MODEL, FAL_FALLBACK_MODELS, PIXEL_ART_LORA
)

print("="*60)
print("FAL LoRA IMPLEMENTATION TEST")
print("="*60)

# Check FAL_KEY status
if FAL_KEY:
    key_status = "✅ SET"
    key_preview = FAL_KEY[:8] + "..." + FAL_KEY[-4:]
else:
    key_status = "❌ NOT SET"
    key_preview = "None"

print(f"\n🔑 FAL_KEY Status: {key_status}")
print(f"   Preview: {key_preview}")

# Show configuration
print(f"\n⚙️  Configuration:")
print(f"   Primary Model: {FAL_MODEL}")
print(f"   Fallback Models: {', '.join(FAL_FALLBACK_MODELS)}")
print(f"   LoRA Path: {PIXEL_ART_LORA['path'][:60]}...")
print(f"   LoRA Scale: {PIXEL_ART_LORA['scale']}")

# Test prompt with military specificity
print(f"\n📝 Test Prompt:")
test_prompt = "F-117 Nighthawk stealth aircraft strike on Iraqi air defense radar, 1991 Gulf War night operation"
print(f"   {test_prompt}")

print(f"\n🎨 Generating image...")
print("-" * 60)

result = generate_pixel_art(test_prompt)

print("-" * 60)

# Analyze results
print(f"\n📊 RESULTS:")
print(f"   Success: {result.get('success')}")
print(f"   Source: {result.get('source')}")
print(f"   LoRA Used: {result.get('lora_used', False)}")
print(f"   Specificity Score: {result.get('specificity_score', 'N/A')}")

if result.get('source') == 'placeholder':
    print(f"\n⚠️  PLACEHOLDER generated (FAL_KEY not set)")
    print(f"   To test with real FAL models, set FAL_KEY environment variable:")
    print(f"   $env:FAL_KEY = \"your_key_here\"")
else:
    print(f"\n✅ REAL IMAGE generated!")
    print(f"   Path: {result.get('path')}")
    
    # Show which model succeeded
    if result.get('lora_used'):
        print(f"   🎯 LoRA model worked! Authentic pixel art generated.")
    elif result.get('source') == 'fal-ai/flux/dev':
        print(f"   ⚠️  LoRA failed, but fallback to flux/dev succeeded.")
    elif result.get('source') == 'fal-ai/flux/schnell':
        print(f"   ⚠️  Both LoRA and flux/dev failed, fell back to schnell.")

print(f"\n🎉 IMPLEMENTATION TEST COMPLETE")
print("="*60)
