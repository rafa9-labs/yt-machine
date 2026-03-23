#!/usr/bin/env python3
"""Quick test of FAL LoRA pixel art generation"""

from video_server.pixel_art_tool import generate_pixel_art

# Test prompt with military specificity
test_prompt = "F-117 Nighthawk stealth aircraft striking Iraqi air defense radar installation at night, 1991 Gulf War"

print("Testing FAL LoRA pixel art generation...")
print(f"Prompt: {test_prompt}\n")

result = generate_pixel_art(test_prompt)

if result.get('success'):
    print(f"✅ SUCCESS")
    print(f"  Model: {result.get('source')}")
    print(f"  LoRA used: {result.get('lora_used', False)}")
    print(f"  Path: {result.get('path')}")
    print(f"  Specificity: {result.get('specificity_score')}/100")
else:
    print(f"❌ FAILED: {result.get('error')}")
