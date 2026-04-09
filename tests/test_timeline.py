"""Quick test of the segment timeline building logic."""

# Mock script with new part_1/part_2 format
script = {
    'greeting': 'Good Evening!',
    'intro_hook': 'Three wild stories today.',
    'stories': [
        {'part_1_narration': 'Story 1 setup happened.', 'part_1_visual': 'Scene 1A',
         'part_2_narration': 'Story 1 punchline.', 'part_2_visual': 'Scene 1B',
         'transition': 'Moving on...'},
        {'part_1_narration': 'Story 2 setup.', 'part_1_visual': 'Scene 2A',
         'part_2_narration': 'Story 2 punchline.', 'part_2_visual': 'Scene 2B',
         'transition': 'And finally...'},
        {'part_1_narration': 'Story 3 setup big one.', 'part_1_visual': 'Scene 3A',
         'part_2_narration': 'Story 3 punchline climax.', 'part_2_visual': 'Scene 3B',
         'transition': ''},
    ],
    'closing': 'Subscribe! I am Masker. See you tomorrow!'
}

greeting = script['greeting']

segment_timeline = []
intro_text = f"{greeting} {script.get('intro_hook', '')}".strip()
segment_timeline.append({'text': intro_text, 'image_idx': 0, 'label': 'intro'})

for i, story in enumerate(script['stories']):
    img_base = i * 2
    part_1 = story.get('part_1_narration', '')
    if part_1:
        segment_timeline.append({'text': part_1, 'image_idx': img_base, 'label': f'story_{i+1}_part1'})
    part_2 = story.get('part_2_narration', '')
    if part_2:
        segment_timeline.append({'text': part_2, 'image_idx': img_base + 1, 'label': f'story_{i+1}_part2'})
    transition = story.get('transition', '')
    if transition:
        segment_timeline.append({'text': transition, 'image_idx': img_base + 1, 'label': f'story_{i+1}_transition'})
    if i < len(script['stories']) - 1:
        segment_timeline.append({'text': '....', 'image_idx': img_base + 1, 'label': f'story_{i+1}_separator', 'is_separator': True})

closing = script.get('closing', '')
segment_timeline.append({'text': closing, 'image_idx': 5, 'label': 'closing'})

print('TIMELINE:')
for seg in segment_timeline:
    print(f'  [{seg["label"]}] img#{seg["image_idx"]}: "{seg["text"]}"')

print(f'\nTotal segments: {len(segment_timeline)}')
print(f'Images used: {max(s["image_idx"] for s in segment_timeline) + 1}')

# Build full_text
full_parts = [seg['text'] for seg in segment_timeline]
full_text = ' '.join(filter(None, full_parts))
print(f'\nFull text: "{full_text}"')
print(f'Word count: {len(full_text.split())}')

# Verify image mapping
print('\nIMAGE MAPPING:')
from collections import defaultdict
img_segments = defaultdict(list)
for seg in segment_timeline:
    img_segments[seg['image_idx']].append(seg['label'])
for img_idx in sorted(img_segments.keys()):
    print(f'  Image {img_idx}: {img_segments[img_idx]}')