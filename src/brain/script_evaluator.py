"""
Script Evaluator — Semantic Dedup + LLM Continuity Critic

Runs AFTER script synthesis (step 4) and BEFORE visual prompt generation (step 4.5).

Sprint 1: Semantic dedup — embed stories, detect near-duplicate pairs (>0.90 cosine),
           merge the weaker, regenerate a replacement story from unused news analyses.

Sprint 2: Continuity critic — fast LLM pass to check narrative consistency,
           flag contradictions, entity discontinuity, timeline errors.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_MODEL = None


def _get_embedding_model():
    global _MODEL
    if _MODEL is None:
        if not _HAS_EMBEDDINGS:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL


def _story_text(story: dict) -> str:
    parts = [
        story.get('part_1_narration', ''),
        story.get('part_2_narration', ''),
        story.get('real_talk', ''),
        story.get('fallout', ''),
    ]
    return ' '.join(p for p in parts if p).strip()


def compute_story_similarities(
    stories: List[dict],
) -> List[Tuple[int, int, float]]:
    """
    Compute pairwise cosine similarity between stories.
    Returns list of (idx_a, idx_b, similarity_score).
    """
    if not _HAS_EMBEDDINGS or not _HAS_NUMPY:
        print("  [EVAL] sentence-transformers or numpy not available — skipping dedup")
        return []

    model = _get_embedding_model()
    texts = [_story_text(s) for s in stories]
    if not any(texts):
        return []

    embeddings = model.encode(texts, show_progress_bar=False)
    sim_matrix = _cosine_sim(embeddings)

    pairs = []
    n = len(stories)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sim_matrix[i][j])))
    return pairs


def detect_duplicates(
    stories: List[dict],
    threshold: float = 0.90,
) -> List[Tuple[int, int, float]]:
    """
    Return pairs of stories whose cosine similarity exceeds the threshold.
    Each pair is (weaker_idx, stronger_idx, score).
    """
    pairs = compute_story_similarities(stories)
    dups = []
    for i, j, score in pairs:
        if score >= threshold:
            len_i = len(_story_text(stories[i]).split())
            len_j = len(_story_text(stories[j]).split())
            weaker = i if len_i <= len_j else j
            stronger = j if weaker == i else i
            dups.append((weaker, stronger, score))
    return dups


def merge_and_replace(
    script: dict,
    news_analyses: List[dict],
    dup_pairs: List[Tuple[int, int, float]],
    llm_interface=None,
) -> dict:
    """
    For each duplicate pair, merge the weaker story into the stronger one,
    then request the LLM to generate a replacement story using an unused
    news analysis.

    Returns the updated script.
    """
    stories = script.get('stories', [])
    if not stories:
        return script

    used_indices = set(range(len(stories)))
    replaced = set()

    for weaker, stronger, score in dup_pairs:
        if weaker in replaced:
            continue

        print(f"  [EVAL-DEDUP] Stories {weaker+1} and {stronger+1} are {score:.2f} similar — merging {weaker+1} into {stronger+1}")

        strong_text = _story_text(stories[stronger])
        weak_text = _story_text(stories[weaker])

        if len(strong_text.split()) >= len(weak_text.split()):
            keep = stronger
            drop = weaker
        else:
            keep = weaker
            drop = stronger

        used_topic_words = set()
        for s_idx in range(len(stories)):
            if s_idx != drop:
                for w in _story_text(stories[s_idx]).lower().split():
                    used_topic_words.add(w)

        best_analysis_idx = None
        best_diff = -1
        for a_idx, analysis in enumerate(news_analyses):
            if a_idx in used_indices and a_idx < len(stories):
                continue
            analysis_text = (
                analysis.get('topic', '') + ' ' +
                ' '.join(analysis.get('key_facts', [])) + ' ' +
                analysis.get('angle', '')
            ).lower()
            analysis_words = set(analysis_text.split())
            diff = len(analysis_words - used_topic_words)
            if diff > best_diff:
                best_diff = diff
                best_analysis_idx = a_idx

        replacement = None
        if llm_interface and best_analysis_idx is not None:
            replacement = _generate_replacement_story(
                llm_interface, script, news_analyses[best_analysis_idx], stories[keep]
            )

        if replacement:
            stories[drop] = replacement
            print(f"  [EVAL-DEDUP] Replaced story {drop+1} with new topic: {news_analyses[best_analysis_idx].get('topic', 'N/A')[:60]}")
        else:
            merged = _merge_stories(stories[keep], stories[drop])
            stories[drop] = merged
            print(f"  [EVAL-DEDUP] Merged story {drop+1} into {keep+1} (no LLM available for replacement)")

        replaced.add(drop)

    script['stories'] = stories

    if replaced:
        full_parts = []
        for s in stories:
            full_parts.append(s.get('part_1_narration', ''))
            full_parts.append(s.get('part_2_narration', ''))
            full_parts.append(s.get('real_talk', ''))
            full_parts.append(s.get('fallout', ''))
            full_parts.append(s.get('segue', ''))
        script['full_text'] = ' '.join(p for p in full_parts if p)

    return script


def _generate_replacement_story(
    llm_interface,
    script: dict,
    analysis: dict,
    reference_story: dict,
) -> Optional[dict]:
    """
    Ask the LLM to generate a replacement story from a fresh news analysis.
    Returns a story dict or None.
    """
    prompt = f"""Generate ONE story for a Mask-style news video. The existing stories cover similar ground, so we need a FRESH angle.

NEW TOPIC: {analysis.get('topic', 'N/A')}
KEY FACTS: {', '.join(analysis.get('key_facts', []))}
ANGLE: {analysis.get('angle', 'N/A')}
IMPACT: {analysis.get('impact_score', 5)}/10

REFERENCE STYLE (match this tone and structure):
- part_1_narration: {reference_story.get('part_1_narration', '')[:100]}...
- part_2_narration: {reference_story.get('part_2_narration', '')[:100]}...
- real_talk: {reference_story.get('real_talk', '')[:60]}...

CRITICAL RULES:
- Output ONLY a JSON object with keys: part_1_narration, part_2_narration, part_1_visual, part_2_visual, real_talk, fallout, segue
- part_1 = THE HOOK (cartoonish entrance, Looney Tunes metaphors)
- part_2 = THE PAYOFF (speed-talk facts, dense information)
- real_talk = SERIOUS drop-the-act moment (NO caps, NO exclamations, flat truth)
- fallout = WHAT HAPPENS NEXT — the second-order consequence, the domino that falls after (10-14 words)
- segue = frantic cartoonish transition to next story, bridging FROM fallout (8-15 words)
- GEOGRAPHIC ANCHOR: every country name on first mention must carry a regional descriptor
- CTA QUARANTINE: NO subscribe/like/share/sign-off text in any narration field
- Target: 18-22 words per part_1, 22-28 words per part_2, 12-16 words for real_talk, 10-14 words for fallout
- NEVER repeat topics from existing stories"""

    try:
        response = llm_interface.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=500,
            task_name="script_evaluation"
        )
        if not response:
            return None

        story = llm_interface._extract_json(response)
        if not story or not isinstance(story, dict):
            return None

        required = ['part_1_narration', 'part_2_narration', 'real_talk', 'fallout']
        for key in required:
            if key not in story or len(story.get(key, '').split()) < 5:
                return None

        return story

    except Exception as e:
        print(f"  [EVAL-DEDUP] Replacement story generation failed: {e}")
        return None


def _merge_stories(strong: dict, weak: dict) -> dict:
    """
    Merge two stories by taking the stronger content + the weaker real_talk as backup.
    """
    merged = dict(strong)
    for key in ('part_1_narration', 'part_2_narration'):
        strong_len = len(strong.get(key, '').split())
        weak_len = len(weak.get(key, '').split())
        if weak_len > strong_len:
            merged[key] = weak[key]
    if not merged.get('real_talk') or len(merged.get('real_talk', '').split()) < 5:
        merged['real_talk'] = weak.get('real_talk', 'The truth is always stranger than fiction.')
    return merged


def evaluate_continuity(
    script: dict,
    llm_interface=None,
) -> Tuple[dict, List[dict]]:
    """
    LLM continuity critic. Checks for contradictory facts, entity
    discontinuity, and timeline errors across all stories.

    Returns (updated_script, issues_list).
    """
    stories = script.get('stories', [])
    if not stories or not llm_interface:
        return script, []

    story_texts = []
    for i, s in enumerate(stories):
        story_texts.append(
            f"Story {i+1}:\n"
            f"  part_1: {s.get('part_1_narration', '')}\n"
            f"  part_2: {s.get('part_2_narration', '')}\n"
            f"  real_talk: {s.get('real_talk', '')}\n"
            f"  fallout: {s.get('fallout', '')}\n"
            f"  segue: {s.get('segue', '')}"
        )

    prompt = f"""Analyze these {len(stories)} news stories for narrative consistency issues.

{chr(10).join(story_texts)}

Check for:
1. CONTRADICTORY FACTS: If story A says "X was destroyed" and story B says "X is thriving" without explanation.
2. ENTITY DISCONTINUITY: Characters/countries that appear, vanish, and reappear inconsistently.
3. TIMELINE ERRORS: Events described in impossible temporal order.
4. REPETITION: Same phrases, facts, or metaphors reused across stories (excluding the Mask persona style).

If NO issues found, respond with: {{"issues": []}}

If issues ARE found, respond with a JSON object:
{{
  "issues": [
    {{
      "type": "contradiction|entity|timeline|repetition",
      "story_index": 0,
      "field": "part_1_narration|part_2_narration|real_talk|fallout|segue",
      "description": "What's wrong",
      "suggested_fix": "The corrected text for this field"
    }}
  ]
}}

Return ONLY the JSON object. No explanation before or after."""

    try:
        response = llm_interface.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
            task_name="script_evaluation"
        )
        if not response:
            return script, []

        result = llm_interface._extract_json(response)
        if not result or not isinstance(result, dict):
            return script, []

        issues = result.get('issues', [])
        if not issues:
            print("  [EVAL-CRITIC] No continuity issues found")
            return script, []

        print(f"  [EVAL-CRITIC] Found {len(issues)} continuity issue(s)")
        updated = apply_continuity_fixes(script, issues)
        return updated, issues

    except Exception as e:
        print(f"  [EVAL-CRITIC] Continuity check failed: {e}")
        return script, []


def apply_continuity_fixes(script: dict, issues: List[dict]) -> dict:
    """
    Apply specific field-level fixes from the continuity critic.
    Only overwrites a field if the suggested fix is reasonable:
    - Longer than 5 words
    - Doesn't introduce CTA text
    - Preserves the Mask persona style
    """
    stories = script.get('stories', [])
    cta_pattern = re.compile(
        r'(?:subscribe|like\s+and\s+share|follow\s+for\s+more|thanks\s+for\s+watching)',
        re.IGNORECASE
    )

    for issue in issues:
        idx = issue.get('story_index', -1)
        field = issue.get('field', '')
        suggested = issue.get('suggested_fix', '')

        if idx < 0 or idx >= len(stories):
            continue
        if field not in ('part_1_narration', 'part_2_narration', 'real_talk', 'fallout', 'segue'):
            continue
        if len(suggested.split()) < 5:
            continue
        if cta_pattern.search(suggested):
            print(f"  [EVAL-CRITIC] Skipping fix for story {idx+1} {field}: contains CTA text")
            continue

        original = stories[idx].get(field, '')
        if len(suggested.split()) >= len(original.split()) * 0.5:
            stories[idx][field] = suggested
            print(f"  [EVAL-CRITIC] Fixed story {idx+1} {field}: \"{suggested[:60]}...\"")
        else:
            print(f"  [EVAL-CRITIC] Skipping fix for story {idx+1} {field}: suggested text too short relative to original")

    script['stories'] = stories

    full_parts = []
    for s in stories:
        full_parts.append(s.get('part_1_narration', ''))
        full_parts.append(s.get('part_2_narration', ''))
        full_parts.append(s.get('real_talk', ''))
        full_parts.append(s.get('fallout', ''))
        full_parts.append(s.get('segue', ''))
    greeting = script.get('greeting', '')
    intro = script.get('intro_hook', '')
    closing = script.get('closing', '')
    script['full_text'] = ' '.join(p for p in [greeting, intro] + full_parts + [closing] if p)

    return script


def run_script_evaluation(
    script: dict,
    news_analyses: List[dict],
    llm_interface=None,
    similarity_threshold: float = 0.90,
) -> dict:
    """
    Full pipeline: semantic dedup → continuity critic.

    1. Detect duplicate stories (cosine similarity > threshold)
    2. Merge + replace duplicates
    3. Run continuity critic
    4. Apply fixes
    5. Rebuild segment_timeline and full_text

    Returns the updated script dict.
    """
    stories = script.get('stories', [])
    if len(stories) < 2:
        print("  [EVAL] Fewer than 2 stories — skipping dedup")
    else:
        dup_pairs = detect_duplicates(stories, threshold=similarity_threshold)
        if dup_pairs:
            print(f"  [EVAL-DEDUP] Found {len(dup_pairs)} duplicate pair(s)")
            script = merge_and_replace(script, news_analyses, dup_pairs, llm_interface)
        else:
            print("  [EVAL-DEDUP] No duplicates found — all stories are distinct")

    if llm_interface:
        script, issues = evaluate_continuity(script, llm_interface)
        if issues:
            print(f"  [EVAL-CRITIC] Applied {len(issues)} continuity fix(es)")
    else:
        print("  [EVAL-CRITIC] No LLM interface — skipping continuity check")

    if script.get('stories'):
        from src.brain.llm_interface import LLMInterface
        if '_enforce_segues' in dir(llm_interface) and isinstance(llm_interface, LLMInterface):
            script = llm_interface._enforce_segues(script)
        if '_enforce_fallout' in dir(llm_interface) and isinstance(llm_interface, LLMInterface):
            script = llm_interface._enforce_fallout(script, news_analyses)

    if script.get('stories') and script.get('segment_timeline'):
        _rebuild_timeline(script)

    print(f"  [EVAL] Evaluation complete — {len(script.get('stories', []))} stories")
    return script


def _rebuild_timeline(script: dict) -> None:
    """
    Rebuild segment_timeline from stories after dedup/continuity fixes.
    """
    stories = script.get('stories', [])
    greeting = script.get('greeting', '')
    intro_hook = script.get('intro_hook', '')
    closing = script.get('closing', '')

    timeline = []

    intro_text = f"{greeting} {intro_hook}".strip()
    if intro_text:
        timeline.append({'text': intro_text, 'image_idx': 0, 'label': 'intro'})
    timeline.append({'text': '...', 'image_idx': 0, 'label': 'intro_pause', 'is_separator': True})

    for i, story in enumerate(stories):
        img_base = i * 4

        part_1 = story.get('part_1_narration', '')
        if part_1:
            timeline.append({'text': part_1, 'image_idx': img_base, 'label': f'story_{i+1}_part1'})

        part_2 = story.get('part_2_narration', '')
        real_talk = story.get('real_talk', '')

        if real_talk and real_talk.strip() in part_2:
            part_2 = part_2.replace(real_talk.strip(), '').strip()
            part_2 = re.sub(r'\s*[-—]+\s*$', '', part_2).strip()

        closing_text = script.get('closing', '')
        if closing_text and i == len(stories) - 1 and closing_text.strip() in part_2:
            part_2 = part_2.replace(closing_text.strip(), '').strip()
            part_2 = re.sub(r'\s*[-—]+\s*$', '', part_2).strip()

        if part_2:
            timeline.append({'text': part_2, 'image_idx': img_base + 1, 'label': f'story_{i+1}_part2'})

        if real_talk:
            timeline.append({'text': real_talk, 'image_idx': img_base + 2, 'label': f'story_{i+1}_real_talk'})

        fallout = story.get('fallout', '')
        if fallout:
            timeline.append({'text': fallout, 'image_idx': img_base + 3, 'label': f'story_{i+1}_fallout'})

        segue = story.get('segue', '')
        if segue and i < len(stories) - 1:
            timeline.append({'text': segue, 'image_idx': img_base + 3, 'label': f'story_{i+1}_segue'})

        if i < len(stories) - 1:
            timeline.append({'text': '....', 'image_idx': img_base + 3, 'label': f'story_{i+1}_separator', 'is_separator': True})

    timeline.append({'text': closing, 'image_idx': (len(stories) - 1) * 4 + 3, 'label': 'closing'})

    script['segment_timeline'] = timeline

    existing_scenes = script.get('all_visual_scenes', [])
    visual_scenes = []
    narration_fields = ['part_1_narration', 'part_2_narration', 'real_talk', 'fallout']
    for i, story in enumerate(stories):
        field_defaults = [
            ('part_1_visual', story.get('mini_hook', story.get('part_1_narration', ''))),
            ('part_2_visual', story.get('body', story.get('part_2_narration', ''))),
            ('real_talk_visual', story.get('part_2_visual', story.get('real_talk', ''))),
            ('fallout_visual', story.get('second_order_visual', story.get('fallout_visual', story.get('fallout', '')))),
        ]
        scene_names = [f'story_{i+1}_part1', f'story_{i+1}_part2',
                       f'story_{i+1}_real_talk', f'story_{i+1}_fallout']
        for j, (field, fallback_text) in enumerate(field_defaults):
            existing_desc = ''
            for es in existing_scenes:
                if es.get('scene') == scene_names[j]:
                    existing_desc = es.get('description', '')
                    break
            desc = existing_desc or story.get(field, fallback_text)
            if not desc:
                desc = story.get(narration_fields[j], '')
            visual_scenes.append({'scene': scene_names[j], 'description': desc})
    script['all_visual_scenes'] = visual_scenes

    full_parts = []
    for seg in timeline:
        if seg.get('is_separator'):
            full_parts.append(seg['text'])
        elif seg.get('text'):
            full_parts.append(seg['text'])
    script['full_text'] = ' '.join(filter(None, full_parts))