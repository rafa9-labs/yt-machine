"""
Utility functions for the video generation pipeline.
Extracted here so they can be imported without triggering pipeline side effects.
"""
import re


def bridge_timestamp_gaps(image_times: list, total_dur: float) -> list:
    """
    Fill None values, bridge gaps, and enforce minimum duration for scene timestamps.
    
    Args:
        image_times: List of {'start': float|None, 'end': float|None}
        total_dur: Total audio duration in seconds
    
    Returns:
        List of {'start': float, 'end': float} with no gaps, all values filled
    """
    num_images = len(image_times)
    
    # Fill None values
    for i, it in enumerate(image_times):
        if it['start'] is None:
            it['start'] = (total_dur / num_images) * i if total_dur > 0 else 0
        if it['end'] is None:
            if i + 1 < num_images and image_times[i + 1]['start'] is not None:
                it['end'] = image_times[i + 1]['start']
            else:
                it['end'] = (total_dur / num_images) * (i + 1)
    
    # Ensure boundaries
    if image_times:
        image_times[-1]['end'] = max(image_times[-1]['end'], total_dur)
        image_times[0]['start'] = 0
    
    # Bridge gaps: extend images so there are NO black frames
    for i in range(len(image_times) - 1):
        current_end = image_times[i]['end']
        next_start = image_times[i + 1]['start']
        gap = next_start - current_end
        if gap > 0.1:
            split_point = current_end + gap * 0.7
            image_times[i]['end'] = split_point
            image_times[i + 1]['start'] = split_point
    
    # Ensure minimum duration per image (at least 1 second)
    for i, it in enumerate(image_times):
        dur = it['end'] - it['start']
        if dur < 1.0:
            needed = 1.0 - dur
            if i > 0:
                steal = min(needed / 2, max(0, image_times[i-1]['end'] - image_times[i-1]['start'] - 1.0))
                if steal > 0:
                    image_times[i-1]['end'] -= steal
                    it['start'] -= steal
                    needed -= steal
            if needed > 0 and i < len(image_times) - 1:
                steal = min(needed, max(0, image_times[i+1]['end'] - image_times[i+1]['start'] - 1.0))
                if steal > 0:
                    image_times[i+1]['start'] += steal
                    it['end'] += steal
    
    return image_times


def build_fallback_prompt(narration_text: str, story_idx: int, part_idx: int,
                          news_analyses: list) -> str:
    """
    Build a script-aware fallback visual prompt from narration text.
    Extracts concrete nouns, locations, and entities to avoid generic prompts.
    """
    if not narration_text or len(narration_text) < 10:
        if story_idx < len(news_analyses):
            topic = news_analyses[story_idx].get('topic', '')
            if topic:
                return f"Strategic overview of {topic}, geopolitical tension, dramatic scene"
        return "Geopolitical strategic map, world leaders in tension, dramatic lighting"
    
    text = narration_text.lower()
    
    locations = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', narration_text)
    skip = {'The','This','That','And','But','For','Not','In','On','At','With','It','Is',
            'Was','Are','Were','Has','Have','Had','Been','Will','Would','Could','Should',
            'May','Might','They','Their','There','These','Those','Each','Every','Which',
            'What','When','Where','Who','How','Why','More','Most','Some','Such','Than',
            'Then','Now','Just','Also','Very','Even','Still','Only','About','After',
            'Before','Between','Through','During','Without','Against','Another','While',
            'Last','First','Next','Both','All','Many','Much','Own','Other','New','Old',
            'Good','Great','Big','Small'}
    geo_locations = [loc for loc in locations if loc not in skip]
    
    action_words = []
    for verb in ['strikes','attacks','blockades','deploy','signs','negotiates',
                  'collapses','surges','protests','evacuates','launches','invades',
                  'sanctions','threatens','agrees','rejects','builds','destroys']:
        if verb in text:
            action_words.append(verb)
    
    numbers = re.findall(r'\b\d+[\d,]*\b', narration_text)
    
    parts = []
    if geo_locations:
        parts.append(f"scene in {' '.join(geo_locations[:2])}")
    if action_words:
        parts.append(f"{' '.join(action_words[:2])}")
    if numbers:
        parts.append(f"involving {numbers[0]} units")
    
    if part_idx == 0:
        parts.append("wide establishing shot")
    else:
        parts.append("dramatic close-up, tension")
    
    if len(parts) < 2 and story_idx < len(news_analyses):
        topic = news_analyses[story_idx].get('topic', '')
        if topic:
            parts.append(topic)
    
    desc = ', '.join(parts)
    if len(desc) < 20:
        desc = "Geopolitical strategic map, world leaders in tension, dramatic lighting"
    
    return desc