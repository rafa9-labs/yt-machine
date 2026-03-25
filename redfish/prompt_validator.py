"""
Prompt Quality Validator - Ensures generated prompts meet quality standards
Validates prompts for specificity, accuracy, and relevance
Enhanced with geopolitical accuracy validation
"""

import re
from typing import Dict, Any, List
from .military_equipment_db import get_all_locations, MILITARY_EQUIPMENT_DB
from .geopolitical_validator import GeopoliticalValidator


def validate_prompt_quality(prompt: str) -> Dict[str, Any]:
    """
    Enhanced prompt quality validation with keyword presence detection.
    
    Args:
        prompt: Generated image prompt
        
    Returns:
        Dictionary with validation results
    """
    checks = {
        'has_specific_equipment': _check_specific_equipment(prompt),
        'has_real_location': _check_real_location(prompt),
        'has_action_verb': _check_action_verb(prompt),
        'no_generic_terms': _check_no_generic_terms(prompt),
        'has_style_suffix': _check_style_suffix(prompt),
        'has_token_weighting': _check_token_weighting(prompt),
        'has_visual_grounding': _check_visual_grounding(prompt)
    }
    
    passed = sum(checks.values())
    total = len(checks)
    score = (passed / total) * 100
    
    return {
        'score': score,
        'checks': checks,
        'passed': passed >= 5,  # Must pass at least 5/7 checks now
        'details': _get_check_details(checks)
    }


def check_keyword_presence(prompt: str, critical_keywords: List[str]) -> Dict[str, Any]:
    """
    Check if critical keywords are present and properly weighted in the prompt.
    
    Args:
        prompt: Generated image prompt
        critical_keywords: List of keywords that must be present
        
    Returns:
        Dictionary with presence analysis
    """
    prompt_lower = prompt.lower()
    present_keywords = []
    missing_keywords = []
    weighted_keywords = []
    
    for keyword in critical_keywords:
        if keyword.lower() in prompt_lower:
            present_keywords.append(keyword)
            # Check if keyword is weighted
            if f"({keyword}" in prompt_lower or f"{keyword}:" in prompt_lower:
                weighted_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    return {
        'present': present_keywords,
        'missing': missing_keywords,
        'weighted': weighted_keywords,
        'presence_rate': len(present_keywords) / len(critical_keywords) if critical_keywords else 0,
        'weighting_rate': len(weighted_keywords) / len(present_keywords) if present_keywords else 0,
        'passes': len(missing_keywords) == 0
    }


def validate_visual_type_correlation(prompt: str, visual_type: str) -> Dict[str, Any]:
    """
    Validate that prompt matches the expected visual type characteristics.
    
    Args:
        prompt: Generated image prompt
        visual_type: Expected visual type (military, economic, diplomatic, human_impact)
        
    Returns:
        Validation results
    """
    type_keywords = {
        'military': ['missile', 'tank', 'aircraft', 'warship', 'naval', 'tactical', 'formation'],
        'economic': ['market', 'price', 'trading', 'financial', 'indicator', 'display'],
        'diplomatic': ['summit', 'negotiation', 'official', 'formal', 'treaty', 'agreement'],
        'human_impact': ['civilian', 'people', 'crowd', 'protest', 'evacuee', 'human scale']
    }
    
    expected_keywords = type_keywords.get(visual_type, [])
    prompt_lower = prompt.lower()
    
    found_keywords = [kw for kw in expected_keywords if kw in prompt_lower]
    correlation_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
    
    return {
        'visual_type': visual_type,
        'expected_keywords': expected_keywords,
        'found_keywords': found_keywords,
        'correlation_score': correlation_score,
        'passes': correlation_score >= 0.3  # At least 30% of expected keywords
    }



# Entity alias table: maps enhanced visual descriptions back to their source keywords.
# This bridges the gap between enriched prompt language and raw script words.
_ENTITY_ALIAS_TABLE: Dict[str, List[str]] = {
    # Country visual descriptions → raw country names
    'iranian revolutionary guard': ['iran', 'irgc', 'iranian'],
    'islamic revolutionary guard': ['iran', 'irgc'],
    'green and amber camouflage': ['iran', 'iranian'],
    'persian script': ['iran', 'iranian'],
    'idf insignia': ['israel', 'israeli'],
    'star of david': ['israel', 'israeli'],
    'iaf roundel': ['israel', 'israeli'],
    'red star': ['russia', 'russian', 'china', 'chinese'],
    'cyrillic': ['russia', 'russian'],
    'pla insignia': ['china', 'chinese'],
    'trident insignia': ['ukraine', 'ukrainian'],
    'blue and yellow': ['ukraine', 'ukrainian'],
    'stars and stripes': ['usa', 'american', 'united states'],
    'us roundel': ['usa', 'american'],
    'nato emblem': ['nato'],
    'crossed swords roundel': ['saudi_arabia', 'saudi'],
    'crescent and star': ['turkey', 'turkish', 'pakistan', 'pakistani'],
    'al-qassam': ['hamas', 'palestinian'],
    'ansar allah': ['houthi', 'yemeni'],
    # Equipment → countries that use them
    'iron dome': ['israel', 'israeli'],
    'f-35i adir': ['israel', 'israeli'],
    'f-35': ['usa', 'american'],
    'f-22 raptor': ['usa', 'american'],
    'su-35': ['russia', 'russian'],
    'j-20': ['china', 'chinese'],
    'bayraktar tb2': ['turkey', 'turkish'],
    'jf-17': ['pakistan', 'pakistani'],
    # Settings → implied actors/countries
    'strait of hormuz': ['iran', 'irgc', 'hormuz'],
    'persian gulf': ['iran', 'gulf'],
    'red sea': ['houthi', 'red sea'],
    'naval blockade': ['blockade', 'naval'],
    # Common enhanced action phrases → script verbs
    'precision airstrike': ['strike', 'attack', 'bomb'],
    'missile launch': ['launch', 'missile'],
    'naval blockade formation': ['blockade', 'naval'],
    'tactical withdrawal': ['retreat', 'withdraw'],
    'troop mobilization': ['mobilize', 'troops', 'forces'],
    'high-level diplomatic': ['negotiate', 'diplomacy', 'talks'],
    'mass evacuation': ['evacuate', 'evacuation', 'flee'],
}


def _stem(word: str) -> str:
    """Minimal suffix-stripping stem: removes common English suffixes.
    No external dependencies — covers 80%+ of news vocabulary stems.
    """
    word = word.lower()
    # Order matters: strip longer suffixes first
    for suffix in ('ational', 'tional', 'ization', 'isation', 'ation', 'ness',
                   'ment', 'tion', 'ling', 'ian', 'ing', 'ies', 'ied', 'ers',
                   'ed', 'er', 'ly', 'al', 'ic', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def calculate_prompt_relevance(prompt: str, article_text: str) -> int:
    """
    Calculate how relevant the prompt is to the article/script content.
    Uses stem-matching + entity alias table for BM25-style scoring,
    which bridges the gap between enhanced prompt language and raw script words.
    
    Args:
        prompt: Generated image prompt
        article_text: Original article or script segment text
        
    Returns:
        Relevance score (0-100)
    """
    score = 0
    article_lower = article_text.lower()
    prompt_lower = prompt.lower()

    # ── Check for military equipment matches (high value: 20 pts) ──
    equipment_found = _extract_equipment_from_prompt(prompt)
    for equipment in equipment_found:
        if equipment.lower() in article_lower:
            score += 20
            break

    # ── Check for real locations (high value: 25 pts) ──
    all_locations = get_all_locations()
    for loc in all_locations:
        if loc.lower() in prompt_lower and loc.lower() in article_lower:
            score += 25
            break

    # ── Entity alias matching: enhanced descriptions → source keywords (15 pts) ──
    alias_hit = False
    for alias_phrase, source_keywords in _ENTITY_ALIAS_TABLE.items():
        if alias_phrase in prompt_lower:
            if any(kw in article_lower for kw in source_keywords):
                score += 15
                alias_hit = True
                break

    # ── Stem-based word overlap (replaces naive set intersection) ──
    # Stem both prompt and article, then check overlap
    stop = {'with', 'from', 'this', 'that', 'they', 'have', 'will', 'been',
            'their', 'over', 'into', 'more', 'than', 'also', 'while', 'when',
            'after', 'which', 'were', 'some', 'time', 'year', 'said', 'high'}
    prompt_stems = set(
        _stem(w) for w in re.findall(r'[a-z]+', prompt_lower)
        if len(w) > 4 and w not in stop
    )
    article_stems = set(
        _stem(w) for w in re.findall(r'[a-z]+', article_lower)
        if len(w) > 4 and w not in stop
    )
    stem_overlap = prompt_stems & article_stems
    if len(stem_overlap) >= 4:
        score += 20
    elif len(stem_overlap) >= 2:
        score += 12
    elif len(stem_overlap) >= 1:
        score += 6

    # ── Action verbs from both prompt and article (10 pts) ──
    action_verbs = [
        'striking', 'launching', 'banking', 'deploying', 'intercepting',
        'evading', 'conducting', 'executing', 'maneuvering', 'surging',
        'signing', 'queuing', 'collapsing', 'negotiating', 'protesting',
        'shipping', 'trading', 'rising', 'falling', 'advancing', 'retreating',
        'mobilizing', 'blockading', 'evacuating', 'escalating', 'targeting'
    ]
    prompt_has_action = any(verb in prompt_lower for verb in action_verbs)
    article_has_action = any(verb in article_lower for verb in action_verbs)
    if prompt_has_action and article_has_action:
        score += 10
    elif prompt_has_action:
        score += 5

    # ── Temporal context (low value: 5 pts) ──
    temporal_words = ['dawn', 'dusk', 'night', 'morning', 'afternoon', 'golden hour']
    if any(word in prompt_lower for word in temporal_words):
        score += 5

    # ── Penalize generic terms ──
    generic_terms = ['generic', 'abstract', 'futuristic', 'sci-fi', 'future']
    if any(term in prompt_lower for term in generic_terms):
        score -= 50

    return max(0, min(100, score))


def _check_specific_equipment(prompt: str) -> bool:
    """Check if prompt contains specific military equipment nomenclature or named units."""
    # Pattern for specific equipment — trailing word optional
    patterns = [
        r'F-\d+[A-Z]{0,2}',        # F-35, F-14, F-16I
        r'S-\d{3,4}',               # S-400, S-300
        r'MiG-\d+',                 # MiG-29
        r'Su-\d+',                  # Su-35, Su-57
        r'J-\d+',                   # J-20, J-16
        r'JF-\d+',                  # JF-17
        r'MQ-\d+',                  # MQ-9
        r'AH-\d+',                  # AH-64
        r'B-\d+',                   # B-52
        r'USS\s+\w+',               # USS Nimitz
        r'Type\s+\d+',              # Type 055
        r'DF-\d+',                  # DF-21
        r'HQ-\d+',                  # HQ-9
        r'M\d+[A-Z]?\d?',           # M1A2, M60T
    ]
    prompt_str = prompt
    for pattern in patterns:
        if re.search(pattern, prompt_str, re.IGNORECASE):
            return True

    # Named military organizations and systems (highly specific)
    named_units = [
        'IRGC', 'IDF', 'IAF', 'USAF', 'USN', 'USMC', 'PLAN', 'PLAAF',
        'VVS', 'PAF', 'RSAF', 'Iron Dome', "David's Sling", 'Patriot',
        'THAAD', 'Tomahawk', 'Kalibr', 'Iskander', 'Shahab', 'Bayraktar',
        'Qassam', 'Al-Qassam', 'Ansar Allah', 'Arleigh Burke', 'Nimitz',
        'Liaoning', 'Kuznetsov', 'Shahed',
    ]
    prompt_lower = prompt.lower()
    for unit in named_units:
        if unit.lower() in prompt_lower:
            return True

    # Check against equipment database
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        for equipment_key, equipment_info in equipment_dict.items():
            if isinstance(equipment_info, dict):
                full_name = equipment_info.get('full_name', equipment_key)
            else:
                full_name = equipment_info
            if full_name.lower() in prompt_lower:
                return True

    return False


def _check_real_location(prompt: str) -> bool:
    """Check if prompt contains real geographic location"""
    all_locations = get_all_locations()
    
    for loc in all_locations:
        if loc.lower() in prompt.lower():
            return True
    
    return False


def _check_action_verb(prompt: str) -> bool:
    """Check if prompt contains dynamic action verbs"""
    action_verbs = [
        'striking', 'launching', 'banking', 'deploying', 'intercepting',
        'evading', 'conducting', 'executing', 'maneuvering', 'dropping',
        'firing', 'targeting', 'advancing', 'securing', 'patrolling'
    ]
    
    return any(verb in prompt.lower() for verb in action_verbs)


def _check_no_generic_terms(prompt: str) -> bool:
    """Check that prompt doesn't contain generic/vague terms"""
    generic_terms = [
        'generic', 'abstract', 'futuristic', 'sci-fi', 'future',
        'unknown', 'unspecified', 'various', 'some', 'several'
    ]
    
    return not any(term in prompt.lower() for term in generic_terms)


def _check_style_suffix(prompt: str) -> bool:
    """Check if prompt contains required pixel art style suffix"""
    required_elements = ['isometric', 'pixel art']
    return all(elem in prompt.lower() for elem in required_elements)


def _check_token_weighting(prompt: str) -> bool:
    """Check if prompt uses token weighting for emphasis"""
    # Look for (keyword:weight) pattern
    weighting_pattern = r'\([^:]+:[\d.]+\)'
    return bool(re.search(weighting_pattern, prompt))


def _check_visual_grounding(prompt: str) -> bool:
    """Check if prompt contains visual grounding instructions"""
    grounding_indicators = [
        'formation', 'positioned', 'arrangement', 'layout',
        'perspective', 'composition', 'foreground', 'background'
    ]
    return any(indicator in prompt.lower() for indicator in grounding_indicators)


def _extract_equipment_from_prompt(prompt: str) -> List[str]:
    """Extract equipment mentions from prompt"""
    equipment_found = []
    
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        for equipment_key, equipment_info in equipment_dict.items():
            # Handle both old string format and new dict format
            if isinstance(equipment_info, dict):
                full_name = equipment_info.get('full_name', equipment_key)
            else:
                full_name = equipment_info
            
            if full_name.lower() in prompt.lower():
                equipment_found.append(full_name)
    
    return equipment_found


def _get_check_details(checks: Dict[str, bool]) -> List[str]:
    """Get human-readable details about failed checks"""
    details = []
    
    if not checks['has_specific_equipment']:
        details.append("Missing specific military equipment nomenclature")
    
    if not checks['has_real_location']:
        details.append("Missing real geographic location")
    
    if not checks['has_action_verb']:
        details.append("Missing dynamic action verb")
    
    if not checks['no_generic_terms']:
        details.append("Contains generic/vague terms")
    
    if not checks['has_style_suffix']:
        details.append("Missing pixel art style suffix")
    
    return details


def validate_article_prompt_correlation_with_geopolitical(
    prompt: str,
    article_text: str,
    visual_elements: Dict[str, Any],
    script_text: str = None
) -> Dict[str, Any]:
    """
    Comprehensive validation including geopolitical accuracy.
    
    Args:
        prompt: Generated prompt
        article_text: Original article text
        visual_elements: Extracted visual elements
        script_text: Original script text for context (optional)
        
    Returns:
        Enhanced validation results with geopolitical accuracy
    """
    # Standard validation
    standard_validation = validate_article_prompt_correlation(prompt, article_text, visual_elements)
    
    # Geopolitical validation
    geo_validator = GeopoliticalValidator()
    geopolitical_validation = geo_validator.validate_prompt_geopolitical_accuracy(prompt, script_text)
    
    # Combine results
    combined_validation = {
        'standard': standard_validation,
        'geopolitical': geopolitical_validation,
        'overall_pass': (
            standard_validation['overall_pass'] and 
            geopolitical_validation['passed']
        ),
        'combined_score': (
            (standard_validation['relevance_score'] + geopolitical_validation['accuracy_score']) / 2
        ),
        'total_issues': (
            len(standard_validation['recommendation'].split(', ') if standard_validation['recommendation'] != "ACCEPT" else []) +
            len(geopolitical_validation['issues'])
        ),
        'recommendations': {
            'standard': standard_validation['recommendation'],
            'geopolitical': geopolitical_validation['recommendations']
        }
    }
    
    # Final recommendation
    if combined_validation['overall_pass']:
        if combined_validation['combined_score'] >= 85:
            combined_validation['final_recommendation'] = "ACCEPT - High quality with geopolitical accuracy"
        else:
            combined_validation['final_recommendation'] = "ACCEPT - Good quality, acceptable accuracy"
    else:
        if not standard_validation['overall_pass']:
            combined_validation['final_recommendation'] = f"REGENERATE - Quality issues: {standard_validation['recommendation']}"
        elif not geopolitical_validation['passed']:
            combined_validation['final_recommendation'] = f"REGENERATE - Geopolitical issues: {'; '.join(geopolitical_validation['issues'][:2])}"
        else:
            combined_validation['final_recommendation'] = "REGENERATE - Multiple issues detected"
    
    return combined_validation


def _get_recommendation(
    quality: Dict[str, Any],
    relevance: int,
    correlation: Dict[str, bool]
) -> str:
    """Generate recommendation based on validation results"""
    if quality['passed'] and relevance >= 70:
        return "ACCEPT - High quality and relevance"
    
    if quality['passed'] and relevance >= 50:
        return "ACCEPT - Good quality, acceptable relevance"
    
    if not quality['passed']:
        failed_checks = quality['details']
        return f"REGENERATE - Quality issues: {', '.join(failed_checks)}"
    
    if relevance < 50:
        return "REGENERATE - Low relevance to article content"
    
    if sum(correlation.values()) < 2:
        return "REGENERATE - Poor correlation with extracted elements"
    
    return "REVIEW - Manual review recommended"
