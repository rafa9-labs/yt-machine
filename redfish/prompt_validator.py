"""
Prompt Quality Validator - Ensures generated prompts meet quality standards
Validates prompts for specificity, accuracy, and relevance
"""

import re
from typing import Dict, Any, List
from .military_equipment_db import get_all_locations, MILITARY_EQUIPMENT_DB


def validate_prompt_quality(prompt: str) -> Dict[str, Any]:
    """
    Validate prompt quality against multiple criteria
    
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
        'has_style_suffix': _check_style_suffix(prompt)
    }
    
    passed = sum(checks.values())
    total = len(checks)
    score = (passed / total) * 100
    
    return {
        'score': score,
        'checks': checks,
        'passed': passed >= 4,  # Must pass at least 4/5 checks
        'details': _get_check_details(checks)
    }


def calculate_prompt_relevance(prompt: str, article_text: str) -> int:
    """
    Calculate how relevant the prompt is to the article content
    
    Args:
        prompt: Generated image prompt
        article_text: Original article text
        
    Returns:
        Relevance score (0-100)
    """
    score = 0
    
    # Check for specific subjects from article (high value)
    # Extract key nouns from article for matching
    article_lower = article_text.lower()
    prompt_lower = prompt.lower()
    
    # Check for military equipment matches
    equipment_found = _extract_equipment_from_prompt(prompt)
    for equipment in equipment_found:
        if equipment.lower() in article_lower:
            score += 20
            break
    
    # Check for real locations (high value)
    all_locations = get_all_locations()
    for loc in all_locations:
        if loc.lower() in prompt_lower and loc.lower() in article_lower:
            score += 25
            break
    
    # Check for general subject-article word overlap (medium value)
    prompt_words = set(w for w in prompt_lower.split() if len(w) > 4)
    article_words = set(w for w in article_lower.split() if len(w) > 4)
    overlap = prompt_words & article_words
    if len(overlap) >= 3:
        score += 20
    elif len(overlap) >= 1:
        score += 10
    
    # Check for action verbs (medium value)
    action_verbs = ['striking', 'launching', 'banking', 'deploying', 'intercepting',
                    'evading', 'conducting', 'executing', 'maneuvering', 'surging',
                    'signing', 'queuing', 'collapsing', 'negotiating', 'protesting',
                    'shipping', 'trading', 'rising', 'falling']
    if any(verb in prompt_lower for verb in action_verbs):
        score += 15
    
    # Check for temporal context (low value)
    temporal_words = ['dawn', 'dusk', 'night', 'morning', 'afternoon', 'golden hour']
    if any(word in prompt_lower for word in temporal_words):
        score += 10
    
    # Penalize generic terms
    generic_terms = ['generic', 'abstract', 'futuristic', 'sci-fi', 'future']
    if any(term in prompt_lower for term in generic_terms):
        score -= 50
    
    # Ensure score is in valid range
    return max(0, min(100, score))


def _check_specific_equipment(prompt: str) -> bool:
    """Check if prompt contains specific military equipment nomenclature"""
    # Pattern for specific equipment
    patterns = [
        r'F-\d+[A-Z]?\s+\w+',      # F-35 Lightning II
        r'S-\d+\s+\w+',             # S-400 Triumf
        r'USS\s+\w+',               # USS Boxer
        r'M\d+A?\d?\s+\w+',         # M1A2 Abrams
        r'MQ-\d+\s+\w+',            # MQ-9 Reaper
        r'AH-\d+\s+\w+',            # AH-64 Apache
        r'Type\s+\d+',              # Type 055
    ]
    
    for pattern in patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True
    
    # Also check against equipment database
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        for full_name in equipment_dict.values():
            if full_name.lower() in prompt.lower():
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


def _extract_equipment_from_prompt(prompt: str) -> List[str]:
    """Extract equipment mentions from prompt"""
    equipment_found = []
    
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        for full_name in equipment_dict.values():
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


def validate_article_prompt_correlation(
    prompt: str,
    article_text: str,
    visual_elements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive validation of prompt against article and extracted elements
    
    Args:
        prompt: Generated prompt
        article_text: Original article
        visual_elements: Extracted visual elements
        
    Returns:
        Validation results with recommendations
    """
    quality = validate_prompt_quality(prompt)
    relevance = calculate_prompt_relevance(prompt, article_text)
    
    # Check correlation with extracted elements
    correlation_checks = {
        'equipment_match': False,
        'location_match': False,
        'action_match': False
    }
    
    # Check subject correlation (supports both old and new schema)
    subjects = visual_elements.get('primary_subjects', visual_elements.get('military_equipment', []))
    if subjects:
        for subj in subjects:
            if subj.lower() in prompt.lower():
                correlation_checks['equipment_match'] = True
                break
    
    # Check location/setting correlation (supports both old and new schema)
    settings = visual_elements.get('settings', visual_elements.get('locations', []))
    if settings:
        for loc in settings:
            if loc.lower() in prompt.lower():
                correlation_checks['location_match'] = True
                break
    
    # Check action correlation
    if visual_elements.get('actions'):
        for action in visual_elements['actions']:
            if action.lower() in prompt.lower():
                correlation_checks['action_match'] = True
                break
    
    # Overall assessment
    overall_pass = (
        quality['passed'] and
        relevance >= 50 and
        sum(correlation_checks.values()) >= 2
    )
    
    return {
        'quality': quality,
        'relevance_score': relevance,
        'correlation': correlation_checks,
        'overall_pass': overall_pass,
        'recommendation': _get_recommendation(quality, relevance, correlation_checks)
    }


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
