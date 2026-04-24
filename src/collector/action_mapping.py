"""
Action Verb Mapping for Dynamic Visual Descriptions
Maps military action verbs to visual descriptions for image generation
"""

ACTION_VISUAL_MAP = {
    # Air operations
    'strike': 'dropping precision-guided munitions on',
    'strikes': 'dropping precision-guided munitions on',
    'striking': 'dropping precision-guided munitions on',
    'bomb': 'conducting bombing runs against',
    'bombs': 'conducting bombing runs against',
    'bombing': 'conducting bombing runs against',
    'attack': 'executing coordinated attack on',
    'attacks': 'executing coordinated attack on',
    'attacking': 'executing coordinated attack on',
    'intercept': 'intercepting incoming projectiles over',
    'intercepts': 'intercepting incoming projectiles over',
    'intercepting': 'intercepting incoming projectiles over',
    'patrol': 'conducting patrol operations near',
    'patrols': 'conducting patrol operations near',
    'patrolling': 'conducting patrol operations near',
    'fly': 'banking sharply over',
    'flies': 'banking sharply over',
    'flying': 'banking sharply over',
    'engage': 'engaging hostile forces at',
    'engages': 'engaging hostile forces at',
    'engaging': 'engaging hostile forces at',
    
    # Naval operations
    'launch': 'launching missiles from',
    'launches': 'launching missiles from',
    'launching': 'launching missiles from',
    'deploy': 'deploying forces to',
    'deploys': 'deploying forces to',
    'deploying': 'deploying forces to',
    'sail': 'conducting freedom of navigation operations near',
    'sails': 'conducting freedom of navigation operations near',
    'sailing': 'conducting freedom of navigation operations near',
    'blockade': 'establishing naval blockade of',
    'blockades': 'establishing naval blockade of',
    'blockading': 'establishing naval blockade of',
    'escort': 'escorting commercial vessels through',
    'escorts': 'escorting commercial vessels through',
    'escorting': 'escorting commercial vessels through',
    
    # Ground operations
    'advance': 'advancing tactical positions toward',
    'advances': 'advancing tactical positions toward',
    'advancing': 'advancing tactical positions toward',
    'secure': 'securing strategic positions at',
    'secures': 'securing strategic positions at',
    'securing': 'securing strategic positions at',
    'occupy': 'occupying territory in',
    'occupies': 'occupying territory in',
    'occupying': 'occupying territory in',
    'defend': 'defending positions against assault on',
    'defends': 'defending positions against assault on',
    'defending': 'defending positions against assault on',
    'retreat': 'conducting tactical withdrawal from',
    'retreats': 'conducting tactical withdrawal from',
    'retreating': 'conducting tactical withdrawal from',
    
    # Missile/artillery operations
    'fire': 'firing artillery barrage at',
    'fires': 'firing artillery barrage at',
    'firing': 'firing artillery barrage at',
    'target': 'targeting military installations in',
    'targets': 'targeting military installations in',
    'targeting': 'targeting military installations in',
    'hit': 'striking targets in',
    'hits': 'striking targets in',
    'hitting': 'striking targets in',
    
    # Surveillance/reconnaissance
    'monitor': 'conducting surveillance operations over',
    'monitors': 'conducting surveillance operations over',
    'monitoring': 'conducting surveillance operations over',
    'surveil': 'surveilling military movements near',
    'surveils': 'surveilling military movements near',
    'surveilling': 'surveilling military movements near',
    'track': 'tracking hostile forces in',
    'tracks': 'tracking hostile forces in',
    'tracking': 'tracking hostile forces in',
    
    # Defensive operations
    'evade': 'evading incoming fire over',
    'evades': 'evading incoming fire over',
    'evading': 'evading incoming fire over',
    'counter': 'countering offensive operations at',
    'counters': 'countering offensive operations at',
    'countering': 'countering offensive operations at',
    'respond': 'responding to provocation in',
    'responds': 'responding to provocation in',
    'responding': 'responding to provocation in'
}

# Scene-specific action modifiers
SCENE_ACTION_MODIFIERS = {
    'hook': [
        'banking sharply while',
        'diving aggressively toward',
        'launching rapidly from',
        'streaking toward',
        'maneuvering evasively over'
    ],
    'body': [
        'conducting operations at',
        'maintaining position near',
        'executing maneuvers in',
        'deploying assets to',
        'establishing presence at'
    ],
    'twist': [
        'revealing aftermath of',
        'exposing consequences at',
        'demonstrating capability over',
        'showing impact on',
        'unveiling results in'
    ]
}

def enhance_action(verb: str, target: str, scene_type: str = None) -> str:
    """
    Convert action verb to dynamic visual description
    
    Args:
        verb: Action verb from article (e.g., 'strike', 'launch')
        target: Target location or object
        scene_type: Optional scene type (hook, body, twist) for modifiers
        
    Returns:
        Enhanced action description for image prompt
    """
    verb_lower = verb.lower().strip()
    
    # Get base visual action
    visual_action = ACTION_VISUAL_MAP.get(verb_lower, verb)
    
    # Add scene-specific modifier if provided
    if scene_type and scene_type in SCENE_ACTION_MODIFIERS:
        import random
        modifier = random.choice(SCENE_ACTION_MODIFIERS[scene_type])
        return f"{modifier} {visual_action} {target}"
    
    return f"{visual_action} {target}"

def extract_action_verbs(text: str) -> list:
    """
    Extract action verbs from text
    
    Args:
        text: Article text or description
        
    Returns:
        List of action verbs found in text
    """
    found_verbs = []
    text_lower = text.lower()
    
    for verb in ACTION_VISUAL_MAP.keys():
        if verb in text_lower:
            found_verbs.append(verb)
    
    return found_verbs

def get_dynamic_action_phrase(article_text: str, scene_type: str = 'body') -> str:
    """
    Generate dynamic action phrase from article text
    
    Args:
        article_text: Full article text
        scene_type: Type of scene (hook, body, twist)
        
    Returns:
        Dynamic action phrase for image generation
    """
    verbs = extract_action_verbs(article_text)
    
    if not verbs:
        # Default actions by scene type
        defaults = {
            'hook': 'banking sharply while striking',
            'body': 'conducting operations at',
            'twist': 'revealing aftermath of'
        }
        return defaults.get(scene_type, 'operating in')
    
    # Use first found verb
    primary_verb = verbs[0]
    return ACTION_VISUAL_MAP.get(primary_verb, primary_verb)
