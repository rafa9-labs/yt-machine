"""
Visual Prompt Generator - Narrative-driven prompt construction for image generation
Builds 5 scene prompts from extracted visual elements + script segments
"""

import random
from typing import Dict, Any, List, Optional
from .action_mapping import enhance_action, get_dynamic_action_phrase, SCENE_ACTION_MODIFIERS
from .military_equipment_db import normalize_equipment
from .historical_equipment_db import get_equipment_for_era, normalize_historical_equipment


# Scene composition templates — 6 scenes matching script segments with historical anchoring
SCENE_TEMPLATES = {
    'hook': {
        'perspective': 'dramatic close-up or wide aerial view',
        'focus': 'attention-grabbing visual matching the opening fact',
        'style_notes': 'high tension, dynamic motion, visceral impact',
        'era': '2020s'
    },
    'historical_1': {
        'perspective': 'wide historical view or tactical angle',
        'focus': 'first historical parallel — era-accurate equipment and settings',
        'style_notes': 'authoritative, historical accuracy, pattern recognition',
        'era': '1990s'  # Default, will be overridden by script
    },
    'historical_2': {
        'perspective': 'strategic historical overview',
        'focus': 'second historical parallel — different era or angle',
        'style_notes': 'depth, multi-layered context, expertise',
        'era': '1980s'  # Default, will be overridden by script
    },
    'modern_pivot': {
        'perspective': 'dynamic return to present',
        'focus': 'current 2026 situation — modern equipment and dynamics',
        'style_notes': 'contrast with history, what changed, new players',
        'era': '2020s'
    },
    'consequence': {
        'perspective': 'ground-level human perspective',
        'focus': 'human impact — civilian life, prices, shortages, queues, empty shelves',
        'style_notes': 'empathy, real-world consequences, relatable imagery',
        'era': '2020s'
    },
    'future_outlook': {
        'perspective': 'revealing wide shot or strategic map view',
        'focus': 'strategic implication — where this leads, geopolitical positioning',
        'style_notes': 'forward-looking, strategic, lingering question',
        'era': '2020s'
    }
}


class VisualPromptGenerator:
    """
    Generates 5 narrative-driven image prompts from visual elements and script segments.
    Each prompt is co-derived from the script to ensure visual-narrative coherence.
    """
    
    def __init__(self, news_analysis: Dict[str, Any], visual_elements: Dict[str, Any],
                 script: Optional[Dict[str, Any]] = None):
        """
        Args:
            news_analysis: Analysis from LLM (topic, angle, etc.)
            visual_elements: Extracted visual elements (primary_subjects, settings, actions)
            script: Optional synthesized script dict with hook/context/escalation/consequence/twist
        """
        self.news_analysis = news_analysis
        self.visual_elements = visual_elements
        self.script = script or {}
        self.style_suffix = "true 16-bit pixel art, retro SNES style, isometric perspective, hard pixel edges, limited color palette, detailed proportions, flat colors, NO blur"
    
    def generate_scene_prompt(self, scene_type: str, fallback_prompt: str = None) -> str:
        """
        Generate dynamic prompt for a specific scene.
        
        Args:
            scene_type: One of hook, context, escalation, consequence, twist
            fallback_prompt: Optional fallback if extraction fails
            
        Returns:
            Complete image generation prompt
        """
        # Fallback to 'hook' if scene_type not found (works for both 5 and 6 segment structures)
        template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['hook'])
        
        # Try to use visual_scenes from script first (LLM-generated descriptions)
        llm_scene_desc = self._get_script_visual_scene(scene_type)
        
        if llm_scene_desc and len(llm_scene_desc) > 20:
            # LLM provided a visual scene description — use it as the base
            base_prompt = f"{template['perspective']}: {llm_scene_desc}"
        else:
            # Build from extracted visual elements
            subject = self._get_subject_for_scene(scene_type)
            setting = self._get_setting_for_scene(scene_type)
            action = self._get_action_for_scene(scene_type)
            context = self._get_temporal_context()
            
            prompt_parts = []
            if subject:
                prompt_parts.append(f"{template['perspective']}: {subject}")
            if action:
                prompt_parts.append(action)
            if setting:
                prompt_parts.append(f"at {setting}")
            if context:
                prompt_parts.append(context)
            
            if len(prompt_parts) < 2 and fallback_prompt:
                base_prompt = fallback_prompt
            else:
                base_prompt = " ".join(filter(None, prompt_parts))
        
        # Ensure we have something
        if not base_prompt or len(base_prompt) < 10:
            topic = self.news_analysis.get('topic', 'geopolitical event')
            base_prompt = f"{template['perspective']}: {topic}, {template['focus']}"
        
        return f"{base_prompt}, {self.style_suffix}"
    
    def generate_all_scenes(self) -> Dict[str, str]:
        """
        Generate prompts for all 6 scenes (or 5 for backward compatibility).
        
        Returns:
            Dictionary with scene prompts
        """
        pixel_art_prompts = self.news_analysis.get('pixel_art_prompts', [])
        
        # Check if script has 6-segment structure (historical anchoring)
        if self.script and 'historical_1' in self.script:
            scene_names = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
        else:
            # Fallback to 5-segment structure
            scene_names = ['hook', 'context', 'escalation', 'consequence', 'twist']
        
        fallbacks = {}
        for i, name in enumerate(scene_names):
            fallbacks[name] = pixel_art_prompts[i] if i < len(pixel_art_prompts) else None
        
        return {
            name: self.generate_scene_prompt(name, fallbacks[name])
            for name in scene_names
        }
    
    def _get_script_visual_scene(self, scene_type: str) -> str:
        """Get the LLM-generated visual_scene description for this segment."""
        visual_scenes = self.script.get('visual_scenes', [])
        for scene in visual_scenes:
            if isinstance(scene, dict) and scene.get('segment') == scene_type:
                return scene.get('description', '')
        return ''
    
    def _get_subject_for_scene(self, scene_type: str) -> str:
        """Get the most appropriate visual subject for this scene type."""
        subjects = self.visual_elements.get('primary_subjects', [])
        
        if not subjects:
            return ''
        
        # Distribute subjects across scenes
        scene_index = {
            'hook': 0,
            'context': 1,
            'escalation': 2,
            'consequence': 3,
            'twist': 4
        }
        idx = scene_index.get(scene_type, 0)
        
        if idx < len(subjects):
            return subjects[idx]
        
        # Cycle through available subjects
        return subjects[idx % len(subjects)]
    
    def _get_setting_for_scene(self, scene_type: str) -> str:
        """Get the most appropriate setting/location for this scene type."""
        settings = self.visual_elements.get('settings', [])
        
        if not settings:
            # Fallback: extract from topic
            topic = self.news_analysis.get('topic', '')
            location_keywords = [
                'Strait of Hormuz', 'Persian Gulf', 'Tehran', 'Tel Aviv',
                'Gaza', 'Ukraine', 'Taiwan Strait', 'South China Sea',
                'Black Sea', 'Red Sea', 'Syria', 'Iraq', 'Washington',
                'Wall Street', 'Brussels', 'Beijing'
            ]
            for loc in location_keywords:
                if loc.lower() in topic.lower():
                    return loc
            return ''
        
        scene_index = {
            'hook': 0,
            'context': 0,
            'escalation': 1 if len(settings) > 1 else 0,
            'consequence': min(2, len(settings) - 1),
            'twist': min(3, len(settings) - 1)
        }
        idx = scene_index.get(scene_type, 0)
        return settings[idx]
    
    def _get_action_for_scene(self, scene_type: str) -> str:
        """Get a dynamic action phrase appropriate for the scene type."""
        actions = self.visual_elements.get('actions', [])
        
        if actions:
            # Pick an action, cycling through available ones
            scene_index = {'hook': 0, 'context': 1, 'escalation': 2, 'consequence': 3, 'twist': 4}
            idx = scene_index.get(scene_type, 0) % len(actions)
            base_action = actions[idx]
            return base_action
        
        # Fallback: scene-type-specific defaults
        defaults = {
            'hook': 'in dramatic motion',
            'context': 'positioned strategically',
            'escalation': 'escalating rapidly',
            'consequence': 'affecting civilian life',
            'twist': 'revealing hidden connections'
        }
        return defaults.get(scene_type, '')
    
    def _get_temporal_context(self) -> str:
        """Extract temporal/weather context."""
        context = self.visual_elements.get('temporal_context', '')
        
        if context:
            return f"at {context}"
        
        # Fallback: use mood to suggest atmosphere
        mood = self.visual_elements.get('mood', 'tense')
        
        mood_lighting = {
            'tense': 'at dusk with dramatic orange sky',
            'chaotic': 'under stormy skies',
            'urgent': 'at dawn',
            'hopeful': 'at golden hour',
            'calm': 'under clear blue sky'
        }
        return mood_lighting.get(mood, '')
    
    def regenerate_strict(self, scene_type: str) -> str:
        """
        Regenerate prompt with stricter constraints.
        Used when relevance score is too low.
        
        Args:
            scene_type: Type of scene
            
        Returns:
            More constrained prompt
        """
        subject = self._get_subject_for_scene(scene_type) or "strategic forces"
        setting = self._get_setting_for_scene(scene_type) or "strategic location"
        action = self._get_action_for_scene(scene_type)
        topic = self.news_analysis.get('topic', '')
        
        strict_prompt = f"{subject} {action} at {setting}, {topic}, {self.style_suffix}"
        return strict_prompt
