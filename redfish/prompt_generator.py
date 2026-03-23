"""
Visual Prompt Generator - Script-first prompt construction for image generation
Generates image prompts directly from script content with trending context boost
Phase 5.0: Script-to-image synchronization - images match what script says
"""

import random
from typing import Dict, Any, List, Optional
from .action_mapping import enhance_action, get_dynamic_action_phrase, SCENE_ACTION_MODIFIERS
from .military_equipment_db import normalize_equipment
from .historical_equipment_db import get_equipment_for_era, normalize_historical_equipment
from .script_parser import ScriptParser


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
    Generates image prompts directly from script content (script-first approach).
    Phase 5.0: Script-to-image synchronization with trending context boost.
    """
    
    def __init__(self, script: Dict[str, Any], trending_context: Dict[str, Any] = None,
                 news_analysis: Dict[str, Any] = None, visual_elements: Dict[str, Any] = None):
        """
        Args:
            script: Synthesized script dict with segments (PRIMARY SOURCE)
            trending_context: Trending words from TrendingAnalyzer (for boost)
            news_analysis: Analysis from LLM (fallback only)
            visual_elements: Extracted visual elements (fallback only)
        """
        self.script = script or {}
        self.trending_context = trending_context or {}
        self.news_analysis = news_analysis or {}
        self.visual_elements = visual_elements or {}
        self.style_suffix = "true 16-bit pixel art, retro SNES style, isometric perspective, hard pixel edges, limited color palette with dark navy blues and amber accents, detailed proportions, flat colors, NO blur, NO text"
        
        # Script parser with visual concept extraction
        self._parser = ScriptParser()
        
        # Extract visual concepts from each script segment
        self._visual_concepts: Dict[str, Dict] = {}
        if self.script:
            for segment_name in self._get_segment_names():
                segment_text = self.script.get(segment_name, '')
                if segment_text:
                    self._visual_concepts[segment_name] = self._parser.extract_visual_concepts(
                        segment_text, 
                        self.trending_context
                    )
    
    def generate_scene_prompt(self, scene_type: str, fallback_prompt: str = None) -> str:
        """
        Generate prompt for a specific scene FROM SCRIPT CONTENT.
        Phase 5.0: Script-first approach - image matches what script says.
        
        Args:
            scene_type: One of hook, historical_1, historical_2, modern_pivot, consequence, future_outlook
            fallback_prompt: Optional fallback if script extraction fails
            
        Returns:
            Complete image generation prompt
        """
        # PRIMARY: Build from script visual concepts
        if scene_type in self._visual_concepts:
            concepts = self._visual_concepts[scene_type]
            prompt = self._build_prompt_from_concepts(concepts, scene_type)
            if prompt and len(prompt) > 20:
                return f"{prompt}, {self.style_suffix}"
        
        # FALLBACK 1: Try old script parser method
        segment_text = self.script.get(scene_type, '')
        if segment_text:
            parsed = self._parser.parse_segment(segment_text, scene_type)
            parser_prompt = self._parser.build_action_prompt(parsed, self.style_suffix)
            if parser_prompt and len(parser_prompt) > 30:
                return parser_prompt
        
        # FALLBACK 2: LLM-generated visual scene description
        llm_scene_desc = self._get_script_visual_scene(scene_type)
        if llm_scene_desc and len(llm_scene_desc) > 20:
            template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['hook'])
            base_prompt = f"{template['perspective']}: {llm_scene_desc}"
            return f"{base_prompt}, {self.style_suffix}"
        
        # FALLBACK 3: Build from extracted visual elements (article-based)
        subject = self._get_subject_for_scene(scene_type)
        setting = self._get_setting_for_scene(scene_type)
        action = self._get_action_for_scene(scene_type)
        context = self._get_temporal_context()
        
        template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['hook'])
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
        
        # Final fallback
        if not base_prompt or len(base_prompt) < 10:
            topic = self.news_analysis.get('topic', 'geopolitical event')
            base_prompt = f"{template['perspective']}: {topic}, {template['focus']}"
        
        return f"{base_prompt}, {self.style_suffix}"
    
    def generate_all_scenes(self) -> Dict[str, str]:
        """
        Generate prompts for all scenes FROM SCRIPT.
        Phase 5.0: Each prompt matches its script segment content.
        
        Returns:
            Dictionary with scene prompts
        """
        scene_names = self._get_segment_names()
        
        # Get LLM fallback prompts if available
        pixel_art_prompts = self.news_analysis.get('pixel_art_prompts', [])
        fallbacks = {}
        for i, name in enumerate(scene_names):
            fallbacks[name] = pixel_art_prompts[i] if i < len(pixel_art_prompts) else None
        
        prompts = {}
        for name in scene_names:
            prompt = self.generate_scene_prompt(name, fallbacks[name])
            prompts[name] = prompt
            
            # Log visual concept info for debugging
            if name in self._visual_concepts:
                concepts = self._visual_concepts[name]
                print(f"  [{name}] Type: {concepts.get('visual_type', 'unknown')}, "
                      f"Boost: {concepts.get('trending_boost', 0.0):.2f}")
        
        return prompts
    
    def _get_segment_names(self) -> List[str]:
        """Get segment names based on script structure."""
        if self.script and 'historical_1' in self.script:
            return ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
        else:
            return ['hook', 'context', 'escalation', 'consequence', 'twist']
    
    def _build_prompt_from_concepts(self, concepts: Dict[str, Any], scene_type: str) -> str:
        """
        Build image prompt from extracted visual concepts.
        This is the PRIMARY method for script-to-image synchronization.
        """
        template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['hook'])
        
        # Get concept elements
        primary_concept = concepts.get('primary_concept', '')
        subjects = concepts.get('subjects', [])
        setting = concepts.get('setting', '')
        action = concepts.get('action', '')
        mood = concepts.get('mood', 'tense')
        visual_type = concepts.get('visual_type', 'general')
        emphasis = concepts.get('emphasis', '')
        trending_boost = concepts.get('trending_boost', 0.0)
        
        # Build prompt parts
        prompt_parts = []
        
        # Start with perspective from template
        prompt_parts.append(template['perspective'])
        
        # Add primary concept or build from subjects/action
        if primary_concept:
            prompt_parts.append(primary_concept)
        elif subjects:
            subject_str = subjects[0]
            if action and action != 'in dramatic confrontation':
                prompt_parts.append(f"{subject_str} {action}")
            else:
                prompt_parts.append(subject_str)
            
            if setting:
                prompt_parts.append(f"at {setting}")
        
        # Add emphasis based on visual type
        if emphasis:
            prompt_parts.append(emphasis)
        
        # Add mood/atmosphere
        mood_descriptors = {
            'tense': 'high tension atmosphere',
            'chaotic': 'chaotic energy',
            'economic': 'economic impact focus',
            'hopeful': 'hopeful tone',
            'historical': 'historical accuracy'
        }
        if mood in mood_descriptors:
            prompt_parts.append(mood_descriptors[mood])
        
        # Boost trending elements
        if trending_boost > 0.5:
            prompt_parts.append('emphasize key trending elements')
        
        return ', '.join(prompt_parts)
    
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
        # Try to use script concepts first
        if scene_type in self._visual_concepts:
            concepts = self._visual_concepts[scene_type]
            subjects = concepts.get('subjects', [])
            setting = concepts.get('setting', '')
            action = concepts.get('action', '')
            
            subject = subjects[0] if subjects else "strategic forces"
            setting = setting or "strategic location"
            action = action or "in position"
        else:
            # Fallback to article-based extraction
            subject = self._get_subject_for_scene(scene_type) or "strategic forces"
            setting = self._get_setting_for_scene(scene_type) or "strategic location"
            action = self._get_action_for_scene(scene_type)
        
        topic = self.news_analysis.get('topic', '')
        strict_prompt = f"{subject} {action} at {setting}, {topic}, {self.style_suffix}"
        return strict_prompt
