"""
Visual Prompt Generator - Script-first prompt construction for image generation
Generates image prompts directly from script content with trending context boost
Phase 5.0: Script-to-image synchronization - images match what script says
Phase 6.0: Flux best-practice prompt hierarchy + scene composition from config
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from .action_mapping import enhance_action, get_dynamic_action_phrase, SCENE_ACTION_MODIFIERS
from .military_equipment_db import normalize_equipment
from .historical_equipment_db import get_equipment_for_era, normalize_historical_equipment
from .script_parser import ScriptParser
from .geopolitical_accuracy import get_country_visual_spec, validate_country_equipment_combination

# Load image style config — single source of truth (shared with pixel_art_tool.py)
_STYLE_CONFIG_PATH = Path(__file__).parent.parent / "config" / "image_style.json"
with open(_STYLE_CONFIG_PATH, 'r', encoding='utf-8') as _f:
    _IMAGE_STYLE_CONFIG = json.load(_f)

_SCENE_COMPOSITION = _IMAGE_STYLE_CONFIG.get('scene_composition', {})


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
        self.style_suffix = _IMAGE_STYLE_CONFIG.get('style_suffix',
            'Retro Pixel, (true 16-bit pixel art:1.5), (retro SNES style:1.3), isometric perspective, '
            '(hard pixel edges:1.2), limited color palette, detailed proportions, flat colors, dramatic lighting')
        
        # Script parser with visual concept extraction
        self._parser = ScriptParser()
        
        # Extract visual concepts from each script segment
        self._visual_concepts: Dict[str, Dict] = {}
        if self.script:
            for segment_name in self._get_segment_names():
                segment_data = self.script.get(segment_name, '')
                # Handle string, dict with 'narration'/'text'/'content', or other formats
                if isinstance(segment_data, str):
                    segment_text = segment_data
                elif isinstance(segment_data, dict):
                    segment_text = (segment_data.get('narration')
                                    or segment_data.get('text')
                                    or segment_data.get('content')
                                    or str(segment_data))
                else:
                    segment_text = str(segment_data) if segment_data else ''
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
        Build image prompt from extracted visual concepts using Flux best-practice hierarchy.
        Flux weights earlier tokens more heavily, so structure matters.
        
        Hierarchy (most important first):
          1. SUBJECT — what the image shows (front-loaded for max Flux weight)
          2. ACTION — what's happening (dynamic verbs)
          3. ENVIRONMENT — where (setting, geography)
          4. COMPOSITION — camera angle, framing from scene config (drives diversity)
          5. LIGHTING — locked per scene type from config (drives consistency)
          6. MOOD — atmosphere cues
        
        Style suffix and color palette are appended by pixel_art_tool.py, NOT here.
        """
        template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['hook'])
        scene_comp = _SCENE_COMPOSITION.get(scene_type, {})
        
        # Get concept elements
        subjects = concepts.get('subjects', [])
        setting = concepts.get('setting', '')
        action = concepts.get('action', '')
        mood = concepts.get('mood', 'tense')
        visual_type = concepts.get('visual_type', 'general')
        emphasis = concepts.get('emphasis', '')
        trending_boost = concepts.get('trending_boost', 0.0)
        
        # Geopolitical accuracy elements
        countries = concepts.get('countries', [])
        equipment = concepts.get('equipment', [])
        required_elements = concepts.get('required_elements', [])
        
        prompt_parts = []
        
        # ── 1. SUBJECT (front-loaded — Flux gives these tokens the most weight) ──
        if subjects:
            primary_subject = subjects[0]
            # Weight critical subjects more heavily
            if visual_type == 'military' and any(kw in primary_subject.lower() for kw in ['missile', 'tank', 'aircraft', 'warship']):
                prompt_parts.append(f"({primary_subject}:1.4)")
            elif visual_type == 'economic' and any(kw in primary_subject.lower() for kw in ['oil', 'prices', 'market']):
                prompt_parts.append(f"({primary_subject}:1.3)")
            else:
                prompt_parts.append(f"({primary_subject}:1.2)")
            # Inject top 1 trending term at position 2 — directly after primary subject
            # Flux weights position 2 tokens highly, making these visually dominant
            # LIMIT to 1 term per scene to avoid polluting historical/non-aligned scenes
            top_trending_terms = concepts.get('top_trending_terms', [])
            for term in top_trending_terms[:1]:  # Only the TOP trending term
                if term.lower() not in ', '.join(prompt_parts).lower():
                    prompt_parts.append(f"({term}:1.3)")
            # Add secondary subjects at lower weight
            for subj in subjects[1:3]:
                prompt_parts.append(subj)
        
        # Named entities (model numbers, named systems) — high specificity, front-loaded
        named_entities = concepts.get('named_entities', [])
        for entity in named_entities[:2]:
            entity_lower = entity.lower()
            if entity_lower not in ', '.join(prompt_parts).lower():
                prompt_parts.append(f"({entity}:1.3)")

        # Specificity modifiers (numbers, quantities) — anchor the visual concretely
        specificity_modifiers = concepts.get('specificity_modifiers', [])
        for modifier in specificity_modifiers[:2]:
            if modifier.lower() not in ', '.join(prompt_parts).lower():
                prompt_parts.append(modifier)

        # Historical scenes: use ERA-based injections instead of country-specific
        # Country injections (e.g. "Persian script", "IRGC") pollute historical scenes
        is_historical = scene_type in ('historical_1', 'historical_2')

        if is_historical:
            # Era-based visual terms — give richness without fighting the pixel art style
            era = template.get('era', '1990s')
            era_visuals = {
                '1980s': 'retro technology, CRT displays, analog instruments, cold war aesthetic',
                '1990s': 'early digital era, vintage military hardware, desert camouflage patterns',
                '2000s': 'early 2000s technology, transitional military equipment',
                '2010s': 'modern digital displays, contemporary military hardware',
            }
            era_desc = era_visuals.get(era, 'historical era-accurate equipment and settings')
            prompt_parts.append(f"({era_desc}:1.2)")
        else:
            # Country-specific equipment (critical for geopolitical accuracy — modern scenes only)
            for equip in equipment:
                for country in countries:
                    from .military_equipment_db import get_country_specific_variant, get_equipment_markings
                    if validate_country_equipment_combination(country, equip):
                        variant = get_country_specific_variant(equip, country)
                        prompt_parts.append(f"({variant}:1.3)")
                        markings = get_equipment_markings(equip, country)
                        if markings:
                            prompt_parts.append(f"({markings}:1.2)")

            # Country flag colors (modern scenes only)
            if countries:
                for country in countries:
                    country_spec = get_country_visual_spec(country)
                    if country_spec and 'flag_colors' in country_spec:
                        prompt_parts.append(f"({country_spec['flag_colors']}:1.2)")

        # Required geopolitical visual elements (all scenes)
        for element in required_elements:
            prompt_parts.append(f"({element}:1.3)")
        
        # ── 2. ACTION (dynamic verbs — what's happening) ──
        if action and action != 'in dramatic confrontation':
            action_enhanced = self._enhance_action_with_visuals(action, visual_type)
            prompt_parts.append(action_enhanced)
        
        # ── 3. ENVIRONMENT (setting, geography) ──
        if setting:
            setting_enhanced = self._enhance_setting_with_context(setting, visual_type)
            prompt_parts.append(f"at {setting_enhanced}")
        
        # ── 4. COMPOSITION (from scene config — drives DIVERSITY between segments) ──
        if scene_comp:
            camera = scene_comp.get('camera', '')
            framing = scene_comp.get('framing', '')
            spatial = scene_comp.get('spatial', '')
            if camera:
                prompt_parts.append(camera)
            if framing:
                prompt_parts.append(framing)
            if spatial:
                prompt_parts.append(spatial)
        else:
            # Fallback to old composition method
            composition = self._get_composition_instructions(visual_type, scene_type)
            if composition:
                prompt_parts.append(composition)
        
        # ── 5. LIGHTING (from scene config — locked per scene type for consistency) ──
        if scene_comp and scene_comp.get('lighting'):
            prompt_parts.append(scene_comp['lighting'])
        
        # ── Script anchors: raw keywords preserved verbatim for relevance scoring ──
        # These ensure stem-matching hits even when visual language is heavily enhanced.
        script_anchors = concepts.get('script_anchors', [])
        for anchor in script_anchors[:3]:
            if anchor.lower() not in ', '.join(prompt_parts).lower():
                prompt_parts.append(anchor)
        
        # ── 6. MOOD (atmosphere — lower priority, Flux de-weights late tokens) ──
        mood_descriptors = {
            'tense': '(high tension atmosphere:1.1)',
            'chaotic': '(chaotic energy:1.1)',
            'economic': '(economic impact focus:1.1)',
            'hopeful': '(hopeful tone:1.1)',
            'historical': '(historical accuracy:1.1)'
        }
        if mood in mood_descriptors:
            prompt_parts.append(mood_descriptors[mood])
        
        if emphasis:
            prompt_parts.append(f"({emphasis}:1.1)")
        
        # Trending boost: top terms already injected at position 2 above;
        # only add generic emphasis if boost is very high but no terms were injected
        if trending_boost > 0.7 and not concepts.get('top_trending_terms'):
            prompt_parts.append("(trending visual emphasis:1.2)")
        
        # Visual grounding (spatial relationships)
        grounding = self._parser.add_visual_grounding(concepts, scene_type)
        if grounding:
            prompt_parts.append(grounding)
        
        return ', '.join(prompt_parts)
    
    def _enhance_action_with_visuals(self, action: str, visual_type: str) -> str:
        """Enhance action with visual type-specific details"""
        action_enhancements = {
            'military': {
                'launch': 'missile launch with dramatic exhaust trail',
                'strike': 'precision airstrike with explosion plume',
                'deploy': 'military deployment in tactical formation',
                'blockade': 'naval blockade with warship line',
                'patrol': 'naval patrol cutting through waves'
            },
            'economic': {
                'surge': 'price surge with market indicators',
                'collapse': 'market collapse with falling screens',
                'trading': 'active trading floor with data screens'
            },
            'diplomatic': {
                'negotiate': 'high-level diplomatic negotiation',
                'sign': 'formal treaty signing ceremony',
                'meet': 'official summit meeting'
            },
            'human_impact': {
                'protest': 'mass civilian protest demonstration',
                'evacuate': 'emergency evacuation with crowds',
                'queue': 'civilian queue formation'
            }
        }
        
        type_enhancements = action_enhancements.get(visual_type, {})
        return type_enhancements.get(action, action)
    
    def _enhance_setting_with_context(self, setting: str, visual_type: str) -> str:
        """Enhance setting with visual type-specific context"""
        setting_enhancements = {
            'military': {
                'strait of hormuz': 'Strait of Hormuz with naval vessels and oil tankers',
                'persian gulf': 'Persian Gulf with military activity',
                'border': 'border region with fortified positions'
            },
            'economic': {
                'wall street': 'Wall Street trading floor with stock tickers',
                'market': 'financial market with price displays',
                'gas station': 'gas station with price board and queues'
            },
            'diplomatic': {
                'washington': 'Washington DC with government buildings',
                'summit': 'summit venue with flags and officials',
                'embassy': 'embassy with diplomatic insignia'
            },
            'human_impact': {
                'city': 'city with civilian activity',
                'residential': 'residential area with everyday life',
                'street': 'street level with people'
            }
        }
        
        type_enhancements = setting_enhancements.get(visual_type, {})
        return type_enhancements.get(setting.lower(), setting)
    
    def _get_composition_instructions(self, visual_type: str, scene_type: str) -> str:
        """Get composition instructions based on visual type and scene"""
        base_compositions = {
            'military': 'isometric tactical view, strategic overview, formations visible',
            'economic': 'isometric market view, human scale, activity centers',
            'diplomatic': 'formal balanced composition, professional setting',
            'human_impact': 'ground-level perspective, emotional framing'
        }
        
        scene_modifiers = {
            'hook': 'dramatic foreground, attention-grabbing',
            'historical_1': 'historical perspective, archival feel',
            'historical_2': 'strategic overview, tactical display',
            'modern_pivot': 'dynamic composition, contemporary feel',
            'consequence': 'human-scale composition, relatable',
            'future_outlook': 'revealing perspective, strategic depth'
        }
        
        base = base_compositions.get(visual_type, 'isometric view, balanced composition')
        modifier = scene_modifiers.get(scene_type, '')
        
        return f"{base}, {modifier}" if modifier else base
    
    def _get_style_weighting(self, visual_type: str) -> str:
        """Get style weighting based on visual type"""
        base_style = "(true 16-bit pixel art:1.5), (retro SNES style:1.3), (isometric perspective:1.2)"
        
        type_specific = {
            'military': '(tactical pixel art:1.2), (detailed equipment:1.1)',
            'economic': '(market pixel style:1.2), (data visualization:1.1)',
            'diplomatic': '(formal pixel style:1.2), (official insignia:1.1)',
            'human_impact': '(emotional pixel style:1.2), (relatable imagery:1.1)'
        }
        
        specific = type_specific.get(visual_type, '')
        return f"{base_style}, {specific}" if specific else base_style
    
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
