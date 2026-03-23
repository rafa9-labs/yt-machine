"""
Script Parser - Extracts specific actions, subjects, and settings from script segments
to drive highly relevant, action-specific image generation prompts.
"""

import re
from typing import Dict, Any, List, Optional


# Action verb categories mapped to visual descriptions
ACTION_VISUAL_MAPPINGS = {
    # Military actions
    "intercept": "intercepts mid-flight, anti-missile system firing, debris and explosion",
    "launch": "launches missile, rocket exhaust trail rising dramatically",
    "strike": "delivers precision airstrike, explosion and smoke plume",
    "deploy": "deploys forces, military convoy moving through terrain",
    "bomb": "aerial bombing run, explosions below, aircraft banking",
    "blockade": "naval blockade formation, warships in tight line across strait",
    "invade": "ground invasion, armor crossing border, troops advancing",
    "evacuate": "mass evacuation, crowds fleeing, convoys of vehicles",
    "sanction": "economic sanctions, frozen bank screens, financial terminals",
    "negotiate": "high-level negotiation, diplomats around table, flags visible",
    "sign": "treaty signing ceremony, officials shaking hands at formal table",
    "escalate": "military buildup, forces massing at border",
    "mobilize": "troop mobilization, soldiers boarding transport aircraft",
    "patrol": "naval patrol, warship cutting through water at speed",
    "seize": "forces seizing strategic point, flags being raised",
    "retreat": "tactical withdrawal, forces moving back under fire",
    "collapse": "economic collapse, stock market screens flashing red",
    "surge": "price surge, commodity screens, trading floor in chaos",
    "protest": "mass protest, crowds in streets, signs and smoke",
    "declare": "formal declaration, press conference, officials at podium",
}

# Subject-specific visual enhancements
SUBJECT_VISUAL_ENHANCEMENTS = {
    "iran": "Iranian Revolutionary Guard, green and amber camouflage",
    "israel": "Israeli Defense Forces, star of David insignia",
    "russia": "Russian military, red star markings, olive drab",
    "china": "People's Liberation Army, red star, modern equipment",
    "ukraine": "Ukrainian forces, blue and yellow markings",
    "nato": "NATO multi-nation forces, distinctive insignia",
    "us": "US military, American flag markings, modern equipment",
    "hamas": "militant forces, urban combat, Gaza cityscape",
    "hezbollah": "militia forces, Lebanese terrain, fortified positions",
    "oil": "oil tanker, crude oil prices board, petroleum refinery",
    "missile": "ballistic missile mid-flight, contrail against sky",
    "drone": "UAV swarm, quadcopters, military drone hovering",
    "nuclear": "nuclear facility, cooling towers, radiation warning signs",
    "sanctions": "frozen assets screen, banking terminal, SWIFT network diagram",
}

# Setting-specific visual context
SETTING_VISUAL_CONTEXT = {
    "strait of hormuz": "narrow strait, tankers passing, military vessels, Persian Gulf waters at dusk",
    "hormuz": "narrow strait waters, oil tankers queued, IRGC patrol boats",
    "persian gulf": "open gulf waters, oil platforms on horizon, military activity",
    "red sea": "Red Sea shipping lane, Houthi threat, naval escorts",
    "gaza": "urban warfare, dense cityscape, destroyed buildings, smoke",
    "ukraine": "Eastern European steppe, destroyed armor, winter conditions",
    "taiwan": "island fortress, strait waters, PLA naval threat",
    "south china sea": "contested waters, artificial islands, military installations",
    "washington": "US Capitol, White House, diplomatic setting, formal halls",
    "tehran": "Iranian capital, Persian architecture, government buildings",
    "beijing": "Chinese capital, Tiananmen, modern skyline, government towers",
    "moscow": "Red Square, Kremlin towers, Russian government buildings",
    "wall street": "financial district, stock exchange trading floor, screens",
    "arctic": "frozen tundra, ice shelf, military assets in white camouflage",
}


class ScriptParser:
    """
    Extracts specific visual information from script segments for precise image generation.
    Produces action-specific prompts instead of generic scene descriptions.
    """

    def parse_segment(self, segment_text: str, segment_name: str) -> Dict[str, Any]:
        """
        Parse a single script segment for visual elements.

        Args:
            segment_text: The script segment text
            segment_name: e.g. 'hook', 'historical_1', 'modern_pivot'

        Returns:
            Dict with action, subject, setting, era, mood, visual_description
        """
        text_lower = segment_text.lower()

        action = self._extract_primary_action(text_lower)
        subject = self._extract_primary_subject(text_lower)
        setting = self._extract_primary_setting(text_lower)
        era = self._extract_era(segment_text, segment_name)
        mood = self._extract_mood(text_lower)
        numbers = self._extract_key_numbers(segment_text)

        return {
            "segment": segment_name,
            "raw_text": segment_text,
            "action": action,
            "subject": subject,
            "setting": setting,
            "era": era,
            "mood": mood,
            "numbers": numbers,
        }

    def parse_all_segments(self, script: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse all script segments and return ordered visual data.

        Args:
            script: Full script dict with segment keys

        Returns:
            List of parsed segment dicts in script order
        """
        if "historical_1" in script:
            segment_names = [
                "hook", "historical_1", "historical_2",
                "modern_pivot", "consequence", "future_outlook"
            ]
        else:
            segment_names = ["hook", "context", "escalation", "consequence", "twist"]

        parsed = []
        for name in segment_names:
            text = script.get(name, "")
            if text:
                parsed.append(self.parse_segment(text, name))

        return parsed

    def build_action_prompt(self, parsed: Dict[str, Any], style_suffix: str) -> str:
        """
        Convert parsed segment data into a highly specific image generation prompt.

        Args:
            parsed: Output of parse_segment()
            style_suffix: Visual style string to append

        Returns:
            Complete, action-specific image generation prompt
        """
        parts = []

        subject = parsed.get("subject", "")
        action = parsed.get("action", "")
        setting = parsed.get("setting", "")
        mood = parsed.get("mood", "tense")
        era = parsed.get("era", "2020s")
        numbers = parsed.get("numbers", [])

        # Subject enhancement
        subject_visual = self._get_subject_visual(subject)
        if subject_visual:
            parts.append(subject_visual)
        elif subject:
            parts.append(subject)

        # Action visual
        action_visual = self._get_action_visual(action)
        if action_visual:
            parts.append(action_visual)
        elif action:
            parts.append(action)

        # Setting enhancement
        setting_visual = self._get_setting_visual(setting)
        if setting_visual:
            parts.append(setting_visual)
        elif setting:
            parts.append(f"at {setting}")

        # Mood and lighting
        mood_lighting = {
            "tense": "dramatic dusk lighting, orange and red sky",
            "chaotic": "stormy skies, smoke and debris",
            "urgent": "harsh midday light, high contrast shadows",
            "historical": "sepia-amber tint, archival atmosphere",
            "hopeful": "golden hour light, warm tones",
            "economic": "cold blue financial district lighting",
        }
        era_lighting = {
            "1960s": "grainy vintage film look, muted colors, historical",
            "1970s": "warm amber 70s palette, film grain",
            "1980s": "cold war era, olive drab, high contrast",
            "1990s": "early 90s tech, desert sand palette",
            "2000s": "digital era, sharp clean lines",
            "2010s": "modern HD feel, realistic lighting",
            "2020s": mood_lighting.get(mood, "dramatic dusk lighting"),
        }

        lighting = era_lighting.get(era, mood_lighting.get(mood, "dramatic lighting"))
        parts.append(lighting)

        # Key numbers for specificity
        if numbers:
            num_str = numbers[0]
            parts.append(f"numerical display showing {num_str}")

        prompt_body = ", ".join(filter(None, parts))

        # Final negative reinforcement
        negative_note = "NO text overlays, NO watermarks, NO UI elements"
        return f"{prompt_body}, {negative_note}, {style_suffix}"

    # ─── Private helpers ───────────────────────────────────────────────────────

    def _extract_primary_action(self, text: str) -> str:
        """Find the most visually significant action verb in text."""
        for verb in ACTION_VISUAL_MAPPINGS:
            if verb in text:
                return verb
        # Fallback: look for any strong verb
        strong_verbs = [
            "attacking", "defending", "crossing", "advancing", "retreating",
            "collapsing", "surging", "halting", "marching", "fleeing",
            "trading", "bombing", "firing", "moving"
        ]
        for verb in strong_verbs:
            if verb in text:
                return verb.rstrip("ing")
        return "in dramatic confrontation"

    def _extract_primary_subject(self, text: str) -> str:
        """Extract the main visual subject from text."""
        for subject_key in SUBJECT_VISUAL_ENHANCEMENTS:
            if subject_key in text:
                return subject_key
        # Fallback: look for capitalized proper nouns (names, orgs)
        words = text.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                return word
        return ""

    def _extract_primary_setting(self, text: str) -> str:
        """Extract the primary geographic setting."""
        # Check multi-word settings first (longer matches take priority)
        for setting in sorted(SETTING_VISUAL_CONTEXT.keys(), key=len, reverse=True):
            if setting in text:
                return setting
        # Common location keywords
        locations = [
            "strait", "gulf", "sea", "ocean", "coast", "border",
            "capital", "city", "port", "base", "facility", "summit"
        ]
        for loc in locations:
            if loc in text:
                # Try to get context word before it
                match = re.search(rf'(\w+)\s+{loc}', text)
                if match:
                    return f"{match.group(1)} {loc}"
                return loc
        return ""

    def _extract_era(self, text: str, segment_name: str) -> str:
        """Determine the time era from text or segment name."""
        # Historical segments
        if segment_name in ("historical_1", "historical_2"):
            for year_pattern in [
                r'\b(19[0-9]{2})\b',  # 1900s
                r'\b(20[0-2][0-9])\b',  # 2000-2029
            ]:
                match = re.search(year_pattern, text)
                if match:
                    year = int(match.group(1))
                    if year < 1970:
                        return "1960s"
                    elif year < 1980:
                        return "1970s"
                    elif year < 1990:
                        return "1980s"
                    elif year < 2000:
                        return "1990s"
                    elif year < 2010:
                        return "2000s"
                    elif year < 2020:
                        return "2010s"
            return "1990s"  # default historical

        # Check for decade mentions in any segment
        decade_map = {
            "nineteen sixties": "1960s", "1960s": "1960s",
            "nineteen seventies": "1970s", "1970s": "1970s",
            "nineteen eighties": "1980s", "1980s": "1980s",
            "nineteen nineties": "1990s", "1990s": "1990s",
        }
        text_lower = text.lower()
        for phrase, decade in decade_map.items():
            if phrase in text_lower:
                return decade

        return "2020s"

    def _extract_mood(self, text: str) -> str:
        """Determine the emotional mood from text."""
        mood_keywords = {
            "tense": ["crisis", "war", "attack", "strike", "missiles", "bombs", "blockade", "escalat"],
            "chaotic": ["chaos", "collapse", "collapse", "panic", "fleeing", "explosion", "destabil"],
            "historical": ["decades ago", "in the", "years", "century", "history", "back in"],
            "economic": ["oil", "prices", "billion", "trillion", "markets", "inflation", "dollar"],
            "hopeful": ["deal", "peace", "agreement", "ceasefire", "negotiat", "diplomacy"],
        }
        scores = {mood: 0 for mood in mood_keywords}
        for mood, keywords in mood_keywords.items():
            for kw in keywords:
                if kw in text:
                    scores[mood] += 1
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "tense"

    def _extract_key_numbers(self, text: str) -> List[str]:
        """Extract significant numbers and percentages from text."""
        # Percentages
        pcts = re.findall(r'\d+(?:\.\d+)?%', text)
        # Dollar amounts
        dollars = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion))?', text)
        # Plain numbers with context
        nums = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        result = pcts + dollars
        if not result and nums:
            result = nums[:1]
        return result[:2]

    def _get_action_visual(self, action: str) -> str:
        """Look up enhanced visual description for an action."""
        return ACTION_VISUAL_MAPPINGS.get(action.lower().strip(), "")

    def _get_subject_visual(self, subject: str) -> str:
        """Look up enhanced visual description for a subject."""
        return SUBJECT_VISUAL_ENHANCEMENTS.get(subject.lower().strip(), "")

    def _get_setting_visual(self, setting: str) -> str:
        """Look up enhanced visual description for a setting."""
        return SETTING_VISUAL_CONTEXT.get(setting.lower().strip(), "")
    
    def extract_visual_concepts(self, segment_text: str, trending_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract ALL visual concepts from a script segment for precise image generation.
        This is the primary method for script-to-image synchronization.
        
        Args:
            segment_text: The script segment text
            trending_context: Optional trending words dictionary from TrendingAnalyzer
            
        Returns:
            Dictionary with comprehensive visual concept data:
            {
                "primary_concept": str,
                "secondary_concepts": List[str],
                "visual_type": str,  # military, economic, diplomatic, human_impact, mixed
                "trending_boost": float,  # 0.0-1.0
                "action": str,
                "subjects": List[str],
                "setting": str,
                "mood": str,
                "numbers": List[str],
                "emphasis": str  # What to emphasize in the image
            }
        """
        if not segment_text:
            return self._get_empty_concepts()
        
        text_lower = segment_text.lower()
        trending_context = trending_context or {}
        
        # Extract basic elements
        action = self._extract_primary_action(text_lower)
        primary_subject = self._extract_primary_subject(text_lower)
        setting = self._extract_primary_setting(text_lower)
        mood = self._extract_mood(text_lower)
        numbers = self._extract_key_numbers(segment_text)
        
        # Extract ALL subjects (not just primary)
        all_subjects = self._extract_all_subjects(text_lower)
        if primary_subject and primary_subject not in all_subjects:
            all_subjects.insert(0, primary_subject)
        
        # Extract secondary concepts
        secondary_concepts = self._extract_secondary_concepts(text_lower)
        
        # Determine visual type
        visual_type = self._determine_visual_type(text_lower, all_subjects, action)
        
        # Calculate trending boost
        trending_boost = self._calculate_trending_boost(
            segment_text, all_subjects, action, setting, trending_context
        )
        
        # Determine primary concept (what the image should show)
        primary_concept = self._determine_primary_concept(
            text_lower, all_subjects, action, setting, visual_type
        )
        
        # Determine emphasis
        emphasis = self._determine_emphasis(visual_type, primary_concept, trending_boost)
        
        return {
            "primary_concept": primary_concept,
            "secondary_concepts": secondary_concepts,
            "visual_type": visual_type,
            "trending_boost": trending_boost,
            "action": action,
            "subjects": all_subjects,
            "setting": setting,
            "mood": mood,
            "numbers": numbers,
            "emphasis": emphasis
        }
    
    def _extract_all_subjects(self, text: str) -> List[str]:
        """Extract all visual subjects mentioned in text."""
        subjects = []
        
        # Check all known subjects
        for subject_key in SUBJECT_VISUAL_ENHANCEMENTS.keys():
            if subject_key in text:
                subjects.append(subject_key)
        
        # Economic subjects
        economic_subjects = [
            'oil prices', 'gas prices', 'stock market', 'trading floor',
            'price board', 'gas station', 'families', 'civilians', 'queues',
            'shelves', 'market', 'inflation', 'dollar', 'economy'
        ]
        for subj in economic_subjects:
            if subj in text and subj not in subjects:
                subjects.append(subj)
        
        # Diplomatic subjects
        diplomatic_subjects = [
            'summit', 'negotiation', 'treaty', 'agreement', 'diplomats',
            'ministers', 'officials', 'meeting', 'talks', 'embassy'
        ]
        for subj in diplomatic_subjects:
            if subj in text and subj not in subjects:
                subjects.append(subj)
        
        # Human impact subjects
        human_subjects = [
            'families', 'civilians', 'people', 'residents', 'refugees',
            'protesters', 'crowds', 'victims', 'evacuees'
        ]
        for subj in human_subjects:
            if subj in text and subj not in subjects:
                subjects.append(subj)
        
        return subjects[:5]  # Limit to top 5
    
    def _extract_secondary_concepts(self, text: str) -> List[str]:
        """Extract secondary visual concepts."""
        concepts = []
        
        # Look for specific visual elements
        visual_elements = [
            'explosion', 'smoke', 'fire', 'debris', 'wreckage',
            'convoy', 'formation', 'patrol', 'surveillance',
            'screens', 'charts', 'graphs', 'data', 'indicators',
            'flags', 'insignia', 'uniforms', 'equipment',
            'buildings', 'infrastructure', 'facilities'
        ]
        
        for element in visual_elements:
            if element in text:
                concepts.append(element)
        
        return concepts[:3]  # Limit to top 3
    
    def _determine_visual_type(self, text: str, subjects: List[str], action: str) -> str:
        """Determine the primary visual type of the segment."""
        scores = {
            'military': 0,
            'economic': 0,
            'diplomatic': 0,
            'human_impact': 0
        }
        
        # Military keywords
        military_kw = [
            'military', 'forces', 'troops', 'missile', 'strike', 'attack',
            'war', 'combat', 'weapon', 'tank', 'aircraft', 'naval', 'drone',
            'bombing', 'invasion', 'blockade'
        ]
        for kw in military_kw:
            if kw in text:
                scores['military'] += 1
        
        # Economic keywords
        economic_kw = [
            'price', 'economy', 'market', 'inflation', 'dollar', 'oil',
            'gas', 'trade', 'cost', 'surge', 'shortage', 'supply'
        ]
        for kw in economic_kw:
            if kw in text:
                scores['economic'] += 1
        
        # Diplomatic keywords
        diplomatic_kw = [
            'diplomatic', 'negotiation', 'treaty', 'summit', 'talks',
            'agreement', 'deal', 'minister', 'ambassador', 'envoy'
        ]
        for kw in diplomatic_kw:
            if kw in text:
                scores['diplomatic'] += 1
        
        # Human impact keywords
        human_kw = [
            'families', 'civilians', 'people', 'queue', 'protest',
            'refugees', 'evacuate', 'shelter', 'crisis', 'panic'
        ]
        for kw in human_kw:
            if kw in text:
                scores['human_impact'] += 1
        
        # Check subjects
        for subject in subjects:
            if any(kw in subject for kw in ['price', 'gas station', 'market', 'economy']):
                scores['economic'] += 2
            if any(kw in subject for kw in ['families', 'civilians', 'people']):
                scores['human_impact'] += 2
        
        # Return type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'military'  # Default
        
        # If multiple types have high scores, return mixed
        high_scorers = [k for k, v in scores.items() if v >= max_score * 0.7]
        if len(high_scorers) > 1:
            return 'mixed'
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_trending_boost(self, text: str, subjects: List[str], 
                                   action: str, setting: str, 
                                   trending_context: Dict[str, Any]) -> float:
        """Calculate boost score based on trending context."""
        if not trending_context:
            return 0.0
        
        boost_scores = []
        text_lower = text.lower()
        
        # Check if any trending terms appear in text
        for trending_term, data in trending_context.items():
            if trending_term in text_lower:
                boost_scores.append(data.get('score', 0.0))
        
        # Check subjects
        for subject in subjects:
            for trending_term, data in trending_context.items():
                if trending_term in subject.lower() or subject.lower() in trending_term:
                    boost_scores.append(data.get('score', 0.0))
        
        # Check setting
        if setting:
            for trending_term, data in trending_context.items():
                if trending_term in setting.lower() or setting.lower() in trending_term:
                    boost_scores.append(data.get('score', 0.0))
        
        return max(boost_scores) if boost_scores else 0.0
    
    def _determine_primary_concept(self, text: str, subjects: List[str], 
                                    action: str, setting: str, 
                                    visual_type: str) -> str:
        """Determine the primary visual concept (what image should show)."""
        # Build concept from available elements
        parts = []
        
        if subjects:
            parts.append(subjects[0])
        
        if action and action != "in dramatic confrontation":
            parts.append(action)
        
        if setting:
            parts.append(f"at {setting}")
        
        if parts:
            return " ".join(parts)
        
        # Fallback: extract key phrase from text
        sentences = text.split('.')
        if sentences:
            return sentences[0][:80]
        
        return "geopolitical scene"
    
    def _determine_emphasis(self, visual_type: str, primary_concept: str, 
                           trending_boost: float) -> str:
        """Determine what to emphasize in the image."""
        emphasis_map = {
            'military': 'tactical positioning, equipment detail, strategic perspective',
            'economic': 'price indicators, market data, human scale impact',
            'diplomatic': 'formal setting, official insignia, meeting atmosphere',
            'human_impact': 'civilian perspective, emotional impact, everyday life',
            'mixed': 'balanced composition, multiple perspectives'
        }
        
        base_emphasis = emphasis_map.get(visual_type, 'dramatic composition')
        
        # Add trending boost note
        if trending_boost > 0.5:
            base_emphasis += ', emphasize trending elements'
        
        return base_emphasis
    
    def _get_empty_concepts(self) -> Dict[str, Any]:
        """Return empty concepts structure."""
        return {
            "primary_concept": "",
            "secondary_concepts": [],
            "visual_type": "military",
            "trending_boost": 0.0,
            "action": "",
            "subjects": [],
            "setting": "",
            "mood": "tense",
            "numbers": [],
            "emphasis": ""
        }
    
    def add_visual_grounding(self, concepts: Dict[str, Any], scene_type: str) -> str:
        """
        Add spatial grounding instructions for better object positioning.
        This ensures objects are arranged logically in the scene.
        """
        subjects = concepts.get('subjects', [])
        action = concepts.get('action', '')
        setting = concepts.get('setting', '')
        visual_type = concepts.get('visual_type', 'general')
        
        grounding_instructions = []
        
        # Military spatial arrangements
        if visual_type == 'military':
            if 'blockade' in action.lower() or 'naval' in setting.lower():
                grounding_instructions.append("(warships in tactical line formation:1.3)")
                grounding_instructions.append("(strait visible in background:1.1)")
            
            if 'missile' in ' '.join(subjects).lower():
                grounding_instructions.append("(missile positioned prominently:1.2)")
                grounding_instructions.append("(launch trajectory visible:1.1)")
            
            if 'tank' in ' '.join(subjects).lower() or 'armor' in action.lower():
                grounding_instructions.append("(armored formation in foreground:1.2)")
                grounding_instructions.append("(tactical spacing visible:1.1)")
        
        # Economic spatial arrangements
        elif visual_type == 'economic':
            if 'market' in setting.lower() or 'trading' in action.lower():
                grounding_instructions.append("(market displays prominent:1.2)")
                grounding_instructions.append("(human scale perspective:1.1)")
            
            if 'gas station' in setting.lower():
                grounding_instructions.append("(price board clearly visible:1.3)")
                grounding_instructions.append("(queue formation organized:1.1)")
        
        # Diplomatic spatial arrangements
        elif visual_type == 'diplomatic':
            if 'summit' in setting.lower() or 'negotiation' in action.lower():
                grounding_instructions.append("(officials around formal table:1.2)")
                grounding_instructions.append("(flags and insignia visible:1.1)")
            
            if 'treaty' in action.lower() or 'agreement' in action.lower():
                grounding_instructions.append("(document signing focus:1.2)")
                grounding_instructions.append("(professional setting balanced:1.1)")
        
        # Human impact spatial arrangements
        elif visual_type == 'human_impact':
            if 'protest' in action.lower():
                grounding_instructions.append("(crowd formation dynamic:1.2)")
                grounding_instructions.append("(signs and banners visible:1.1)")
            
            if 'evacuee' in ' '.join(subjects).lower() or 'refugee' in ' '.join(subjects).lower():
                grounding_instructions.append("(human scale perspective:1.3)")
                grounding_instructions.append("(emotional composition:1.2)")
        
        # Scene-specific composition
        scene_grounding = self._get_scene_grounding(scene_type)
        if scene_grounding:
            grounding_instructions.append(scene_grounding)
        
        return ', '.join(grounding_instructions)
    
    def _get_scene_grounding(self, scene_type: str) -> str:
        """Get scene-specific spatial grounding instructions."""
        scene_grounding = {
            'hook': '(dramatic foreground composition:1.2)',
            'historical_1': '(historical perspective depth:1.1)',
            'historical_2': '(strategic overview layout:1.2)',
            'modern_pivot': '(dynamic contemporary composition:1.1)',
            'consequence': '(human-centered framing:1.2)',
            'future_outlook': '(revealing strategic perspective:1.1)'
        }
        
        return scene_grounding.get(scene_type, '')
