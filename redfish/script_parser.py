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
