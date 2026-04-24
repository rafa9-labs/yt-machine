"""
News Salience Extractor - Extracts key narrative elements using the news-values framework
(Galtung & Ruge + Dahlstrom 2014 narrative science principles)
"""

import json
from typing import Dict, Any, Optional


class SalienceExtractor:
    """
    Extracts structured salience data from news articles:
    conflict, consequence chain, emotional anchors, surprise angle,
    human impact, key visual subjects, and receptive terms.
    """
    
    def __init__(self, llm_interface):
        """
        Args:
            llm_interface: LLMInterface instance with config loaded
        """
        self.llm = llm_interface
        self.config = llm_interface.config.get("prompts", {}).get("salience_extractor", {})
    
    def extract(self, article_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract salience data from article text.
        
        Args:
            article_text: Full article text (ideally from trafilatura)
            
        Returns:
            Dictionary with salience fields, or None on failure
        """
        system_prompt = self.config.get("system_prompt", "")
        temperature = self.config.get("temperature", 0.4)
        max_tokens = self.config.get("max_tokens", 800)
        
        if not system_prompt:
            print("⚠️  No salience_extractor prompt in config, using inline fallback")
            system_prompt = self._fallback_system_prompt()
        
        prompt = f"Extract the key narrative salience elements from this article:\n\n{article_text}"
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not response:
                return self._empty_salience()
            
            result = self.llm._extract_json(response)
            
            if not result or not isinstance(result, dict):
                print(f"⚠️  Salience JSON parse failed, raw: {response[:200]}")
                return self._empty_salience()
            
            # Validate and normalize fields
            return self._normalize(result)
            
        except Exception as e:
            print(f"⚠️  Salience extraction error: {e}")
            return self._empty_salience()
    
    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected fields exist with correct types."""
        return {
            'conflict': str(raw.get('conflict', '')),
            'consequence_chain': list(raw.get('consequence_chain', [])),
            'emotional_anchors': list(raw.get('emotional_anchors', [])),
            'surprise_angle': str(raw.get('surprise_angle', '')),
            'human_impact': str(raw.get('human_impact', '')),
            'key_visual_subjects': list(raw.get('key_visual_subjects', [])),
            'receptive_terms': list(raw.get('receptive_terms', []))
        }
    
    def _empty_salience(self) -> Dict[str, Any]:
        """Return empty but valid salience structure."""
        return {
            'conflict': '',
            'consequence_chain': [],
            'emotional_anchors': [],
            'surprise_angle': '',
            'human_impact': '',
            'key_visual_subjects': [],
            'receptive_terms': []
        }
    
    def _fallback_system_prompt(self) -> str:
        """Inline fallback if config prompt is missing."""
        return (
            "You are a senior editorial analyst. Read the news article and extract "
            "the most salient narrative elements.\n\n"
            "Output ONLY valid JSON:\n"
            "{\n"
            "  \"conflict\": \"Who vs who and whats at stake\",\n"
            "  \"consequence_chain\": [\"first-order effect\", \"second-order effect\"],\n"
            "  \"emotional_anchors\": [\"specific numbers\", \"recognizable names\"],\n"
            "  \"surprise_angle\": \"The hidden angle\",\n"
            "  \"human_impact\": \"How a real person is affected\",\n"
            "  \"key_visual_subjects\": [\"oil tanker\", \"gas prices\", \"diplomatic meeting\"],\n"
            "  \"receptive_terms\": [\"soar\", \"crisis\", \"collapse\"]\n"
            "}"
        )
