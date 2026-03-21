"""
Visual Element Extractor - Dual-layer (LLM + spaCy) entity extraction
Extracts military equipment, locations, actions, and context from news articles
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import our databases
from .military_equipment_db import (
    MILITARY_EQUIPMENT_DB,
    normalize_equipment,
    is_valid_location,
    get_all_locations
)
from .action_mapping import extract_action_verbs, ACTION_VISUAL_MAP


class VisualElementExtractor:
    """
    Dual-layer visual element extraction system
    Primary: LLM-based structured extraction
    Fallback/Validation: spaCy NER
    """
    
    def __init__(self, llm_interface):
        """
        Initialize extractor with LLM interface
        
        Args:
            llm_interface: LLMInterface instance for extraction
        """
        self.llm = llm_interface
        self.spacy_nlp = None
        
        # Try to load spaCy model
        try:
            import spacy
            self.spacy_nlp = spacy.load("en_core_web_lg")
        except:
            print("⚠️  spaCy not available - using LLM-only extraction")
    
    def extract_visual_elements(self, article_text: str) -> Dict[str, Any]:
        """
        Main extraction method - combines LLM and spaCy
        
        Args:
            article_text: Full article text
            
        Returns:
            Dictionary with extracted visual elements
        """
        # Primary: LLM extraction
        llm_elements = self._llm_extract(article_text)
        
        # Validation: spaCy extraction (if available)
        if self.spacy_nlp:
            spacy_entities = self._spacy_extract(article_text)
            # Merge and validate
            validated = self._validate_and_merge(llm_elements, spacy_entities, article_text)
        else:
            validated = llm_elements
        
        # Normalize any military equipment found in primary_subjects
        if validated.get('primary_subjects'):
            validated['primary_subjects'] = [
                normalize_equipment(subj) if any(kw in subj.lower() for kw in ['f-', 's-', 'missile', 'tank', 'carrier', 'destroyer', 'drone']) else subj
                for subj in validated['primary_subjects']
            ]
        
        return validated
    
    def _llm_extract(self, article_text: str) -> Dict[str, Any]:
        """
        Use LLM for structured extraction
        
        Args:
            article_text: Article text
            
        Returns:
            Extracted elements as dictionary
        """
        extraction_prompt = f"""Extract visual elements from this news article for pixel art image generation.
Extract ALL relevant visual subjects — military, economic, diplomatic, civilian, geographic.

{article_text}

Output ONLY valid JSON with these exact keys:
{{
  "primary_subjects": ["specific visual subjects: oil tankers, fighter jets, gas station price boards, diplomatic meetings, shipping ports, civilian crowds, military convoys, trading floors, protest marches, etc."],
  "settings": ["real geographic locations and settings from the article: Strait of Hormuz, Tehran, Wall Street, gas station, parliament building, etc."],
  "actions": ["dynamic action verbs: surging, deploying, signing, queuing, collapsing, launching, negotiating, protesting, etc."],
  "mood": "tense/hopeful/chaotic/urgent/calm",
  "temporal_context": "time of day or weather if mentioned, otherwise empty string"
}}

Be specific. Extract only what is explicitly mentioned or directly implied by the article.
Include BOTH military AND non-military subjects — economic imagery, civilian impacts, diplomatic settings."""

        try:
            response = self.llm.generate(
                prompt=extraction_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            if not response:
                return self._get_empty_elements()
            
            # Extract JSON from response
            extracted = self.llm._extract_json(response)
            
            if not extracted:
                # Fallback: parse manually
                return self._manual_extract(article_text)
            
            return extracted
            
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}")
            return self._manual_extract(article_text)
    
    def _spacy_extract(self, article_text: str) -> Dict[str, List[str]]:
        """
        Use spaCy NER for entity extraction
        
        Args:
            article_text: Article text
            
        Returns:
            Dictionary of entity types and values
        """
        if not self.spacy_nlp:
            return {}
        
        try:
            doc = self.spacy_nlp(article_text)
            
            entities = {
                'GPE': [],      # Geopolitical entities (countries, cities)
                'LOC': [],      # Locations (non-GPE)
                'ORG': [],      # Organizations (military units)
                'FAC': [],      # Facilities (bases, refineries)
                'PRODUCT': []   # Products (military equipment)
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            return entities
            
        except Exception as e:
            print(f"⚠️  spaCy extraction failed: {e}")
            return {}
    
    def _validate_and_merge(
        self,
        llm_elements: Dict[str, Any],
        spacy_entities: Dict[str, List[str]],
        article_text: str
    ) -> Dict[str, Any]:
        """
        Merge LLM and spaCy results, validate against article
        
        Args:
            llm_elements: Elements from LLM
            spacy_entities: Entities from spaCy
            article_text: Original article for validation
            
        Returns:
            Validated and merged elements
        """
        validated = llm_elements.copy()
        
        # Enhance settings with spaCy GPE and LOC entities
        if spacy_entities.get('GPE') or spacy_entities.get('LOC'):
            spacy_locations = spacy_entities.get('GPE', []) + spacy_entities.get('LOC', [])
            llm_settings = validated.get('settings', [])
            
            for loc in spacy_locations:
                if loc not in llm_settings and loc.lower() in article_text.lower():
                    llm_settings.append(loc)
            
            validated['settings'] = llm_settings
        
        # Enhance primary_subjects with spaCy PRODUCT and ORG entities
        if spacy_entities.get('PRODUCT') or spacy_entities.get('ORG'):
            llm_subjects = validated.get('primary_subjects', [])
            
            for product in spacy_entities.get('PRODUCT', []):
                if product not in llm_subjects:
                    llm_subjects.append(product)
            
            validated['primary_subjects'] = llm_subjects
        
        # Validate settings against known database + article text
        if validated.get('settings'):
            valid_settings = []
            
            for setting in validated['settings']:
                if is_valid_location(setting) or setting.lower() in article_text.lower():
                    valid_settings.append(setting)
            
            validated['settings'] = valid_settings
        
        return validated
    
    def _manual_extract(self, article_text: str) -> Dict[str, Any]:
        """
        Fallback manual extraction using regex and keywords
        
        Args:
            article_text: Article text
            
        Returns:
            Extracted elements
        """
        elements = self._get_empty_elements()
        
        # Extract action verbs
        elements['actions'] = extract_action_verbs(article_text)
        
        # Extract military equipment as subjects using regex patterns
        equipment_patterns = [
            r'F-\d+[A-Z]?\s+\w+',  # F-35 Lightning II
            r'S-\d+\s+\w+',         # S-400 Triumf
            r'USS\s+\w+',           # USS Boxer
            r'M\d+A?\d?\s+\w+',     # M1A2 Abrams
        ]
        
        for pattern in equipment_patterns:
            matches = re.findall(pattern, article_text, re.IGNORECASE)
            elements['primary_subjects'].extend(matches)
        
        # Extract settings from known location database
        all_locations = get_all_locations()
        for loc in all_locations:
            if loc.lower() in article_text.lower():
                elements['settings'].append(loc)
        
        # Extract temporal context
        temporal_keywords = ['dawn', 'dusk', 'night', 'morning', 'afternoon', 'evening']
        for keyword in temporal_keywords:
            if keyword in article_text.lower():
                elements['temporal_context'] = keyword
                break
        
        # Determine mood from article tone
        high_tension = ['attack', 'strike', 'bomb', 'assault', 'invasion', 'crisis', 'collapse']
        if any(word in article_text.lower() for word in high_tension):
            elements['mood'] = 'tense'
        else:
            elements['mood'] = 'urgent'
        
        return elements
    
    def _get_empty_elements(self) -> Dict[str, Any]:
        """Return empty elements structure"""
        return {
            'primary_subjects': [],
            'settings': [],
            'actions': [],
            'mood': 'tense',
            'temporal_context': ''
        }
