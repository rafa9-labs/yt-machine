"""
Historical Analyzer - Identifies historical parallels for current news events
Maps 2026 geopolitical events to past crises for deeper context
"""

from typing import Dict, Any, List, Optional
import json


class HistoricalAnalyzer:
    """
    Analyzes current news and identifies 2-3 relevant historical parallels.
    Uses LLM to map geopolitical patterns across decades.
    """
    
    def __init__(self, llm_interface):
        """
        Initialize historical analyzer
        
        Args:
            llm_interface: LLMInterface instance for generating analysis
        """
        self.llm = llm_interface
    
    def find_historical_parallels(
        self,
        article_text: str,
        news_analysis: Dict[str, Any],
        max_parallels: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Find historical parallels for current news event
        
        Args:
            article_text: Full article text
            news_analysis: Current news analysis from LLM
            max_parallels: Maximum number of historical events to find (default 3)
            
        Returns:
            Dictionary with historical parallels and their details
        """
        topic = news_analysis.get('topic', '')
        shift_vector = news_analysis.get('shift_vector', '')
        
        prompt = f"""Analyze this 2026 news event and identify 2-3 relevant historical parallels from the past 50 years.

CURRENT EVENT (2026):
Topic: {topic}
Shift Vector: {shift_vector}

Article excerpt:
{article_text[:1500]}

Find historical events that share similar geopolitical patterns:
- Similar conflicts (US-Iran, US-Iraq, China-US trade wars, etc.)
- Similar economic pressures (oil crises, sanctions, blockades)
- Similar diplomatic dynamics (interventions, negotiations, alliances)

For each historical parallel, extract:
1. Event name and year
2. Key players (countries, leaders)
3. Military equipment used (era-specific)
4. Geopolitical outcome
5. Relevance to 2026 event (why this comparison matters)

Output ONLY valid JSON:
{{
  "parallels": [
    {{
      "event_name": "1991 Gulf War",
      "year": 1991,
      "era": "1990s",
      "key_players": ["US", "Iraq", "Coalition forces"],
      "military_equipment": ["F-117 Nighthawk", "M1A1 Abrams", "Patriot missiles", "Tomahawk cruise missiles"],
      "outcome": "Coalition victory, Iraq expelled from Kuwait, sanctions imposed",
      "relevance_to_current": "Similar US military intervention in Middle East over oil access and regional stability"
    }},
    {{
      "event_name": "1980s Iran-Iraq Tanker War",
      "year": 1987,
      "era": "1980s",
      "key_players": ["Iran", "Iraq", "US Navy"],
      "military_equipment": ["USS Stark", "Exocet missiles", "Iranian speedboats", "F-14 Tomcat"],
      "outcome": "US Navy escorts protect oil tankers, Operation Earnest Will",
      "relevance_to_current": "Precedent for Strait of Hormuz tensions and naval protection of oil shipments"
    }}
  ],
  "historical_pattern": "Recurring Middle East oil crises trigger US military intervention to secure energy supplies",
  "key_difference_2026": "China now major player in energy markets, complicating traditional US-Iran dynamics"
}}

Be historically accurate. Only include events with clear parallels."""

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1200
        )
        
        if not response:
            return None
        
        result = self.llm._extract_json(response)
        
        if not result or 'parallels' not in result:
            print("⚠️  Historical analysis failed - using fallback")
            return self._get_fallback_parallels(topic, shift_vector)
        
        # Limit to max_parallels
        if len(result['parallels']) > max_parallels:
            result['parallels'] = result['parallels'][:max_parallels]
        
        return result
    
    def _get_fallback_parallels(self, topic: str, shift_vector: str) -> Dict[str, Any]:
        """
        Fallback historical parallels based on topic keywords
        
        Args:
            topic: News topic
            shift_vector: Geopolitical shift vector
            
        Returns:
            Basic historical parallels dictionary
        """
        topic_lower = topic.lower()
        
        # Iran-related fallbacks
        if 'iran' in topic_lower or 'hormuz' in topic_lower:
            return {
                'parallels': [
                    {
                        'event_name': '1991 Gulf War',
                        'year': 1991,
                        'era': '1990s',
                        'key_players': ['US', 'Iraq', 'Coalition'],
                        'military_equipment': ['F-117 Nighthawk', 'M1A1 Abrams', 'Patriot missiles'],
                        'outcome': 'Coalition victory, regional stability restored',
                        'relevance_to_current': 'US military intervention in Persian Gulf region'
                    },
                    {
                        'event_name': '1980s Tanker War',
                        'year': 1987,
                        'era': '1980s',
                        'key_players': ['Iran', 'Iraq', 'US Navy'],
                        'military_equipment': ['USS Stark', 'F-14 Tomcat', 'Iranian speedboats'],
                        'outcome': 'US Navy protection of oil tankers',
                        'relevance_to_current': 'Strait of Hormuz oil shipping security'
                    }
                ],
                'historical_pattern': 'Recurring Persian Gulf crises over oil access',
                'key_difference_2026': 'Multipolar energy market with China as major player'
            }
        
        # China-related fallbacks
        elif 'china' in topic_lower or 'tariff' in topic_lower or 'trade' in topic_lower:
            return {
                'parallels': [
                    {
                        'event_name': '2018-2020 US-China Trade War',
                        'year': 2018,
                        'era': '2010s',
                        'key_players': ['US', 'China'],
                        'military_equipment': [],
                        'outcome': 'Tariffs imposed, Phase One deal signed, tensions remain',
                        'relevance_to_current': 'Precedent for US-China economic confrontation'
                    }
                ],
                'historical_pattern': 'Rising power challenging established hegemon through economic means',
                'key_difference_2026': 'Technology decoupling and supply chain restructuring'
            }
        
        # Generic geopolitical fallback
        else:
            return {
                'parallels': [
                    {
                        'event_name': 'Cold War Proxy Conflicts',
                        'year': 1980,
                        'era': '1980s',
                        'key_players': ['US', 'Soviet Union'],
                        'military_equipment': [],
                        'outcome': 'Superpower competition through regional conflicts',
                        'relevance_to_current': 'Great power competition dynamics'
                    }
                ],
                'historical_pattern': 'Geopolitical competition over strategic resources and influence',
                'key_difference_2026': 'Multipolar world with multiple competing powers'
            }
