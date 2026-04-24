"""
Platform Metadata Generator - Creates optimized captions, titles, and hashtags
for TikTok, YouTube Shorts, and Instagram Reels
"""

from typing import Dict, Any, List


class PlatformMetadataGenerator:
    """
    Generates platform-optimized metadata for maximum reach and monetization.
    """
    
    def __init__(self):
        pass
    
    def generate_all_metadata(
        self,
        script: Dict[str, Any],
        news_analysis: Dict[str, Any],
        historical_parallels: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata for all platforms
        
        Args:
            script: Synthesized script with segments
            news_analysis: News analysis data
            historical_parallels: Historical parallels data
            
        Returns:
            Dictionary with platform-specific metadata
        """
        topic = news_analysis.get('topic', 'Breaking News')
        
        # Extract key terms for SEO
        key_terms = self._extract_key_terms(script, news_analysis, historical_parallels)
        
        return {
            'tiktok': self._generate_tiktok_metadata(topic, key_terms, script),
            'youtube': self._generate_youtube_metadata(topic, key_terms, script),
            'instagram': self._generate_instagram_metadata(topic, key_terms, script),
            'common_hashtags': self._generate_hashtags(key_terms)
        }
    
    def _extract_key_terms(
        self,
        script: Dict[str, Any],
        news_analysis: Dict[str, Any],
        historical_parallels: Dict[str, Any] = None
    ) -> List[str]:
        """Extract key search terms from content"""
        terms = []
        
        # From topic
        topic = news_analysis.get('topic', '')
        if topic:
            terms.append(topic)
        
        # From historical parallels
        if historical_parallels and 'parallels' in historical_parallels:
            for parallel in historical_parallels['parallels']:
                event_name = parallel.get('event_name', '')
                if event_name:
                    terms.append(event_name)
        
        # Common geopolitical terms
        full_text = script.get('full_text', '').lower()
        
        geo_terms = [
            'iran', 'china', 'russia', 'ukraine', 'israel', 'gaza',
            'oil', 'sanctions', 'tariff', 'trade war', 'blockade',
            'strait of hormuz', 'taiwan', 'south china sea'
        ]
        
        for term in geo_terms:
            if term in full_text:
                terms.append(term)
        
        return list(set(terms))[:10]  # Limit to 10 unique terms
    
    def _generate_tiktok_metadata(
        self,
        topic: str,
        key_terms: List[str],
        script: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate TikTok-optimized metadata
        
        TikTok prioritizes:
        - High-volume search terms in caption
        - Educational content classification
        - 60+ second videos for monetization
        """
        # Build search-optimized caption
        search_terms = ' '.join(key_terms[:5])
        
        caption = f"{topic} explained | {search_terms} | Geopolitical analysis 2026"
        
        # Add historical context if available
        if 'historical_1' in script:
            caption += " | Historical comparison"
        
        return {
            'caption': caption[:150],  # TikTok caption limit
            'cta': "Comment INTEL for full analysis",
            'classification': 'Educational'
        }
    
    def _generate_youtube_metadata(
        self,
        topic: str,
        key_terms: List[str],
        script: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate YouTube Shorts-optimized metadata
        
        YouTube prioritizes:
        - Keyword-rich titles
        - Descriptive text
        - Search optimization
        """
        # Build title with historical context
        if 'historical_1' in script and 'visual_scenes' in script:
            # Extract years from historical parallels
            years = []
            for scene in script['visual_scenes']:
                if 'historical' in scene.get('segment', ''):
                    era = scene.get('era', '')
                    if era and era != '2020s':
                        years.append(era)
            
            if years:
                title = f"2026 {topic} vs {years[0]} Crisis: What History Teaches Us"
            else:
                title = f"{topic}: Historical Analysis & 2026 Implications"
        else:
            title = f"{topic}: Geopolitical Analysis 2026"
        
        # Build description
        description = f"{topic}\n\n"
        description += "Deep dive into the geopolitical implications with historical context.\n\n"
        description += f"Key topics: {', '.join(key_terms[:8])}\n\n"
        description += "#geopolitics #news #analysis"
        
        return {
            'title': title[:100],  # YouTube title limit
            'description': description[:500]
        }
    
    def _generate_instagram_metadata(
        self,
        topic: str,
        key_terms: List[str],
        script: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate Instagram Reels-optimized metadata
        
        Instagram prioritizes:
        - Engagement (saves, shares, DMs)
        - Authority building
        - Community interaction
        """
        # Build authority-focused caption
        caption = f"🔍 {topic}\n\n"
        caption += "Geopolitical intelligence analysis with historical context.\n\n"
        
        # Add DM trigger for list building
        caption += "💬 Comment INTEL for the full briefing\n\n"
        
        # Add hashtags
        hashtags = self._generate_hashtags(key_terms)
        caption += ' '.join(hashtags[:15])  # Instagram allows 30, use 15
        
        return {
            'caption': caption[:2200],  # Instagram caption limit
            'cta': 'Comment INTEL',
            'engagement_trigger': 'DM automation'
        }
    
    def _generate_hashtags(self, key_terms: List[str]) -> List[str]:
        """
        Generate hashtag mix: trending + evergreen
        
        Strategy:
        - Mix high-volume and niche hashtags
        - Include evergreen geopolitical tags
        - Add temporal tags (2026, current year)
        """
        hashtags = [
            '#geopolitics',
            '#news',
            '#worldnews',
            '#geopoliticalanalysis',
            '#history',
            '#explained',
            '#education',
            '#2026'
        ]
        
        # Add term-specific hashtags
        for term in key_terms:
            # Clean and format as hashtag
            clean_term = term.lower().replace(' ', '').replace('-', '')
            if len(clean_term) > 2:
                hashtags.append(f'#{clean_term}')
        
        # Add trending geopolitical hashtags
        trending = [
            '#iran',
            '#china',
            '#oilcrisis',
            '#tradewar',
            '#militaryanalysis',
            '#strategicanalysis'
        ]
        
        hashtags.extend(trending)
        
        # Remove duplicates, keep order
        seen = set()
        unique_hashtags = []
        for tag in hashtags:
            if tag not in seen:
                seen.add(tag)
                unique_hashtags.append(tag)
        
        return unique_hashtags[:30]  # Return top 30
