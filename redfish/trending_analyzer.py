"""
Trending Context Analyzer - Adaptive extraction of trending words/themes from article batch
Analyzes current news cycle to understand what's important (no pre-defined categories)
"""

from typing import List, Dict, Any
from collections import Counter
import re


class TrendingAnalyzer:
    """
    Analyzes a batch of articles to extract trending words/phrases and categorize them.
    Uses TF-IDF-like approach to identify important terms in current news cycle.
    """
    
    def __init__(self):
        self.stop_words = {
            # Standard English stop words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'his',
            'her', 'we', 'our', 'you', 'your', 'said', 'says', 'new', 'also',
            'more', 'most', 'some', 'such', 'than', 'other', 'into', 'over',
            'after', 'before', 'between', 'about', 'which', 'when', 'where',
            'what', 'while', 'there', 'here', 'just', 'only', 'very', 'even',
            'back', 'being', 'still', 'then', 'like', 'many', 'much', 'each',
            'made', 'make', 'come', 'came', 'take', 'took', 'know', 'year',
            'last', 'first', 'time', 'well', 'way',
            # HTML/web artifacts — safety net against leaked markup
            'div', 'span', 'class', 'href', 'https', 'http', 'www', 'com',
            'html', 'img', 'src', 'alt', 'style', 'width', 'height', 'nbsp',
            'amp', 'quot', 'apos', 'link', 'meta', 'body', 'head', 'script',
            'type', 'text', 'content', 'name', 'value', 'data', 'title',
            'font', 'color', 'size', 'border', 'table', 'form', 'input',
            'button', 'label', 'iframe', 'embed', 'object', 'param',
            'target', 'blank', 'self', 'none', 'true', 'false', 'null',
            'undefined', 'function', 'return', 'display', 'inline',
            'block', 'margin', 'padding', 'float', 'clear', 'position',
            'relative', 'absolute', 'overflow', 'hidden', 'visible',
        }
        
        # Category keywords for auto-classification
        self.category_keywords = {
            'military': [
                'military', 'forces', 'troops', 'army', 'navy', 'air force',
                'missile', 'weapon', 'strike', 'attack', 'defense', 'war',
                'combat', 'soldier', 'tank', 'aircraft', 'warship', 'drone',
                'bombing', 'invasion', 'blockade', 'deployment'
            ],
            'economic': [
                'economy', 'economic', 'market', 'price', 'inflation', 'trade',
                'dollar', 'currency', 'stock', 'oil', 'gas', 'energy', 'cost',
                'recession', 'growth', 'gdp', 'debt', 'finance', 'bank',
                'tariff', 'sanction', 'supply chain', 'shortage', 'surge'
            ],
            'diplomatic': [
                'diplomatic', 'diplomacy', 'treaty', 'agreement', 'negotiation',
                'summit', 'talks', 'minister', 'ambassador', 'embassy', 'deal',
                'alliance', 'coalition', 'un', 'nato', 'envoy', 'foreign policy'
            ],
            'geographic': [
                'strait', 'sea', 'ocean', 'gulf', 'border', 'region', 'territory',
                'island', 'peninsula', 'coast', 'port', 'capital', 'city'
            ],
            'human_impact': [
                'civilian', 'people', 'families', 'residents', 'refugees',
                'casualties', 'victims', 'protest', 'demonstration', 'crisis',
                'humanitarian', 'evacuation', 'shelter', 'aid', 'relief'
            ],
            'technology': [
                'technology', 'cyber', 'ai', 'artificial intelligence', 'drone',
                'satellite', 'digital', 'internet', 'hack', 'software', 'chip',
                'semiconductor', 'quantum', 'surveillance'
            ]
        }
    
    def analyze(self, articles: List[Dict[str, Any]], top_n: int = 40) -> Dict[str, Dict[str, Any]]:
        """
        Analyze articles to extract trending words/phrases.
        
        Args:
            articles: List of article dicts with 'title' and 'summary' keys
            top_n: Number of top trending terms to return
            
        Returns:
            Dictionary of trending terms with metadata:
            {
                "term": {
                    "frequency": int,
                    "category": str,
                    "score": float,
                    "contexts": List[str]
                }
            }
        """
        if not articles:
            return {}
        
        # Extract all text
        all_text = []
        for article in articles:
            title = article.get('title', '')
            summary = article.get('summary', '')
            all_text.append(f"{title} {summary}")
        
        combined_text = " ".join(all_text).lower()
        
        # Extract single words
        word_freq = self._extract_word_frequencies(all_text)
        
        # Extract 2-3 word phrases
        phrase_freq = self._extract_phrase_frequencies(all_text)
        
        # Combine and score
        all_terms = {}
        
        # Add words
        for word, freq in word_freq.items():
            if freq >= 3:  # Minimum frequency threshold
                all_terms[word] = {
                    'frequency': freq,
                    'category': self._categorize_term(word),
                    'score': self._calculate_importance_score(word, freq, len(articles)),
                    'contexts': self._extract_contexts(word, all_text[:5])  # Sample contexts
                }
        
        # Add phrases (higher weight)
        for phrase, freq in phrase_freq.items():
            if freq >= 2:  # Lower threshold for phrases
                all_terms[phrase] = {
                    'frequency': freq,
                    'category': self._categorize_term(phrase),
                    'score': self._calculate_importance_score(phrase, freq * 1.5, len(articles)),
                    'contexts': self._extract_contexts(phrase, all_text[:5])
                }
        
        # Sort by score and return top N
        sorted_terms = sorted(
            all_terms.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_n]
        
        return dict(sorted_terms)
    
    def _extract_word_frequencies(self, texts: List[str]) -> Counter:
        """Extract single word frequencies."""
        words = []
        for text in texts:
            text_lower = text.lower()
            # Extract words (alphanumeric + hyphens)
            found_words = re.findall(r'\b[a-z][\w-]*\b', text_lower)
            words.extend([
                w for w in found_words 
                if len(w) > 3 and w not in self.stop_words
            ])
        
        return Counter(words)
    
    def _extract_phrase_frequencies(self, texts: List[str]) -> Counter:
        """Extract 2-3 word phrase frequencies."""
        phrases = []
        for text in texts:
            text_lower = text.lower()
            words = re.findall(r'\b[a-z][\w-]*\b', text_lower)
            
            # 2-word phrases — both words must be meaningful
            for i in range(len(words) - 1):
                if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) > 6:  # Minimum phrase length
                        phrases.append(phrase)
            
            # 3-word phrases — at least 2 of 3 words must be meaningful
            for i in range(len(words) - 2):
                non_stop_count = sum(1 for w in words[i:i+3] if w not in self.stop_words)
                if non_stop_count >= 2:
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if len(phrase) > 10:
                        phrases.append(phrase)
        
        return Counter(phrases)
    
    def _categorize_term(self, term: str) -> str:
        """Categorize a term based on keyword matching."""
        term_lower = term.lower()
        
        # Check each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in term_lower:
                    score += 1
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return category with highest score
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _calculate_importance_score(self, term: str, frequency: float, num_articles: int) -> float:
        """
        Calculate importance score for a term.
        Higher frequency + appearing in multiple articles = higher score.
        """
        # Base score from frequency
        score = frequency / num_articles
        
        # Boost for proper nouns (capitalized)
        if term and term[0].isupper():
            score *= 1.2
        
        # Boost for multi-word phrases
        if ' ' in term:
            score *= 1.3
        
        # Boost for specific high-value categories
        category = self._categorize_term(term)
        if category in ['military', 'economic', 'diplomatic']:
            score *= 1.1
        
        return round(score, 3)
    
    def _extract_contexts(self, term: str, texts: List[str]) -> List[str]:
        """Extract sample contexts where term appears."""
        contexts = []
        for text in texts:
            if term.lower() in text.lower():
                # Find sentence containing term
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if term.lower() in sentence.lower():
                        contexts.append(sentence.strip()[:100])
                        break
            if len(contexts) >= 2:
                break
        return contexts
    
    def get_trending_by_category(self, trending_context: Dict[str, Dict[str, Any]], 
                                  category: str) -> List[str]:
        """
        Get trending terms filtered by category.
        
        Args:
            trending_context: Output from analyze()
            category: Category to filter by
            
        Returns:
            List of trending terms in that category
        """
        return [
            term for term, data in trending_context.items()
            if data['category'] == category
        ]
    
    def is_trending(self, term: str, trending_context: Dict[str, Dict[str, Any]], 
                    threshold: float = 0.3) -> bool:
        """
        Check if a term is trending.
        
        Args:
            term: Term to check
            trending_context: Output from analyze()
            threshold: Minimum score to be considered trending
            
        Returns:
            True if term is trending
        """
        term_lower = term.lower()
        
        # Direct match
        if term_lower in trending_context:
            return trending_context[term_lower]['score'] >= threshold
        
        # Partial match (term contains or is contained in trending term)
        for trending_term, data in trending_context.items():
            if (term_lower in trending_term or trending_term in term_lower):
                if data['score'] >= threshold:
                    return True
        
        return False
    
    def get_boost_score(self, term: str, trending_context: Dict[str, Dict[str, Any]]) -> float:
        """
        Get boost score for a term based on trending context.
        
        Args:
            term: Term to check
            trending_context: Output from analyze()
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        term_lower = term.lower()
        
        # Direct match
        if term_lower in trending_context:
            return min(trending_context[term_lower]['score'], 1.0)
        
        # Partial match - return highest matching score
        max_score = 0.0
        for trending_term, data in trending_context.items():
            if term_lower in trending_term or trending_term in term_lower:
                max_score = max(max_score, data['score'])
        
        return min(max_score, 1.0)
