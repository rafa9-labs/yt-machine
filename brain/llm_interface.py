import json
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

class LLMInterface:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "system_prompts.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_url = self.config["model_config"]["base_url"]
        self.default_model = self.config["model_config"]["default_model"]
        self.timeout = self.config["model_config"]["timeout"]
        self.retry_attempts = self.config["model_config"]["retry_attempts"]
        self.num_ctx = self.config["model_config"].get("num_ctx", 4096)
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        import re
        
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON boundaries
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            return None
        
        json_str = response[json_start:json_end]
        
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to extract just the outermost JSON object
            try:
                # Count braces to find complete JSON
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(response):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            json_str = response[start_idx:i+1]
                            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                            return json.loads(json_str)
            except:
                pass
            
            # Last resort: Try to close incomplete JSON
            try:
                json_str = response[json_start:]
                # Count open braces/brackets
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                # Add closing characters
                json_str += '}' * open_braces
                json_str += ']' * open_brackets
                
                # Remove trailing commas before closing
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                return json.loads(json_str)
            except:
                pass
            
            return None
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], attempt: int = 1) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(
                url,
                json=payload,
                timeout=(10, self.timeout),
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        full_response += chunk["response"]
                    if chunk.get("done", False):
                        break
            
            return {"response": full_response.strip()}
        
        except requests.exceptions.RequestException as e:
            if attempt < self.retry_attempts:
                print(f"Request failed (attempt {attempt}/{self.retry_attempts}): {e}")
                time.sleep(2 ** attempt)
                return self._make_request(endpoint, payload, attempt + 1)
            else:
                print(f"Request failed after {self.retry_attempts} attempts: {e}")
                return None
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        if model is None:
            model = self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        result = self._make_request("/api/generate", payload)
        return result["response"] if result else None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        if model is None:
            model = self.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx
            }
        }
        
        result = self._make_request("/api/chat", payload)
        return result["response"] if result else None
    
    def process_news(self, article_text: str) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["news_processor"]
        
        prompt = f"Analyze this news article and extract viral-worthy information:\n\n{article_text}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        result = self._extract_json(response)
        if not result:
            print(f"Failed to parse JSON response")
            print(f"Raw response: {response[:300]}")
        return result
    
    def debate_skeptic(self, news_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["debate_skeptic"]
        
        prompt = f"Critique this news story:\n\nTopic: {news_summary.get('topic', '')}\nKey Facts: {', '.join(news_summary.get('key_facts', []))}\nAngle: {news_summary.get('angle', '')}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        result = self._extract_json(response)
        if not result:
            return {"critique": response, "key_question": ""}
        return result
    
    def debate_explainer(self, news_summary: Dict[str, Any], skeptic_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["debate_explainer"]
        
        prompt = f"Respond to this critique:\n\nTopic: {news_summary.get('topic', '')}\nSkeptic's Critique: {skeptic_response.get('critique', '')}\nKey Question: {skeptic_response.get('key_question', '')}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        result = self._extract_json(response)
        if not result:
            return {"explanation": response, "analogy": ""}
        return result
    
    def synthesize_script(
        self,
        news_summary: Dict[str, Any],
        skeptic_response: Dict[str, Any],
        explainer_response: Dict[str, Any],
        salience_data: Optional[Dict[str, Any]] = None,
        historical_parallels: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["script_synthesizer"]
        
        # 6-segment structure names for recovery
        _SEGMENT_NAMES_6 = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
        _SEGMENT_NAMES_5 = ['hook', 'context', 'escalation', 'consequence', 'twist']

        salience_block = ""
        if salience_data:
            # Compress salience to top facts if it would bloat the prompt
            consequence_chain = salience_data.get('consequence_chain') or []
            emotional_anchors = (salience_data.get('emotional_anchors') or [])[:2]
            key_visual_subjects = (salience_data.get('key_visual_subjects') or [])[:3]
            salience_block = f"""
Salience Analysis:
- Conflict: {salience_data.get('conflict', 'N/A')}
- Consequence Chain: {' -> '.join(consequence_chain[:2])}
- Emotional Anchors: {', '.join(emotional_anchors)}
- Surprise Angle: {salience_data.get('surprise_angle', 'N/A')}
- Human Impact: {salience_data.get('human_impact', 'N/A')}
- Key Visual Subjects: {', '.join(key_visual_subjects)}
"""
        
        historical_block = ""
        if historical_parallels and 'parallels' in historical_parallels:
            historical_block = "\nHistorical Parallels to Reference:\n"
            # Limit to top 2 parallels and compress fields to reduce token load
            for i, parallel in enumerate(historical_parallels['parallels'][:2], 1):
                equipment_str = ', '.join((parallel.get('military_equipment') or [])[:2])
                historical_block += f"""
{i}. {parallel.get('event_name', 'N/A')} ({parallel.get('year', 'N/A')}):
   - Players: {', '.join((parallel.get('key_players') or [])[:3])}
   - Equipment: {equipment_str}
   - Outcome: {parallel.get('outcome', 'N/A')}
"""
            historical_block += f"\nHistorical Pattern: {historical_parallels.get('historical_pattern', 'N/A')}"
            historical_block += f"\nKey Difference in 2026: {historical_parallels.get('key_difference_2026', 'N/A')}\n"
        
        prompt = f"""Create a 60-80 second news narration script with historical anchoring from this analysis:

Topic: {news_summary.get('topic', '')}
Key Facts: {', '.join(news_summary.get('key_facts', []))}
Angle: {news_summary.get('angle', '')}
{salience_block}{historical_block}
Skeptic's Critique: {skeptic_response.get('critique', '')}
Skeptic's Question: {skeptic_response.get('key_question', '')}

Explainer's Response: {explainer_response.get('explanation', '')}
Explainer's Analogy: {explainer_response.get('analogy', '')}

CRITICAL: Output ONLY the JSON object. NO explanatory text before or after. NO markdown. Start with {{ and end with }}.

Synthesize into a compelling 60-80 second professional news narration script with 6 segments that weaves historical parallels into the narrative."""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        script = self._extract_json(response)
        if not script or not isinstance(script, dict):
            print(f"Failed to parse script JSON or invalid format")
            print(f"Raw response: {response[:1000]}")
            print(f"Response length: {len(response)} chars")
            return None
        
        # Missing-segment recovery: detect truncation and request missing segments
        expected_names = _SEGMENT_NAMES_6 if 'historical_1' in script else _SEGMENT_NAMES_5
        missing_segments = [s for s in expected_names if not script.get(s)]
        if missing_segments:
            print(f"  [RECOVERY] Script truncated — missing segments: {missing_segments}")
            recovery_prompt = (
                f"The previous script JSON was truncated. It has these segments: "
                f"{[s for s in expected_names if script.get(s)]}. "
                f"Output ONLY a JSON object with these missing segments filled in, "
                f"continuing the same narrative about: {news_summary.get('topic', 'geopolitical event')}. "
                f"Missing keys: {missing_segments}. Each value should be 1-2 sentences of narration. "
                f"Start with {{ and end with }}. Nothing else."
            )
            recovery_response = self.generate(
                prompt=recovery_prompt,
                temperature=0.7,
                max_tokens=1200
            )
            if recovery_response:
                recovery_data = self._extract_json(recovery_response)
                if recovery_data and isinstance(recovery_data, dict):
                    for seg in missing_segments:
                        if seg in recovery_data:
                            script[seg] = recovery_data[seg]
                            print(f"    [RECOVERY] Restored: {seg}")
                    remaining = [s for s in expected_names if not script.get(s)]
                    if not remaining:
                        print(f"  [RECOVERY] All segments restored successfully")
                    else:
                        print(f"  [RECOVERY] Still missing: {remaining} — using fallback text")
                        for seg in remaining:
                            script[seg] = f"The situation continues to develop with significant implications for regional stability."
            else:
                print(f"  [RECOVERY] Recovery call failed — using fallback text for missing segments")
                for seg in missing_segments:
                    script[seg] = f"The situation continues to develop with significant implications for regional stability."
        
        # CRITICAL FIX: Build full_text from segments if missing or incomplete
        # This ensures TTS gets the complete narration, not just the title/hook
        if 'full_text' not in script or not script['full_text'] or len(script['full_text'].split()) < 50:
            segments = []
            # Support both 6-segment and 5-segment structures
            if 'historical_1' in script:
                segment_names = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
            else:
                segment_names = ['hook', 'context', 'escalation', 'consequence', 'twist']
            
            for seg in segment_names:
                segment_data = script.get(seg, '')
                if segment_data:
                    # Handle both string and dict formats
                    if isinstance(segment_data, dict):
                        # If it's a dict, try to extract text content
                        text = segment_data.get('text', segment_data.get('content', str(segment_data)))
                    else:
                        text = str(segment_data)
                    
                    if text and text.strip():
                        segments.append(text)
            
            script['full_text'] = ' '.join(segments)
            print(f"  [SCRIPT] Built full_text from {len(segments)} segments ({len(script['full_text'].split())} words)")
        
        if "word_count" not in script:
            def count_words(field):
                value = script.get(field, "")
                if isinstance(value, str):
                    return len(value.split())
                elif isinstance(value, list):
                    return sum(len(str(item).split()) for item in value)
                return 0
            
            # Support both 5-segment and 6-segment structures
            if "historical_1" in script:
                # 6-segment structure
                total_words = count_words("hook") + count_words("historical_1") + \
                             count_words("historical_2") + count_words("modern_pivot") + \
                             count_words("consequence") + count_words("future_outlook")
            else:
                # 5-segment structure (fallback)
                total_words = count_words("hook") + count_words("context") + \
                             count_words("escalation") + count_words("consequence") + \
                             count_words("twist")
            
            script["word_count"] = total_words
            script["estimated_duration"] = int(total_words / 2.5)
        return script
    
    def _get_time_greeting(self) -> str:
        """Get time-of-day greeting for Masker personality — short & punchy."""
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            return "Good Morning! I'm Masker!"
        elif hour < 18:
            return "Good Afternoon! I'm Masker!"
        else:
            return "Good Evening! I'm Masker!"
    
    def synthesize_multi_news_script(
        self,
        news_analyses: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a 3-news Masker personality script.
        Each news_analysis dict should have: topic, key_facts, angle, impact_score.
        """
        prompt_config = self.config["prompts"]["multi_news_synthesizer"]
        
        greeting = self._get_time_greeting()
        
        # Build news summaries block
        news_block = ""
        for i, analysis in enumerate(news_analyses, 1):
            news_block += f"""
NEWS STORY {i}:
- Topic: {analysis.get('topic', 'N/A')}
- Key Facts: {', '.join(analysis.get('key_facts', []))}
- Angle: {analysis.get('angle', 'N/A')}
- Impact Score: {analysis.get('impact_score', 5)}/10
- Second-order consequence: {analysis.get('second_order_consequence', 'N/A')}
"""
        
        prompt = f"""Create a fun, sassy 3-news script for Masker the news host.

GREETING TO USE: "{greeting}"

{news_block}

CRITICAL: Output ONLY the JSON object. NO explanatory text before or after. NO markdown. Start with {{ and end with }}.

IMPORTANT: The greeting field should be EXACTLY: {greeting}
The full_text must include the greeting, all three stories with transitions, and the closing as one continuous narration paragraph.

Remember: Be witty, sassy, but accurate. Simplify complex geopolitics so anyone can understand it.
Target: 180-250 words total for 75-90 seconds."""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        script = self._extract_json(response)
        if not script or not isinstance(script, dict):
            print(f"Failed to parse multi-news script JSON")
            print(f"Raw response: {response[:500]}")
            return None
        
        # Ensure greeting is set correctly
        script['greeting'] = greeting
        
        # Ensure intro_hook exists
        if not script.get('intro_hook'):
            script['intro_hook'] = "Three stories today — and trust me, you'll want to hear the last one."
        
        # Ensure stories exist
        if 'stories' not in script or not script.get('stories'):
            print(f"  [MULTI-NEWS] No stories in response — falling back")
            return None
        
        # Build full_text if missing or incomplete
        if not script.get('full_text') or len(script.get('full_text', '').split()) < 30:
            parts = [greeting, script.get('intro_hook', '')]
            for story in script['stories']:
                parts.append(story.get('mini_hook', ''))
                parts.append(story.get('body', ''))
                punchline = story.get('punchline', '')
                if punchline:
                    parts.append(punchline)
                transition = story.get('transition', '')
                if transition:
                    parts.append(transition)
            parts.append(script.get('closing', 
                "And with that we conclude the news for today. Subscribe, like, do what you gotta do — I was Masker and see you tomorrow!"))
            script['full_text'] = ' '.join(filter(None, parts))
        
        # Calculate accurate word count and duration
        script['word_count'] = len(script['full_text'].split())
        script['estimated_duration'] = int(script['word_count'] / 2.5)
        
        # Extract all visual scenes into flat list for image generation
        all_visual_scenes = []
        for story in script['stories']:
            for scene in story.get('visual_scenes', []):
                all_visual_scenes.append(scene)
        script['all_visual_scenes'] = all_visual_scenes
        
        print(f"  [MULTI-NEWS] Script: {len(script['stories'])} stories, {script['word_count']} words, ~{script['estimated_duration']}s")
        
        return script
    
    def curate_script(self, full_text: str) -> Optional[str]:
        """
        Second-pass LLM curation: transforms written script into natural spoken language.
        Optimizes rhythm, pauses, emphasis, pacing — without changing any facts.
        """
        prompt_config = self.config["prompts"]["script_curator"]
        
        prompt = f"""Transform this news script from written text into natural, human-sounding spoken language.

RULES:
- NEVER change facts, numbers, or country names
- NEVER add or remove information
- Break long sentences into short punchy ones
- Use '...' for dramatic pauses
- Use '—' for abrupt contrasts
- Move key numbers to end of sentences (punch position)
- Use contractions ALWAYS (it's, they're, won't)
- Create rhythm: alternate short punchy + longer explanatory sentences
- Balance all 3 stories to roughly equal word count (40-55 words each)

ORIGINAL SCRIPT:
{full_text}

Output ONLY the curated spoken script as plain text. No JSON. No explanations."""

        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            print(f"  [CURATOR] Curation failed, using original script")
            return full_text
        
        # Clean any accidental markdown wrapping
        curated = response.strip()
        if curated.startswith('```'):
            curated = curated.split('\n', 1)[-1]
        if curated.endswith('```'):
            curated = curated.rsplit('```', 1)[0]
        curated = curated.strip()
        
        word_count_original = len(full_text.split())
        word_count_curated = len(curated.split())
        
        # Sanity check: curated script should be similar length (±30%)
        if word_count_curated < word_count_original * 0.5:
            print(f"  [CURATOR] Curated script too short ({word_count_curated} vs {word_count_original}), using original")
            return full_text
        
        print(f"  [CURATOR] Script curated: {word_count_original} → {word_count_curated} words")
        return curated
    
    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def warmup_model(self, model: str = None) -> bool:
        if model is None:
            model = self.default_model
        
        print(f"Warming up model '{model}'... (this may take 30-60 seconds)")
        try:
            response = self.generate(
                prompt="Hello",
                model=model,
                temperature=0.1,
                max_tokens=10
            )
            return response is not None
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False
    
    def extract_visual_elements(self, article_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract visual elements for image generation
        
        Args:
            article_text: Full article text
            
        Returns:
            Dictionary with extracted visual elements
        """
        extraction_prompt = f"""Extract visual elements from this news article for pixel art image generation.
Extract ALL relevant visual subjects — military, economic, diplomatic, civilian, geographic.

{article_text}

Output ONLY valid JSON with these exact keys:
{{
  "primary_subjects": ["specific visual subjects: oil tankers, fighter jets, gas station price boards, diplomatic meetings, shipping ports, civilian crowds, military convoys, trading floors, protest marches, etc."],
  "settings": ["real geographic locations and settings from the article"],
  "actions": ["dynamic action verbs: surging, deploying, signing, queuing, collapsing, launching, etc."],
  "mood": "tense/hopeful/chaotic/urgent/calm",
  "temporal_context": "time of day or weather if mentioned, otherwise empty string"
}}

Be specific. Extract only what is explicitly mentioned or directly implied by the article.
Include BOTH military AND non-military subjects."""

        response = self.generate(
            prompt=extraction_prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        if not response:
            return None
        
        result = self._extract_json(response)
        return result if result else {
            'primary_subjects': [],
            'settings': [],
            'actions': [],
            'mood': 'tense',
            'temporal_context': ''
        }
