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
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        import re
        
<<<<<<< HEAD
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON boundaries
=======
        response = response.replace('```json', '').replace('```', '')
        
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            return None
        
        json_str = response[json_start:json_end]
        
<<<<<<< HEAD
        # Remove trailing commas
=======
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
<<<<<<< HEAD
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
=======
        except json.JSONDecodeError:
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
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
                "num_predict": max_tokens
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
                "num_predict": max_tokens
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
<<<<<<< HEAD
        explainer_response: Dict[str, Any],
        salience_data: Optional[Dict[str, Any]] = None,
        historical_parallels: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["script_synthesizer"]
        
        salience_block = ""
        if salience_data:
            salience_block = f"""
Salience Analysis:
- Conflict: {salience_data.get('conflict', 'N/A')}
- Consequence Chain: {' -> '.join(salience_data.get('consequence_chain', []))}
- Emotional Anchors: {', '.join(salience_data.get('emotional_anchors', []))}
- Surprise Angle: {salience_data.get('surprise_angle', 'N/A')}
- Human Impact: {salience_data.get('human_impact', 'N/A')}
- Key Visual Subjects: {', '.join(salience_data.get('key_visual_subjects', []))}
"""
        
        historical_block = ""
        if historical_parallels and 'parallels' in historical_parallels:
            historical_block = "\nHistorical Parallels to Reference:\n"
            for i, parallel in enumerate(historical_parallels['parallels'][:3], 1):
                equipment_str = ', '.join(parallel.get('military_equipment', [])[:3])
                historical_block += f"""
{i}. {parallel.get('event_name', 'N/A')} ({parallel.get('year', 'N/A')}):
   - Players: {', '.join(parallel.get('key_players', []))}
   - Equipment: {equipment_str}
   - Outcome: {parallel.get('outcome', 'N/A')}
   - Relevance: {parallel.get('relevance_to_current', 'N/A')}
"""
            historical_block += f"\nHistorical Pattern: {historical_parallels.get('historical_pattern', 'N/A')}"
            historical_block += f"\nKey Difference in 2026: {historical_parallels.get('key_difference_2026', 'N/A')}\n"
        
        prompt = f"""Create a 60-80 second news narration script with historical anchoring from this analysis:
=======
        explainer_response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["script_synthesizer"]
        
        prompt = f"""Create a viral short-form video script from this debate:
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)

Topic: {news_summary.get('topic', '')}
Key Facts: {', '.join(news_summary.get('key_facts', []))}
Angle: {news_summary.get('angle', '')}
<<<<<<< HEAD
{salience_block}{historical_block}
=======

>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
Skeptic's Critique: {skeptic_response.get('critique', '')}
Skeptic's Question: {skeptic_response.get('key_question', '')}

Explainer's Response: {explainer_response.get('explanation', '')}
Explainer's Analogy: {explainer_response.get('analogy', '')}

<<<<<<< HEAD
CRITICAL: Output ONLY the JSON object. NO explanatory text before or after. NO markdown. Start with {{ and end with }}.

Synthesize into a compelling 60-80 second professional news narration script with 6 segments that weaves historical parallels into the narrative."""
=======
Synthesize this into a compelling 45-second script."""
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
        
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            return None
        
        script = self._extract_json(response)
        
        # TEMP: Try relaxed parsing if strict fails
        if not script or not isinstance(script, dict):
            try:
                import json
                cleaned = response.strip().replace('\n', ' ').replace('\r', '')
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = cleaned[start:end]
                    script = json.loads(json_str)
                    print("  [TEMP] Parsed script JSON with relaxed rules")
            except:
                pass
        
        if not script or not isinstance(script, dict):
            print(f"Failed to parse script JSON or invalid format")
<<<<<<< HEAD
            print(f"Raw response: {response[:1000]}")
            print(f"Response length: {len(response)} chars")
=======
            print(f"Raw response: {response[:300]}")
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
            return None
        
        if "word_count" not in script:
            def count_words(field):
                value = script.get(field, "")
                if isinstance(value, str):
                    return len(value.split())
                elif isinstance(value, list):
                    return sum(len(str(item).split()) for item in value)
                return 0
            
<<<<<<< HEAD
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
            
=======
            total_words = count_words("hook") + count_words("body") + \
                         count_words("twist") + count_words("cta")
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
            script["word_count"] = total_words
            script["estimated_duration"] = int(total_words / 2.5)
        return script
    
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
<<<<<<< HEAD
    
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
=======
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
