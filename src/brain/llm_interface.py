import json
import os
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

class LLMInterface:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "system_prompts.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_url = self.config["model_config"]["base_url"]
        self.default_model = self.config["model_config"]["default_model"]
        self.fallback_model = self.config["model_config"].get("fallback_model", "llama3.2:latest")
        self.timeout = self.config["model_config"]["timeout"]
        self.retry_attempts = self.config["model_config"]["retry_attempts"]
        self.num_ctx = self.config["model_config"].get("num_ctx", 4096)
        
        # Task-specific model routing
        self.task_models = self.config["model_config"].get("task_models", {})
        
        # Circuit breaker for model fallback
        self._primary_failures = 0
        self._MAX_PRIMARY_FAILURES = 3
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def _strip_thinking_tokens(text: str) -> str:
        """Strip thinking tokens from LLM response.

        Handles three model families:
          Qwen3: <think...</think
          Gemma 4 Heretic: <|channel>thought<channel|>...<|channel>output<channel|>
          DeepSeek-R1: <think...</think
        These consume output budget and break JSON parsing.
        """
        import re

        # Qwen3 / DeepSeek-R1: <think...</think
        # Some Qwen3 outputs use <think without closing >, or </think without >
        think_match = re.search(r'<think\b', text)
        if think_match:
            close_match = re.search(r'</think\s*>?', text)
            if close_match:
                text = text[close_match.end():]
            else:
                json_start = re.search(r'[{]', text[think_match.start():])
                if json_start:
                    text = text[think_match.start() + json_start.start():]

        # Gemma 4 Heretic: <|channel>output<channel|>
        output_match = re.search(r'<\|?channel\|?>output<\|?channel\|?>', text)
        if output_match:
            text = text[output_match.end():]

        # If nothing matched but text starts with non-JSON, find first {
        if not text.strip().startswith('{') and not text.strip().startswith('['):
            json_start = re.search(r'[{]', text)
            if json_start:
                text = text[json_start.start():]

        # Clean remaining stray special tokens
        text = re.sub(r'</?think[^>]*>?', '', text)
        text = re.sub(r'<\|?[^>]*\|?>', '', text)
        return text.strip()

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        import re

        # Strip thinking tokens from abliterated/heretic models
        response = self._strip_thinking_tokens(response)

        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        json_start = response.find('{')
        if json_start == -1:
            return None

        json_str = response[json_start:]

        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Try brace-counting to extract a complete JSON object
        try:
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
                        candidate = response[start_idx:i+1]
                        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
                        return json.loads(candidate)
        except:
            pass

        # Last resort: auto-close incomplete JSON by counting unclosed delimiters
        try:
            json_str = response[json_start:]
            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')

            json_str += ']' * max(0, open_brackets)
            json_str += '}' * max(0, open_braces)

            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            return json.loads(json_str)
        except:
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
    
    # Circuit breaker state
    _primary_failures = 0
    _MAX_PRIMARY_FAILURES = 3  # After this many consecutive failures, skip straight to fallback
    
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
        
        # If primary model has failed too many times, go straight to fallback
        if model == self.default_model and self._primary_failures >= self._MAX_PRIMARY_FAILURES:
            print(f"  [LLM] ⚠️ Primary model circuit breaker active ({self._primary_failures} failures), using fallback: {self.fallback_model}")
            payload["model"] = self.fallback_model
        
        result = self._make_request("/api/generate", payload)
        
        if result and result.get("response", "").strip():
            # Reset circuit breaker on success
            if payload["model"] == self.default_model:
                self._primary_failures = 0
            return result["response"]
        
        if result and not result.get("response", "").strip():
            print(f"  [LLM] Model {payload['model']} returned empty response (prompt may be too long)")
        
        # Model failed — try fallback chain
        # Build fallback chain: if task model → default model → global fallback → llama3.2
        attempted_models = {payload["model"]}
        fallback_chain = []
        
        if model != self.default_model and self.default_model not in attempted_models:
            fallback_chain.append(self.default_model)
        if self.fallback_model and self.fallback_model not in attempted_models:
            fallback_chain.append(self.fallback_model)
        # Last resort
        if "llama3.2:latest" not in attempted_models and "llama3.2:latest" not in fallback_chain:
            fallback_chain.append("llama3.2:latest")
        
        for fallback in fallback_chain:
            print(f"  [LLM] Model {payload['model']} failed, trying fallback: {fallback}")
            payload["model"] = fallback
            result = self._make_request("/api/generate", payload)
            
            if result and result.get("response", "").strip():
                print(f"  [LLM] ✅ Fallback model ({fallback}) succeeded ({len(result['response'])} chars)")
                return result["response"]
        
        print(f"  [LLM] ❌ All models in fallback chain failed")
        
        return None
    
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
        
        # ── OLLAMA ONLY: News analysis often contains sensitive geopolitical content
        # that triggers GLM-5's content filter (1301). No cloud fallback.
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            print("  [NEWS] Local model unavailable — no cloud fallback for sensitive content")
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
        if 'full_text' not in script or not script['full_text'] or len(script['full_text'].split()) < 50:
            segments = []
            if 'historical_1' in script:
                segment_names = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
            else:
                segment_names = ['hook', 'context', 'escalation', 'consequence', 'twist']
            
            for seg in segment_names:
                segment_data = script.get(seg, '')
                if segment_data:
                    if isinstance(segment_data, dict):
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
            
            if "historical_1" in script:
                total_words = count_words("hook") + count_words("historical_1") + \
                             count_words("historical_2") + count_words("modern_pivot") + \
                             count_words("consequence") + count_words("future_outlook")
            else:
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
    
    # ── SEGUE TEMPLATES: High-quality fallbacks when LLM generates weak segues ──
    _SEGUE_TEMPLATES = [
        "But wait... that's not even the craziest part.",
        "Oh but we're just getting started.",
        "And now for something that'll make your jaw drop.",
        "You think that's wild? Just wait for this next one.",
        "Hold onto your hats... this next one is absolutely insane.",
        "And believe it or not, it gets even crazier.",
    ]
    
    def _ensure_greeting_in_fulltext(self, script: dict) -> dict:
        """
        GUARANTEE: full_text MUST start with the greeting.
        This is called after every full_text modification (synthesis, curation, etc.)
        """
        full_text = script.get('full_text', '')
        greeting = script.get('greeting', '')
        intro_hook = script.get('intro_hook', '')
        
        if not full_text or not greeting:
            return script
        
        # Check if full_text already starts with the greeting
        if full_text.strip().startswith(greeting):
            return script  # All good
        
        # Check if full_text starts with a partial greeting (e.g., just "Look,")
        # In that case, prepend the full greeting
        if intro_hook and full_text.strip().startswith(intro_hook[:10]):
            # full_text starts with intro_hook but missing greeting
            script['full_text'] = f"{greeting} {full_text.strip()}"
            print(f"  [GREETING] Prepended missing greeting to full_text")
        elif greeting.split()[0].lower() not in full_text[:30].lower():
            # Neither greeting nor intro_hook at start — prepend both
            prefix = f"{greeting} {intro_hook}".strip()
            script['full_text'] = f"{prefix} {full_text.strip()}"
            print(f"  [GREETING] Prepended greeting + intro_hook to full_text")
        
        return script
    
    def _enforce_segues(self, script: dict) -> dict:
        """
        GUARANTEE: Every non-last story MUST have a non-empty segue (8-15 words).
        If the LLM generated a weak/empty segue, inject a template one.
        """
        import random
        stories = script.get('stories', [])
        if len(stories) < 2:
            return script
        
        for i, story in enumerate(stories):
            if i >= len(stories) - 1:
                # Last story — segue must be empty
                story['segue'] = ''
                continue
            
            segue = story.get('segue', '').strip()
            
            # Check if segue is valid: non-empty, 5+ words, creates anticipation
            words = segue.split()
            is_valid = len(words) >= 5 and any(
                kw in segue.lower() for kw in ['but', 'and', 'now', 'wait', 'that', 'here', 'check', 'sneaky', 'wild', 'crazy', 'insane', 'believe', 'next', 'last', "won't believe"]
            )
            
            if not is_valid:
                # Pick a template (rotated based on story index to avoid repetition)
                template_idx = (i + hash(str(story.get('part_1_narration', '')))) % len(self._SEGUE_TEMPLATES)
                new_segue = self._SEGUE_TEMPLATES[template_idx]
                story['segue'] = new_segue
                print(f"  [SEGUE] Story {i+1} segue was weak/empty, injected: \"{new_segue}\"")
        
        return script
    
    # Unified closing — the signature Masker farewell (Truman Show inspired)
    UNIFIED_CLOSING = "Stay behind the curtain. Subscribe, like. And if I don't see you. Good morning. Good afternoon. And good night."
    
    def _validate_closing(self, full_text: str) -> str:
        """
        Ensure the script ends with the UNIFIED Masker closing/CTA.
        ALWAYS replaces or appends the canonical closing — never trusts LLM output.
        """
        if not full_text:
            return full_text
        
        text_lower = full_text.lower()
        has_truman = 'good morning' in text_lower and 'good afternoon' in text_lower
        
        if has_truman:
            return full_text
        
        # Strip any existing LLM-generated closing to avoid duplication
        import re
        stripped = re.sub(
            r'\s*\.{3,4}\s*(?:And with that|Subscribe|That\'s all|So there you have|This is Masker|I\'m Masker|see you|And these were|These were|Stay tuned|See you next|Thanks for|follow for|Stay behind|The walls).*$',
            '', full_text, flags=re.IGNORECASE
        ).rstrip()
        
        # Second pass: strip any trailing subscribe/like/CTA patterns
        stripped = re.sub(
            r'\s*(?:Subscribe|subscribe)[,.\s]*(?:like|and\s+like|hit\s+like)?[,.\s]*(?:share|comment|follow)?\s*.*$',
            '', stripped, flags=re.IGNORECASE
        ).rstrip()
        
        result = stripped + ' .... ' + self.UNIFIED_CLOSING
        print(f"  [CLOSING] Injected unified closing (truman={has_truman})")
        return result
    
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
        
        prompt = f"""Create a Masker script — 3 stories, theatrical reveal structure.

GREETING TO USE: "{greeting}"

{news_block}

CRITICAL RULES:
- Output ONLY the JSON object. NO explanatory text before or after. NO markdown.
- The greeting field must be EXACTLY: {greeting}
- Each story: part_1 = THE EFFECT (jaw-drop), part_2 = THE MECHANISM (hidden hand revealed)
- Use theater metaphors: "shadow play", "behind the curtain", "the board", "backstage"
- Make the viewer feel like they just saw behind the curtain of a magic show
- Playful mischief energy — cheeky, clever, theatrical. NOT bleak or cynical.
- Target: 220-280 words total for ~90 seconds.
- ALL 3 stories must be roughly equal word count (50-60 words each). Max 10 words difference.
- Each non-last story must have a "segue" field with a curiosity-gap transition (8-15 words).
- NEVER include subscribe, like, sign-off, closing, or CTA text in any field."""
        
        # ── OPENAI CLOUD FIRST, LOCAL FALLBACK ──
        response = None

        # Try 1: gpt-5-mini via OpenAI (cheap, high quality)
        openai_response = self._call_openai(
            system_prompt=prompt_config["system_prompt"],
            user_prompt=prompt,
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"],
            purpose="script_synthesis"
        )
        if openai_response:
            response = openai_response
            print(f"  [MULTI-NEWS] OpenAI gpt-5-mini succeeded")

        # Try 2: Local model fallback
        if not response:
            task_model = self.task_models.get("multi_news_synthesizer", self.default_model)
            print(f"  [MULTI-NEWS] OpenAI unavailable, using local model: {task_model}")
            response = self.generate(
                prompt=prompt,
                model=task_model,
                system_prompt=prompt_config["system_prompt"],
                temperature=prompt_config["temperature"],
                max_tokens=prompt_config["max_tokens"]
            )
        
        if not response:
            print("  [MULTI-NEWS] Local model unavailable — no cloud fallback for sensitive content")
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
        
        # ── Build segment timeline from part_1/part_2 format ──
        # Each segment maps to an image: [segment_text, image_index]
        segment_timeline = []
        
        # Intro segment → image 0 (first story, part 1)
        intro_text = f"{greeting} {script.get('intro_hook', '')}".strip()
        segment_timeline.append({
            'text': intro_text,
            'image_idx': 0,
            'label': 'intro'
        })
        
        # PAUSE after intro — let the hook land before diving into stories
        segment_timeline.append({
            'text': '...',
            'image_idx': 0,
            'label': 'intro_pause',
            'is_separator': True
        })
        
        for i, story in enumerate(script['stories']):
            img_base = i * 2  # Story 0 → images 0,1; Story 1 → images 2,3; Story 2 → images 4,5
            
            # Part 1 narration → image (img_base)
            part_1 = story.get('part_1_narration', '')
            if part_1:
                segment_timeline.append({
                    'text': part_1,
                    'image_idx': img_base,
                    'label': f'story_{i+1}_part1'
                })
            
            # Part 2 narration → image (img_base + 1)
            part_2 = story.get('part_2_narration', '')
            if part_2:
                segment_timeline.append({
                    'text': part_2,
                    'image_idx': img_base + 1,
                    'label': f'story_{i+1}_part2'
                })
            
            # SEGUE → witty transition to next story (keep same image as part 2)
            segue = story.get('segue', story.get('transition', ''))
            if segue and i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': segue,
                    'image_idx': img_base + 1,
                    'label': f'story_{i+1}_segue'
                })
            
            # Story separator (....) except after last story
            if i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': '....',
                    'image_idx': img_base + 1,
                    'label': f'story_{i+1}_separator',
                    'is_separator': True
                })
        
        # PAUSE after last story — let the final punchline land before closing
        last_story_idx = len(script['stories']) - 1
        segment_timeline.append({
            'text': '...',
            'image_idx': last_story_idx * 2 + 1,
            'label': 'pre_closing_pause',
            'is_separator': True
        })
        
        # Closing → ALWAYS use unified closing (never trust LLM output for this)
        closing = self.UNIFIED_CLOSING
        segment_timeline.append({
            'text': closing,
            'image_idx': (len(script['stories']) - 1) * 2 + 1,
            'label': 'closing'
        })
        
        # Build full_text from timeline (includes segues and separators)
        full_parts = []
        for seg in segment_timeline:
            full_parts.append(seg['text'])
        full_text = ' '.join(filter(None, full_parts))
        
        script['full_text'] = full_text
        script['segment_timeline'] = segment_timeline
        
        # Extract visual prompts from stories (part_1_visual, part_2_visual)
        visual_prompts = []
        for i, story in enumerate(script['stories']):
            visual_prompts.append({
                'scene': f'story_{i+1}_part1',
                'description': story.get('part_1_visual', story.get('mini_hook', ''))
            })
            visual_prompts.append({
                'scene': f'story_{i+1}_part2',
                'description': story.get('part_2_visual', story.get('body', ''))
            })
        script['all_visual_scenes'] = visual_prompts
        
        # ENFORCE SEGUES: Guarantee every non-last story has a strong segue
        script = self._enforce_segues(script)
        
        # Rebuild segment_timeline after segue enforcement
        # (segues may have changed)
        for seg in segment_timeline:
            if 'segue' in seg['label']:
                story_idx = int(seg['label'].split('_')[1]) - 1
                if story_idx < len(script['stories']):
                    seg['text'] = script['stories'][story_idx].get('segue', seg['text'])
        
        # Rebuild full_text with enforced segues
        full_parts = [seg['text'] for seg in segment_timeline]
        full_text = ' '.join(filter(None, full_parts))
        script['full_text'] = full_text
        script['segment_timeline'] = segment_timeline
        
        # GUARANTEE GREETING: full_text MUST start with greeting
        script = self._ensure_greeting_in_fulltext(script)
        
        # VALIDATE CLOSING: Ensure full_text ends with subscribe/CTA
        script['full_text'] = self._validate_closing(script['full_text'])
        
        # Calculate accurate word count and duration
        script['word_count'] = len(script['full_text'].split())
        script['estimated_duration'] = int(script['word_count'] / 2.5)
        
        print(f"  [MULTI-NEWS] Script: {len(script['stories'])} stories, {script['word_count']} words, ~{script['estimated_duration']}s")
        print(f"  [MULTI-NEWS] Timeline: {len(segment_timeline)} segments → 6 images")
        for seg in segment_timeline:
            print(f"    [{seg['label']}] → img#{seg['image_idx']}: \"{seg['text'][:50]}...\"")
        
        return script
    
    def _extract_key_entities(self, text: str) -> set:
        """Extract key proper nouns, country names, and numbers from text."""
        import re
        
        # Capitalized multi-word entities (country names, proper nouns, org names)
        entities = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text))
        
        # Common English words to filter out
        stopwords = {
            'The', 'This', 'That', 'And', 'But', 'For', 'Not', 'In', 'On', 'At',
            'With', 'It', 'Is', 'Was', 'Are', 'Were', 'Has', 'Have', 'Had', 'Been',
            'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'They', 'Their',
            'There', 'These', 'Those', 'Each', 'Every', 'Which', 'What', 'When',
            'Where', 'Who', 'How', 'Why', 'More', 'Most', 'Some', 'Such', 'Than',
            'Then', 'Now', 'Just', 'Also', 'Very', 'Even', 'Still', 'Only', 'About',
            'After', 'Before', 'Between', 'Through', 'During', 'Without', 'Against',
            'Another', 'While', 'Last', 'First', 'Next', 'Both', 'All', 'Many',
            'Much', 'Own', 'Other', 'New', 'Old', 'Good', 'Great', 'Big', 'Small',
            'So', 'If', 'Or', 'An', 'No', 'Do', 'Did', 'Get', 'Got', 'Make',
            'Made', 'Like', 'Well', 'Back', 'Over', 'Into', 'Right', 'Because',
            'Since', 'Being', 'Having', 'Doing', 'Going', 'Coming', 'Taking',
            'Tonight', 'Today', 'Yesterday', 'Tomorrow', 'Subscribe', 'Masker',
            'Afternoon', 'Morning', 'Evening', 'I', 'You', 'We', 'He', 'She',
        }
        entities -= stopwords
        
        # Important numbers
        numbers = set(re.findall(r'\b\d+[\d,]*\b', text))
        
        return entities | numbers
    
    def _check_content_fidelity(self, original: str, curated: str) -> bool:
        """
        Check that curated text covers the SAME topics as the original.
        Returns True if content fidelity is acceptable, False if curator
        hallucinated a completely different script or added unrelated content.
        """
        orig_entities = self._extract_key_entities(original)
        cur_entities = self._extract_key_entities(curated)
        
        if not orig_entities:
            return True  # Can't validate, assume OK
        
        overlap = orig_entities & cur_entities
        
        # Check 1: Original entities must be preserved in curated text
        preservation_ratio = len(overlap) / len(orig_entities)
        
        if preservation_ratio < 0.25:
            print(f"  [CURATOR] ⚠️ CONTENT MISMATCH — only {preservation_ratio:.0%} entity preservation")
            print(f"  [CURATOR] Original entities: {sorted(orig_entities)[:10]}")
            print(f"  [CURATOR] Curated entities: {sorted(cur_entities)[:10]}")
            print(f"  [CURATOR] Overlap: {sorted(overlap)}")
            return False
        
        # Check 2: Curated text must not introduce many NEW entities
        # (indicates the curator added unrelated/hallucinated content)
        novel_entities = cur_entities - orig_entities
        if len(cur_entities) > 0:
            novelty_ratio = len(novel_entities) / len(cur_entities)
            
            if novelty_ratio > 0.40 and len(novel_entities) > 3:
                print(f"  [CURATOR] ⚠️ NOVEL CONTENT DETECTED — {len(novel_entities)} new entities ({novelty_ratio:.0%} of curated)")
                print(f"  [CURATOR] Novel entities: {sorted(novel_entities)[:15]}")
                print(f"  [CURATOR] Expected entities: {sorted(orig_entities)[:10]}")
                return False
        
        return True
    
    def curate_script(self, script: dict) -> Optional[str]:
        """
        Structural Slicing Curation: Only sends story narration bodies to the LLM.
        Greeting, intro_hook, segues, and closing are NEVER touched by the curator.
        They are reassembled deterministically in code after curation.
        
        Args:
            script: Script dict with 'greeting', 'intro_hook', 'stories', 'closing'
            
        Returns:
            Full curated script text with structural elements preserved, or None on failure.
        """
        prompt_config = self.config["prompts"]["script_curator"]
        
        # ── EXTRACT ONLY STORY BODIES ──
        story_bodies = []
        for i, story in enumerate(script.get('stories', [])):
            p1 = story.get('part_1_narration', '')
            p2 = story.get('part_2_narration', '')
            body = f"{p1} {p2}".strip()
            story_bodies.append(body)
        
        if len(story_bodies) < 2:
            print(f"  [CURATOR] Not enough stories to curate ({len(story_bodies)}), using original")
            return script.get('full_text', '')
        
        # Build body-only text with clear story markers
        body_text = "\n\n---\n\n".join(
            f"[STORY {i+1}]\n{body}" for i, body in enumerate(story_bodies)
        )
        
        prompt = f"""Transform these 3 story narrations from written text into natural, human-sounding spoken language.

You receive ONLY the story narration bodies — no greeting, no segues, no closing.
Your job is ONLY to improve the rhythm and naturalness of each story's narration.

RULES:
- NEVER change facts, numbers, or country names
- NEVER add or remove information
- Break long sentences into short punchy ones
- Use PERIODS for dramatic pauses before punchlines
  Example: 'Classic leverage play. Disguised as safety.' NOT 'Classic leverage play... disguised as safety.'
- Use '—' for abrupt contrasts
- Move key numbers to end of sentences (punch position)
- Use contractions ALWAYS (it's, they're, won't)
- Create rhythm: alternate short punchy + longer explanatory sentences
- Before every punchline/reveal, end previous sentence with a PERIOD, start punchline as new sentence
- After rhetorical questions, use a period before the answer
- Balance all 3 stories to roughly equal word count (40-55 words each)
- Keep the [STORY N] markers exactly as they are
- Output all 3 stories, one after another, separated by --- lines

ORIGINAL STORY NARRATIONS:
{body_text}

Output the 3 curated stories as plain text. Keep [STORY N] markers. Separate stories with ---. No JSON. No explanations."""

        # ── OLLAMA ONLY: Curation contains geopolitical narration
        # that triggers GLM-5's content filter (1301). No cloud fallback.
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            print("  [CURATOR] Local model unavailable — no cloud fallback for sensitive content")
            return self._reassemble_script(script, story_bodies)
        
        # Clean any accidental markdown wrapping
        curated = response.strip()
        if curated.startswith('```'):
            curated = curated.split('\n', 1)[-1]
        if curated.endswith('```'):
            curated = curated.rsplit('```', 1)[0]
        curated = curated.strip()
        
        # ── PARSE CURATED BODIES BACK INTO 3 STORIES ──
        curated_bodies = self._parse_curated_stories(curated, len(story_bodies))
        
        if not curated_bodies or len(curated_bodies) < len(story_bodies):
            print(f"  [CURATOR] Could not parse {len(story_bodies)} stories from response (got {len(curated_bodies) if curated_bodies else 0}), using original")
            return self._reassemble_script(script, story_bodies)
        
        # ── PER-STORY FIDELITY CHECK ──
        for i in range(len(story_bodies)):
            if not self._check_content_fidelity(story_bodies[i], curated_bodies[i]):
                print(f"  [CURATOR] ⚠️ Story {i+1} failed fidelity check — using original narration")
                curated_bodies[i] = story_bodies[i]
        
        # ── UPDATE STORY NARRATIONS WITH CURATED TEXT ──
        import re as _re
        for i, body in enumerate(curated_bodies):
            if i < len(script.get('stories', [])):
                sentences = _re.split(r'(?<=[.!?])\s+', body.strip())
                mid = max(1, len(sentences) // 2)
                script['stories'][i]['part_1_narration'] = ' '.join(sentences[:mid])
                script['stories'][i]['part_2_narration'] = ' '.join(sentences[mid:])
        script['_curated'] = True

        # ── REASSEMBLE WITH STRUCTURAL ELEMENTS ──
        result = self._reassemble_script(script, curated_bodies)
        
        total_orig = sum(len(b.split()) for b in story_bodies)
        total_cur = sum(len(b.split()) for b in curated_bodies)
        print(f"  [CURATOR] Stories curated: {total_orig} → {total_cur} words (structural elements preserved)")
        return result
    
    def _parse_curated_stories(self, curated_text: str, expected_count: int) -> Optional[List[str]]:
        """
        Parse curated LLM response back into individual story bodies.
        Handles [STORY N] markers or --- separators.
        """
        import re
        
        stories = []
        
        # Strategy 1: Split by [STORY N] markers
        story_pattern = r'\[STORY\s+\d+\]\s*\n?'
        parts = re.split(story_pattern, curated_text)
        # Filter empty parts
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= expected_count:
            return parts[:expected_count]
        
        # Strategy 2: Split by --- separators
        if '---' in curated_text:
            parts = curated_text.split('---')
            parts = [p.strip() for p in parts if p.strip() and not re.match(r'^\[STORY', p.strip())]
            # Remove [STORY N] prefix from each part
            cleaned = []
            for p in parts:
                p = re.sub(r'^\[STORY\s+\d+\]\s*\n?', '', p).strip()
                if p:
                    cleaned.append(p)
            if len(cleaned) >= expected_count:
                return cleaned[:expected_count]
        
        # Strategy 3: If we got some stories but not all, use what we have
        if parts and len(parts) > 0:
            while len(parts) < expected_count:
                parts.append("")  # Empty placeholder — will be caught by fidelity check
            return parts[:expected_count]
        
        return None
    
    def _reassemble_script(self, script: dict, story_bodies: List[str]) -> str:
        """
        Deterministically reassemble the full script from structural elements + curated bodies.
        Greeting, intro_hook, segues, and closing are NEVER modified.
        """
        parts = []
        
        # Greeting + Intro Hook (always preserved)
        greeting = script.get('greeting', '')
        intro_hook = script.get('intro_hook', '')
        if greeting:
            parts.append(greeting)
        if intro_hook:
            parts.append(intro_hook)
        
        # Stories with segues
        stories = script.get('stories', [])
        for i in range(len(story_bodies)):
            parts.append(story_bodies[i])
            
            # Add segue + separator after non-last stories
            if i < len(story_bodies) - 1:
                story = stories[i] if i < len(stories) else {}
                segue = story.get('segue', '')
                if segue:
                    parts.append(segue)
                parts.append('....')
        
        # Closing (always preserved)
        closing = script.get('closing', '')
        if closing:
            parts.append(closing)
        
        full_text = ' '.join(filter(None, parts))
        
        # VALIDATE CLOSING: Safety net
        full_text = self._validate_closing(full_text)
        
        return full_text
    
    def _call_openai(self, system_prompt: str, user_prompt: str, temperature: float = 0.4, max_tokens: int = 2000, purpose: str = "generation") -> Optional[str]:
        """
        Call gpt-5-mini via OpenAI API.
        Falls back to local Ollama if API key not set or call fails.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print(f"  [OPENAI] No API key found, falling back to local Ollama")
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            print(f"  [OPENAI] Calling gpt-5-mini for {purpose}...")
            
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_tokens * 4,
            )
            
            content = response.choices[0].message.content
            
            if content:
                print(f"  [OPENAI] Responded ({len(content)} chars)")
                return content
            else:
                print(f"  [OPENAI] Empty response (reasoning tokens may have consumed budget). Retrying with 2x budget...")
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=max_tokens * 8,
                )
                content = response.choices[0].message.content
                if content:
                    print(f"  [OPENAI] Retry succeeded ({len(content)} chars)")
                    return content
                print(f"  [OPENAI] Empty response on retry, falling back to Ollama")
                return None
                
        except Exception as e:
            print(f"  [OPENAI] API call failed: {e}, falling back to local model")
            return None
    
    def generate_visual_prompts(self, script: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate 6 dedicated visual prompts from curated narration text.
        Uses local abliterated model exclusively (no cloud API).
        
        Args:
            script: Script dict with 'stories' array containing part_1_narration / part_2_narration
            
        Returns:
            List of 6 dicts: [{'scene': 'story_N_partM', 'description': '...'}, ...]
        """
        prompt_config = self.config["prompts"].get("visual_prompt_generator")
        if not prompt_config:
            print("  [VISUAL-GEN] No visual_prompt_generator config found, skipping")
            return None
        
        stories = script.get('stories', [])
        if not stories:
            print("  [VISUAL-GEN] No stories in script, skipping")
            return None
        
        # Build narration block — explicitly labeled for 1:1 mapping
        narration_block = ""
        for i, story in enumerate(stories, 1):
            p1 = story.get('part_1_narration', story.get('mini_hook', ''))
            p2 = story.get('part_2_narration', story.get('body', ''))
            narration_block += f"""
--- story_{i}_part1 (THE SETUP for Story {i}) ---
NARRATION: "{p1}"

--- story_{i}_part2 (THE PAYOFF for Story {i}) ---
NARRATION: "{p2}"
"""
        
        system_prompt = prompt_config["system_prompt"]
        
        user_prompt = f"""You MUST generate exactly 6 visual scene descriptions. Each scene MUST depict EXACTLY what the corresponding narration says.

CRITICAL MAPPING RULES — DO NOT shuffle or rearrange:
- story_1_part1 → MUST visually depict what Story 1 Part 1 narration describes
- story_1_part2 → MUST visually depict what Story 1 Part 2 narration describes
- story_2_part1 → MUST visually depict what Story 2 Part 1 narration describes
- story_2_part2 → MUST visually depict what Story 2 Part 2 narration describes
- story_3_part1 → MUST visually depict what Story 3 Part 1 narration describes
- story_3_part2 → MUST visually depict what Story 3 Part 2 narration describes

NARRATION SEGMENTS:
{narration_block}

COMPOSITION REQUIREMENTS for each scene description:
1. STYLE: "16-bit isometric pixel art, 30-degree overhead camera angle, retro video game aesthetic"
2. FOREGROUND: Specific subject positioned on the LEFT side (this is a split-screen layout — right side is covered by an avatar overlay)
3. MIDGROUND: The main action/event happening — dynamic pose, motion implied
4. BACKGROUND: The specific geographic location mentioned in narration — recognizable landmarks, terrain, flags
5. COLOR PALETTE: Choose from — warm oranges/reds (conflict), cool blues (diplomatic), golden yellows (economic), dark greens (military) — based on story mood
6. LIGHTING: Time of day that matches the mood — sunset (tense), dawn (hope), night (covert), golden hour (dramatic)
7. DETAILS: Include at least 2 specific recognizable elements (flags, equipment, uniforms, buildings) that make the scene identifiable

Each description MUST be 2-4 FULL SENTENCES. Start with "16-bit isometric pixel art scene:" then describe foreground, midground, background, and lighting. NO '+' shorthand. NO vague language.

Output ONLY valid JSON:
{{
  "scenes": [
    {{"scene": "story_1_part1", "description": "..."}},
    {{"scene": "story_1_part2", "description": "..."}},
    {{"scene": "story_2_part1", "description": "..."}},
    {{"scene": "story_2_part2", "description": "..."}},
    {{"scene": "story_3_part1", "description": "..."}},
    {{"scene": "story_3_part2", "description": "..."}}
  ]
}}"""
        
        # ── TRY LOCAL GEMMA 4 FIRST, FALL BACK TO GLM-5 CLOUD ──
        response = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"]
        )
        
        if not response:
            print("  [VISUAL-GEN] Local model unavailable for visual prompts")
            return None
        
        if not response:
            print("  [VISUAL-GEN] All LLM calls failed")
            return None
        
        result = self._extract_json(response)
        if not result or not isinstance(result, dict):
            print(f"  [VISUAL-GEN] Failed to parse JSON response")
            print(f"  [VISUAL-GEN] Raw: {response[:300]}")
            return None
        
        scenes = result.get('scenes', [])
        if not scenes or len(scenes) < 6:
            print(f"  [VISUAL-GEN] Expected 6 scenes, got {len(scenes)}")
            if len(scenes) >= 3:
                while len(scenes) < 6:
                    scenes.append({
                        'scene': f'fallback_{len(scenes)+1}',
                        'description': '16-bit isometric pixel art scene: Geopolitical world map with highlighted conflict regions, dramatic sunset lighting, military units positioned on left side'
                    })
            else:
                return None
        
        # Validate each scene has a meaningful description
        for i, scene in enumerate(scenes):
            desc = scene.get('description', '')
            word_count = len(desc.split())
            if word_count < 5:
                print(f"  [VISUAL-GEN] Scene {i} too short ({word_count} words), using fallback")
                scenes[i]['description'] = '16-bit isometric pixel art scene: Geopolitical world map with highlighted regions, military assets in foreground left, dramatic sunset lighting'
            scenes[i]['scene'] = scenes[i].get('scene', f'story_{(i//2)+1}_part{(i%2)+1}')
        
        # ── DEDUPLICATION: Detect and regenerate duplicate descriptions ──
        scenes = self._deduplicate_visual_prompts(scenes, user_prompt, system_prompt)
        
        print(f"  [VISUAL-GEN] Generated {len(scenes)} visual prompts")
        for s in scenes:
            print(f"    [{s['scene']}] {s.get('description', '')[:80]}...")
        
        return scenes[:6]  # Ensure exactly 6
    
    def _deduplicate_visual_prompts(self, scenes: List[Dict], user_prompt: str, system_prompt: str) -> List[Dict]:
        """Detect duplicate visual descriptions using keyword/entity overlap and regenerate them."""
        from difflib import SequenceMatcher
        import re
        
        def _extract_entities(text: str) -> set:
            """Extract capitalized proper nouns (countries, cities, waterways, organizations)."""
            words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
            geo_terms = {'strait', 'gulf', 'ocean', 'sea', 'river', 'mountain', 'border',
                        'capital', 'port', 'base', 'channel', 'bay', 'coast'}
            text_lower = text.lower()
            found_geo = {g for g in geo_terms if g in text_lower}
            return set(words) | found_geo
        
        duplicates_found = False
        for i in range(len(scenes)):
            for j in range(i + 1, len(scenes)):
                desc_i = scenes[i].get('description', '')
                desc_j = scenes[j].get('description', '')
                
                text_similarity = SequenceMatcher(None, desc_i.lower(), desc_j.lower()).ratio()
                
                entities_i = _extract_entities(desc_i)
                entities_j = _extract_entities(desc_j)
                shared = entities_i & entities_j
                entity_overlap = len(shared) / max(len(entities_i), len(entities_j), 1)
                
                is_duplicate = text_similarity > 0.6 or entity_overlap > 0.5
                
                if is_duplicate:
                    print(f"  [VISUAL-GEN] \u26a0\ufe0f Duplicate detected: {scenes[i]['scene']} \u2194 {scenes[j]['scene']} (text={text_similarity:.0%}, entity={entity_overlap:.0%}, shared={shared})")
                    duplicates_found = True
                    
                    diff_prompt = f"""You previously generated this visual description:

SCENE A ({scenes[i]['scene']}): {desc_i}

It uses these entities: {', '.join(sorted(entities_i))}.

Now generate a COMPLETELY DIFFERENT description for SCENE B ({scenes[j]['scene']}).
CRITICAL RULES:
- Do NOT reuse ANY of these entities: {', '.join(sorted(entities_i))}
- Pick a DIFFERENT geographic location, DIFFERENT country, DIFFERENT equipment
- The new scene must depict a totally different moment in a different place
- Still follow the pixel art style requirements

Output ONLY JSON: {{"scene": "{scenes[j]['scene']}", "description": "..."}}"""
                    
                    retry = self.generate(
                        prompt=diff_prompt,
                        system_prompt="You are a pixel art scene designer. Generate visually and geographically distinct scenes.",
                        temperature=0.6,
                        max_tokens=300
                    )
                    
                    if retry:
                        parsed = self._extract_json(retry)
                        if parsed and isinstance(parsed, dict) and parsed.get('description'):
                            new_desc = parsed['description']
                            new_entities = _extract_entities(new_desc)
                            still_shared = new_entities & entities_i
                            new_entity_overlap = len(still_shared) / max(len(new_entities), len(entities_i), 1)
                            
                            if new_entity_overlap < 0.4:
                                scenes[j]['description'] = new_desc
                                print(f"  [VISUAL-GEN] \u2705 Regenerated {scenes[j]['scene']} (entity overlap {entity_overlap:.0%} \u2192 {new_entity_overlap:.0%})")
                            else:
                                print(f"  [VISUAL-GEN] \u26a0\ufe0f Regeneration still shares entities ({still_shared}), keeping as-is")
        
        if not duplicates_found:
            print(f"  [VISUAL-GEN] \u2705 All 6 scenes are visually distinct (no duplicates)")
        
        return scenes
    
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
