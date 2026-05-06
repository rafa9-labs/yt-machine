import json
import os
import re
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

class LLMInterface:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "system_prompts.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_url = self.config["model_config"]["base_url"]
        self.default_model = self.config["model_config"]["default_model"]
        self.fallback_model = self.config["model_config"].get("fallback_model", "llama3.2:latest")
        self.timeout = self.config["model_config"]["timeout"]
        self.retry_attempts = self.config["model_config"]["retry_attempts"]
        self.num_ctx = self.config["model_config"].get("num_ctx", 4096)
        
        # Per-call timeouts to prevent Ollama bloat
        self.hard_call_timeout = self.config["model_config"].get("hard_call_timeout", 180)
        self.idle_timeout = self.config["model_config"].get("idle_timeout", 60)
        self.call_timeouts = self.config["model_config"].get("call_timeouts", {})
        
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
        import re

        text = re.sub(
            r'<\|\s*channel\s*(?:\|?\s*)?>\s*thought\s*<\s*channel\s*(?:\|?\s*)?>',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r'<\|\s*channel\s*(?:\|?\s*)?>\s*output\s*<\s*channel\s*(?:\|?\s*)?>',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r'<\|\s*channel\s*(?:\|?\s*)?>',
            '', text, flags=re.IGNORECASE
        )
        text = re.sub(r'<think\b.*?</think\s*>?', '', text, flags=re.DOTALL)
        text = re.sub(r'</?think[^>]*>?', '', text)

        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        json_start = re.search(r'[{\[]', text)
        if json_start:
            text = text[json_start.start():]

        return text.strip()

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        import re

        # Strip thinking tokens from heretic models
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

            # Close any trailing unclosed string literal
            quotes = 0
            _in_string = False
            _escaped = False
            for ch in json_str:
                if _escaped:
                    _escaped = False
                    continue
                if ch == '\\':
                    _escaped = True
                    continue
                if ch == '"':
                    _in_string = not _in_string
            if _in_string:
                json_str += '"'

            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')

            json_str += ']' * max(0, open_brackets)
            json_str += '}' * max(0, open_braces)

            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            return json.loads(json_str)
        except:
            return None
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], attempt: int = 1,
                      hard_timeout: float = None, idle_timeout: float = None) -> Optional[Dict[str, Any]]:
        _hard = hard_timeout if hard_timeout is not None else self.hard_call_timeout
        _idle = idle_timeout if idle_timeout is not None else self.idle_timeout
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
            start = time.monotonic()
            last_data = time.monotonic()
            for line in response.iter_lines():
                now = time.monotonic()
                if now - start > _hard:
                    print(f"  [LLM] Hard timeout ({_hard:.0f}s exceeded) - aborting request")
                    response.close()
                    return None
                if now - last_data > _idle:
                    print(f"  [LLM] Idle timeout ({_idle:.0f}s no data) - aborting request")
                    response.close()
                    return None
                if line:
                    last_data = now
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
                return self._make_request(endpoint, payload, attempt + 1,
                                          hard_timeout=_hard, idle_timeout=_idle)
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
        max_tokens: int = 500,
        task_name: str = None
    ) -> Optional[str]:
        if model is None:
            model = self.default_model
        
        task_timeouts = self.call_timeouts.get(task_name, {}) if task_name else {}
        hard_timeout = task_timeouts.get("hard") if task_timeouts else None
        idle_timeout = task_timeouts.get("idle") if task_timeouts else None
        
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
            print(f"  [LLM] ⚡ Primary model circuit breaker active ({self._primary_failures} failures), using fallback: {self.fallback_model}")
            payload["model"] = self.fallback_model
        
        result = self._make_request("/api/generate", payload,
                                    hard_timeout=hard_timeout,
                                    idle_timeout=idle_timeout)
        
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
            result = self._make_request("/api/generate", payload,
                                         hard_timeout=hard_timeout,
                                         idle_timeout=idle_timeout)
            
            if result and result.get("response", "").strip():
                print(f"  [LLM] ✓ Fallback model ({fallback}) succeeded ({len(result['response'])} chars)")
                return result["response"]
        
        print(f"  [LLM] ✗ All models in fallback chain failed")
        
        return None
    
    def unload_model(self, model: str = None) -> bool:
        model = model or self.default_model
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=10
            )
            print(f"  [LLM] Model '{model}' unloaded (keep_alive=0)")
            return True
        except Exception as e:
            print(f"  [LLM] Failed to unload model: {e}")
            return False
    
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
        
        result = self._make_request("/api/chat", payload,
                                    hard_timeout=self.hard_call_timeout,
                                    idle_timeout=self.idle_timeout)
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
            max_tokens=prompt_config["max_tokens"],
            task_name="news_processor"
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
            max_tokens=prompt_config["max_tokens"],
            task_name="news_processor"
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
            max_tokens=prompt_config["max_tokens"],
            task_name="news_processor"
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
            max_tokens=prompt_config["max_tokens"],
            task_name="script_synthesizer"
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
                max_tokens=1200,
                task_name="script_synthesizer"
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
        """Get time-of-day greeting for The Mask persona — explosive entrance."""
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            return "Ssssmokin'! Good morning, folks! It's SHOWTIME!"
        elif hour < 18:
            return "Ssssmokin'! Good afternoon, baby! It's SHOWTIME!"
        else:
            return "Ssssmokin'! Good evening, folks! It's SHOWTIME!"
    
    # ── SEGUE TEMPLATES: The Mask-style frantic cartoonish bridges ──
    _SEGUE_TEMPLATES = [
        "But WAIT—hold onto your lobsters! That's not even the CRAZIEST part!",
        "Oh we are JUST getting started, baby!",
        "And if you thought THAT was wild... just WAIT!",
        "You think that's something? You ain't seen NOTHING yet!",
        "But here's where it gets REALLY interesting, folks!",
        "And believe it or not, it gets even CRAZIER.",
    ]
    
    INTRO_HOOK_TEMPLATES = [
        "Two stories. One screen. Let us go.",
        "Tonight the dominoes are already falling.",
        "Big moves. Big consequences. Let us dive in.",
        "The world just shifted. Here is what it means.",
        "Hold tight. Two stories that change everything.",
        "Chaos incoming. Two stories you cannot miss.",
    ]

    def _enforce_greeting(self, script: dict) -> dict:
        """
        GUARANTEE: Every script MUST have a non-empty greeting.
        Intro_hook is optional — short greetings don't need it.
        """
        import random
        
        GREETING_TEMPLATES = [
            "Ssssmokin'!",
            "Hold onto your lobsters, folks!",
            "It is showtime!",
            "Did somebody order CHAOS?",
            "Baby, you are NOT ready for this!",
            "Well, well, well. Look what just walked in!",
        ]
        
        if not script.get('greeting', '').strip():
            script['greeting'] = random.choice(GREETING_TEMPLATES)
            print(f"  [GREETING] Generated greeting: \"{script['greeting']}\"")
        
        # intro_hook is optional — clear if present to avoid redundancy with short greetings
        if script.get('intro_hook', '').strip():
            script['intro_hook'] = ''
        
        return script
    
    def _ensure_greeting_in_fulltext(self, script: dict) -> dict:
        """
        GUARANTEE: full_text MUST start with the greeting.
        This is called after every full_text modification (synthesis, curation, etc.)
        Uses fuzzy prefix match to avoid prepending duplicate greetings.
        """
        full_text = script.get('full_text', '').strip()
        greeting = script.get('greeting', '').strip()
        
        if not full_text or not greeting:
            return script
        
        ft_lower = full_text.lower()
        g_lower = greeting.lower()
        g_first_word = greeting.split()[0].lower() if greeting.split() else ''
        
        if ft_lower.startswith(g_lower) or ft_lower.startswith(g_lower.rstrip('.,!?')):
            return script
        
        ft_prefix = ft_lower[:len(g_lower) + 5]
        if g_lower in ft_prefix:
            return script
        
        if g_first_word and g_first_word not in ft_lower[:len(g_lower) + 10]:
            script['full_text'] = f"{greeting} {full_text}"
            print(f"  [GREETING] Prepended missing greeting to full_text")
        
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
                kw in segue.lower() for kw in ['but', 'and', 'now', 'wait', 'that', 'here', 'check', 'sneaky', 'wild', 'crazy', 'insane', 'believe', 'next', 'last', "won't believe", 'lobsters', 'baby', 'folks', 'showtime', 'hold']
            )
            
            if not is_valid:
                # Pick a template (rotated based on story index to avoid repetition)
                template_idx = (i + hash(str(story.get('part_1_narration', '')))) % len(self._SEGUE_TEMPLATES)
                new_segue = self._SEGUE_TEMPLATES[template_idx]
                story['segue'] = new_segue
                print(f"  [SEGUE] Story {i+1} segue was weak/empty, injected: \"{new_segue}\"")
        
        return script
    
    def _enforce_fallout(self, script: dict, analyses: list = None) -> dict:
        """Ensure every story has a non-empty fallout field and a valid fallout_visual.
        If fallout narration is missing, derive from second_order_consequence.
        NEVER copy real_talk verbatim — that causes repetition when both are spoken aloud."""
        stories = script.get('stories', [])
        for i, story in enumerate(stories):
            fallout = story.get('fallout', '').strip()
            if fallout and len(fallout.split()) >= 5:
                pass
            else:
                source = ''
                if analyses and i < len(analyses):
                    source = analyses[i].get('second_order_consequence', '')
                if not source or len(source.split()) < 3:
                    real_talk = story.get('real_talk', '')
                    if real_talk:
                        # Transform real_talk into a forward-looking consequence rather than copying it
                        rt_words = real_talk.split()
                        # Extract key nouns/entities from real_talk for the forward-looking phrase
                        key_words = [w for w in rt_words if w[0].isupper() or any(c.isdigit() for c in w)][:3]
                        if key_words:
                            source = f"The ripple from {' '.join(key_words)} is just beginning — and nobody is watching the next domino."
                        else:
                            source = "The ripple effects are just beginning — and the next domino is already falling."
                
                prefix = story.get('fallout', '').strip()
                if prefix and len(prefix.split()) >= 2:
                    fallout = prefix
                elif source:
                    fallout = source
                else:
                    fallout = "The consequences are still unfolding — and the next domino is already in motion."
                story['fallout'] = fallout
                print(f"  [FALLOUT] Story {i+1} fallout enforced: \"{fallout[:60]}...\"" if len(fallout) > 60 else f"  [FALLOUT] Story {i+1} fallout enforced: \"{fallout}\"")
            
            # Validate fallout_visual: must have composition structure, not narration
            fallout_visual = story.get('fallout_visual', '').strip()
            is_valid_visual = LLMInterface._validate_visual_prompt_composition(fallout_visual) if fallout_visual else False
            if not fallout_visual or (not is_valid_visual and len(fallout_visual.split()) < 10):
                import random
                templates = [
                    "16-bit isometric pixel art scene: chain of dominoes collapsing, consequences spreading in foreground, dark horizon on left, twilight atmosphere",
                    "16-bit isometric pixel art scene: cracked ground spreading toward distant city on right, foreground debris on left, heavy fog, dramatic shadows",
                    "16-bit isometric pixel art scene: single ember igniting dry field, smoke rising in background, aerial perspective, dusk lighting",
                    "16-bit isometric pixel art scene: shadowy figure walking from crumbling structure in foreground, destruction visible in midground, heavy fog",
                    "16-bit isometric pixel art scene: massive wave building offshore, small boats scrambling in foreground, dark clouds in background, aerial view",
                ]
                visual_source = story.get('part_2_visual', story.get('real_talk_visual', ''))
                if visual_source and LLMInterface._validate_visual_prompt_composition(visual_source):
                    subject = visual_source.split(',')[0].strip()
                    subject = subject.replace('Pixel art', '').replace('pixel art', '').replace('pixel_art', '').strip()
                    if subject and len(subject.split()) >= 3:
                        story['fallout_visual'] = f"16-bit isometric pixel art scene: aftermath of {subject.lower()}, consequences visible in foreground, dark horizon on left, twilight atmosphere, ominous sky"
                    else:
                        story['fallout_visual'] = random.choice(templates)
                else:
                    story['fallout_visual'] = random.choice(templates)
                print(f"  [FALLOUT] Story {i+1} fallout_visual enforced: \"{story['fallout_visual'][:70]}...\"")
        script['stories'] = stories
        return script
    
    # Scene type templates for narration-to-visual conversion
    _SCENE_TYPE_TEMPLATES = {
        'hook': '16-bit isometric pixel art scene: dramatic wide establishing shot of',
        'mechanism': '16-bit isometric pixel art scene: tactical close-up view of',
        'truth': '16-bit isometric pixel art scene: somber revealing scene depicting',
        'fallout': '16-bit isometric pixel art scene: forward-looking consequence scene showing',
    }

    _COMPOSITION_KEYWORDS = frozenset([
        'foreground', 'midground', 'background', 'left side', 'right side', 'split-screen',
        'overhead', 'aerial', 'isometric', 'camera', 'shot', 'perspective', 'view',
        'lighting', 'sunset', 'dawn', 'dusk', 'golden hour', 'twilight',
        'scene:', 'depicting', 'showing', 'with', 'positioned', 'visible',
    ])

    _NARRATION_PATTERNS = frozenset([
        'holding a', 'standing in', 'says', 'told', 'explains', 'reveals that',
        'announces', 'reports', 'states', 'declares', 'argues that',
        'close-up of a single', 'a single', 'the aftermath of',
    ])

    METAPHOR_TO_SCENE = {
        'wedge': ('dividing barrier', 'strategic split'),
        'nail': ('fortification breach', 'structural impact'),
        'padlock': ('security lockdown', 'access restriction'),
        'lock': ('security lockdown', 'access restriction'),
        'crumbling': ('collapsing structure', 'institutional decay'),
        'fracture': ('breaking alliance', 'shattered coalition'),
        'fractured': ('broken alliance', 'shattered coalition'),
        'domino': ('cascading failure', 'chain reaction'),
        'fire': ('escalating conflict', 'spreading crisis'),
        'spark': ('initial provocation', 'ignition point'),
        'wall': ('defensive barrier', 'separation structure'),
        'bridge': ('diplomatic connection', 'negotiation channel'),
        'chain': ('linked dependencies', 'interconnected systems'),
        'mirror': ('parallel situation', 'reflected consequence'),
        'mask': ('hidden agenda', 'covert operation'),
        'puppet': ('controlled entity', 'external influence'),
        'game': ('strategic maneuver', 'calculated move'),
        'chess': ('strategic maneuver', 'calculated positioning'),
    }

    _METAPHOR_INDICATORS = frozenset([
        "it's a", "it is a", "like a", "acting as", "serving as",
        "driven into", "driven through", "clamped onto", "hung on",
        "the heart of", "the backbone of", "the cornerstone",
        "a way to", "turning into", "reduced to",
    ])

    @staticmethod
    def _detect_metaphor_narration(narration: str) -> list:
        """
        Detect figurative language in narration that would produce bad visual
        prompts if rendered literally. Returns list of metaphor keywords found
        that also have an indicator phrase (to reduce false positives on words
        like "game" or "fire" used literally).
        """
        if not narration or len(narration) < 5:
            return []
        text_lower = narration.lower()
        has_indicator = any(ind in text_lower for ind in LLMInterface._METAPHOR_INDICATORS)
        if not has_indicator:
            return []
        found = [kw for kw in LLMInterface.METAPHOR_TO_SCENE if kw in text_lower]
        return found

    @staticmethod
    def _validate_visual_prompt_composition(prompt: str) -> bool:
        """
        Check whether a visual prompt has composition structure (foreground/background/lighting)
        or is narration-style (flat description of a subject).

        Returns True if composition-style, False if narration-style.
        """
        if not prompt or len(prompt.strip()) < 20:
            return False
        
        p = prompt.lower()
        
        composition_hits = sum(1 for kw in LLMInterface._COMPOSITION_KEYWORDS if kw in p)
        narration_hits = sum(1 for kw in LLMInterface._NARRATION_PATTERNS if kw in p)
        
        if narration_hits >= 2:
            return False
        
        if composition_hits >= 2:
            return True

        sentences = [s.strip() for s in p.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) < 2:
            return False

        has_scene_structure = any(
            kw in p for kw in ['foreground', 'midground', 'background', 'left', 'lighting', 'camera', 'shot']
        )
        
        starts_with_scene = p.strip().startswith('16-bit') or 'scene:' in p
        
        return has_scene_structure or starts_with_scene
    
    @staticmethod
    def _ensure_visual_prompt(visual_field: str, narration: str, scene_type: str) -> str:
        """
        Ensure a visual prompt is non-empty and describes a visual SCENE with composition,
        not narration-style flat descriptions.

        Fallback chain:
        1. Use the visual field if it passes composition validation
        2. If narration contains metaphor language, convert to concrete scene elements
        3. Convert narration to composition-style visual prompt using templates + entity extraction
        4. Use a generic scene placeholder based on the scene type
        """
        if visual_field and len(visual_field.strip()) >= 15:
            if LLMInterface._validate_visual_prompt_composition(visual_field):
                return visual_field.strip()

        if narration and len(narration.strip()) >= 10:
            template = LLMInterface._SCENE_TYPE_TEMPLATES.get(scene_type, '16-bit isometric pixel art scene depicting')
            lighting = {'hook': 'sunset lighting', 'mechanism': 'golden hour lighting',
                        'truth': 'cold blue lighting', 'fallout': 'twilight atmosphere'}.get(scene_type, 'dramatic lighting')

            metaphors = LLMInterface._detect_metaphor_narration(narration)
            if metaphors:
                scene_elements = []
                for m in metaphors[:2]:
                    scene_elements.extend(LLMInterface.METAPHOR_TO_SCENE.get(m, ()))
                entities = LLMInterface._extract_key_entities(narration)
                metaphor_words = set(LLMInterface.METAPHOR_TO_SCENE.keys())
                geo_entities = [e for e in list(entities) if e[0].isupper() and e.lower() not in metaphor_words][:2]
                all_parts = scene_elements + geo_entities
                if all_parts:
                    return f"{template} {', '.join(all_parts)}, {lighting}, atmospheric depth"

            entities = LLMInterface._extract_key_entities(narration)
            if entities:
                entity_str = ', '.join(list(entities)[:3])
                return f"{template} {entity_str}, {lighting}, atmospheric depth"
            cleaned = narration.strip().rstrip('.!?')
            if len(cleaned) > 120:
                words = cleaned.split()
                cleaned = ' '.join(words[:20])
            return f"{template} {cleaned.lower()}, {lighting}, atmospheric depth"
        
        type_defaults = {
            'hook': '16-bit isometric pixel art scene: dramatic wide establishing shot, military forces positioned on left, strategic landscape at sunset, atmospheric haze',
            'mechanism': '16-bit isometric pixel art scene: tactical close-up of strategic infrastructure, detailed equipment on left side, dynamic composition at golden hour',
            'truth': '16-bit isometric pixel art scene: somber revealing scene, civilian perspective on left, consequences visible in background, cold blue lighting',
            'fallout': '16-bit isometric pixel art scene: forward-looking consequence scene, domino effect visible, dark horizon on left, twilight atmosphere, ominous sky',
        }
        return type_defaults.get(scene_type, type_defaults['hook'])
    
    # Unified closing — The Mask persona (Truman Show inspired)
    UNIFIED_CLOSING_BASE = "Stay behind the curtains, and if I don't see you — good morning, good afternoon, and goodnight."

    def _build_dynamic_closing(self, last_fallout: str = "", last_topic: str = "") -> str:
        """
        Build a dynamic closing that echoes the last story's fallout,
        creating a seamless bridge from fallout to sign-off.
        The bridge drops the manic Mask persona and transitions into melancholy.
        NEVER dumps a headline — extracts 2-3 key words from the topic.
        """
        bridge = ""
        if last_topic:
            # Strip headline format: "China-Iran Beijing Summit: Beijing Formalizes..."
            # becomes just the key concept before the colon
            topic_clean = last_topic.strip().rstrip('.')
            if ':' in topic_clean:
                topic_clean = topic_clean.split(':')[0].strip()
            # Extract 2-3 content words (skip common/topic-structure words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                          'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
                          'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had', 'do', 'does', 'did', 'will', 'would', 'could',
                          'should', 'may', 'might', 'can', 'shall', 'this', 'that',
                          'these', 'those', 'it', 'its', 'not', 'no', 'new', 'how'}
            words = [w for w in re.findall(r'\b\w+\b', topic_clean.lower())
                      if w not in stop_words and len(w) > 2]
            key_concept = ' '.join(words[:3]) if words else 'the board'
            bridge = f"And while {key_concept} reshapes the board... "
        elif last_fallout:
            words = re.findall(r'\b\w+\b', last_fallout.lower())
            key_nouns = [w for w in words[-6:] if len(w) > 3 and w not in
                          {'that', 'this', 'with', 'from', 'they', 'their',
                           'have', 'been', 'will', 'would', 'could', 'what',
                           'when', 'where', 'which', 'there', 'these', 'those'}]
            if key_nouns:
                bridge = f"And while {key_nouns[-1]} reshapes the board... "
            else:
                bridge = "And just like that... "
        else:
            bridge = "And just like that... "

        return bridge + self.UNIFIED_CLOSING_BASE
    
    @staticmethod
    def _scrub_cta_from_field(text: str) -> str:
        """Remove CTA/sign-off phrases from a single story field."""
        import re
        if not text:
            return text
        cta_patterns = re.compile(
            r'(?:Subscribe\s*(?:and\s+)?(?:like|share|comment|follow)[\s,.]*)'
            r'|(?:Like\s+and\s+share[\s,.]*)'
            r'|(?:follow\s+(?:for\s+more|us)[\s,.]*)'
            r'|(?:don\'t\s+forget\s+to\s+(?:like|share|subscribe)[\s,.]*)'
            r'|(?:thanks\s+for\s+watching[\s,.]*)'
            r'|(?:see\s+you\s+(?:next\s+time|soon|later)[\s,.]*)'
            r'|(?:that\'s\s+all\s+for\s+today[\s,.]*)'
            r'|(?:wrapping\s+(?:up|this\s+up)[\s,.]*)'
            r'|(?:stay\s+tuned[\s,.]*)',
            re.IGNORECASE
        )
        cleaned = cta_patterns.sub('', text)
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        return cleaned

    def _scrub_closing_from_stories(self, script: dict) -> dict:
        """
        Scrub CTA/sign-off text from ALL story narration fields.
        The LAST story is the highest risk — its real_talk and part_2_narration
        often absorb CTA that duplicates the canonical closing.
        """
        stories = script.get('stories', [])
        for i, story in enumerate(stories):
            for field in ('part_1_narration', 'part_2_narration', 'real_talk', 'fallout', 'segue'):
                original = story.get(field, '')
                cleaned = self._scrub_cta_from_field(original)
                if cleaned != original:
                    print(f"  [CTA-SCRUB] Story {i+1} {field}: removed CTA fragment")
                    story[field] = cleaned
        script['stories'] = stories
        return script
    
    def _dedup_segue_overlap(self, script: dict) -> dict:
        """
        Remove duplicate phrasing between a story's segue and the next story's part_1.
        If story N's segue starts the same phrase as story N+1's part_1, strip the
        overlap from part_1 so the audience doesn't hear the same clause twice.
        """
        stories = script.get('stories', [])
        if len(stories) < 2:
            return script
        
        for i in range(len(stories) - 1):
            segue = stories[i].get('segue', '').strip()
            next_p1 = stories[i + 1].get('part_1_narration', '').strip()
            
            if not segue or not next_p1:
                continue
            
            segue_words = segue.split()
            next_words = next_p1.split()
            
            max_check = min(3, len(segue_words), len(next_words))
            overlap_count = 0
            for n in range(1, max_check + 1):
                tail = [w.lower().rstrip('.,!?;:') for w in segue_words[-n:]]
                head = [w.lower().rstrip('.,!?;:') for w in next_words[:n]]
                if tail == head:
                    overlap_count = n
            
            if overlap_count >= 2:
                cleaned_p1 = ' '.join(next_words[overlap_count:]).strip()
                if cleaned_p1 and len(cleaned_p1.split()) >= 5:
                    print(f"  [DEDUP] Story {i+2} part_1: stripped {overlap_count} overlapping words from segue")
                    print(f"    Removed: \"{' '.join(next_words[:overlap_count])}\"")
                    stories[i + 1]['part_1_narration'] = cleaned_p1
        
        script['stories'] = stories
        return script

    def _dedup_inter_story_phrases(self, script: dict) -> dict:
        """
        Remove 3-gram overlap between story[i].fallout and story[i+1].part_1.
        Also checks story[i].real_talk against story[i+1].part_1.
        Strips overlapping words from the start of story[i+1].part_1.
        """
        stories = script.get('stories', [])
        if len(stories) < 2:
            return script
        
        def _trigrams(text: str) -> set:
            words = text.lower().split()
            return {' '.join(words[j:j+3]) for j in range(max(0, len(words) - 2))}
        
        for i in range(len(stories) - 1):
            for field in ('fallout', 'real_talk'):
                prev_text = stories[i].get(field, '').strip()
                next_p1 = stories[i + 1].get('part_1_narration', '').strip()
                if not prev_text or not next_p1:
                    continue
                
                prev_trigrams = _trigrams(prev_text)
                next_words = next_p1.split()
                
                overlap_count = 0
                for j in range(min(4, len(next_words))):
                    window = ' '.join(next_words[j:j+3]).lower() if j + 3 <= len(next_words) else ''
                    window_punct = ' '.join(w.lower().rstrip('.,!?;:') for w in next_words[j:j+3]) if j + 3 <= len(next_words) else ''
                    prev_trigrams_clean = {tg.rstrip('.,!?;:') for tg in prev_trigrams}
                    if window_punct in prev_trigrams_clean or window in prev_trigrams:
                        overlap_count = j + 1
                    else:
                        break
                
                if overlap_count >= 2:
                    cleaned_p1 = ' '.join(next_words[overlap_count:]).strip()
                    if cleaned_p1 and len(cleaned_p1.split()) >= 5:
                        print(f"  [DEDUP-INTER] Story {i+2} part_1: stripped {overlap_count} overlapping words from story {i+1} {field}")
                        print(f"    Removed: \"{' '.join(next_words[:overlap_count])}\"")
                        stories[i + 1]['part_1_narration'] = cleaned_p1
        
        script['stories'] = stories
        return script

    def _validate_closing(self, full_text: str) -> str:
        """
        Ensure the script ends with the UNIFIED The Mask closing/CTA.
        ALWAYS replaces or appends the canonical closing — never trusts LLM output.
        Also scrubs stray CTA phrases from mid-string to prevent duplication.
        """
        if not full_text:
            return full_text
        
        import re

        # Scrub stray CTA phrases anywhere in the text (not just at end)
        # LLMs sometimes inject "subscribe and like" mid-narration despite instructions
        cta_mid = re.compile(
            r'(?:Subscribe\s*(?:and\s+)?(?:like|share|comment|follow)[\s,.]*)'
            r'|(?:Like\s+and\s+share[\s,.]*)'
            r'|(?:follow\s+(?:for\s+more|us)[\s,.]*)'
            r'|(?:don\'t\s+forget\s+to\s+(?:like|share|subscribe)[\s,.]*)'
            r'|(?:thanks\s+for\s+watching[\s,.]*)'
            r'|(?:see\s+you\s+(?:next\s+time|soon|later)[\s,.]*)'
            r'|(?:that\'s\s+all\s+for\s+today[\s,.]*)'
            r'|(?:wrapping\s+(?:up|this\s+up)[\s,.]*)'
            r'|(?:stay\s+tuned[\s,.]*)',
            re.IGNORECASE
        )
        cleaned = cta_mid.sub('', full_text)
        cleaned = re.sub(r'  +', ' ', cleaned).strip()
        
        text_lower = cleaned.lower()
        tail = text_lower[-300:] if len(text_lower) > 300 else text_lower
        has_truman = 'good morning' in tail and 'good afternoon' in tail and 'goodnight' in tail
        
        if has_truman:
            return cleaned
        
        # Strip any existing LLM-generated closing to avoid duplication
        stripped = re.sub(
            r'\s*\.{3,4}\s*(?:And with that|Subscribe|That\'s all|So there you have|This is Masker|I\'m Masker|see you|And these were|These were|Stay tuned|See you next|Thanks for|follow for|Stay behind|The walls|Like and share|Ssssmokin|And that is how|And just like that).*$',
            '', cleaned, flags=re.IGNORECASE
        ).rstrip()
        
        # Second pass: strip any trailing subscribe/like/CTA patterns
        stripped = re.sub(
            r'\s*(?:Subscribe|subscribe|Like and share|like and share)[,.\s]*(?:and\s+)?(?:share|like|comment|follow)?\s*.*$',
            '', stripped, flags=re.IGNORECASE
        ).rstrip()
        
        # Build dynamic closing referencing last story topic if available
        if hasattr(self, '_last_story_topic') and self._last_story_topic:
            closing = self._build_dynamic_closing(
                last_fallout=getattr(self, '_last_fallout', ''),
                last_topic=self._last_story_topic
            )
        else:
            closing = self.UNIFIED_CLOSING_BASE
        
        result = stripped + ' .... ' + closing
        print(f"  [CLOSING] Injected dynamic closing (truman={has_truman})")
        return result
    
    def synthesize_multi_news_script(
        self,
        news_analyses: List[Dict[str, Any]],
        num_stories: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a multi-news Masker personality script.
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
        
        prompt = f"""Create a Mask script — {num_stories} stories, Infotainment Satire structure (The Cartoonish Truth).

GREETING TO USE: "{greeting}"

{news_block}

CRITICAL RULES:
- Output ONLY the JSON object. NO explanatory text before or after. NO markdown.
- The greeting field must be EXACTLY: {greeting} (3-6 words MAX, one exclamatory phrase only, no intro_hook)
- Each story: part_1 = THE HOOK (what happened), part_2 = THE MECHANISM (why it matters, the hidden chain), real_talk = THE TRUTH (visceral specific consequence), fallout = THE FALLOUT (one concrete forward consequence, what escalates next)
- part_2_narration must NOT contain real_talk or fallout content. They are SEPARATE fields.
- part_2 must answer SO WHAT — name the concrete second-order consequence. NO vague abstractions.
- real_talk must name ONE specific visceral consequence. NOT abstract principles.
- fallout must name ONE concrete forward consequence — what happens NEXT, what escalates, the ripple effect.
- Create ORIGINAL metaphors. These are BANNED — never use: 'erase the status quo like a bad drawing', 'old switcheroo', 'dance floor on fire', 'crashing the party', 'flip the script'
- The ONLY approved cartoon exclamation is 'Ssssmokin''. Do NOT invent random exclamations.
- Each non-last story must have a "segue" field that bridges FROM this story's FALLOUT to the next story's HOOK. The segue MUST name the next story's specific subject — NOT vague buildup like 'a different kind of X is brewing'. NAME the next event directly. Generic bridges like 'just wait!' are FORBIDDEN.
- Manic, chaotic, fast-talking energy — but the facts are REAL and DENSE
- The real_talk field is where The Mask drops the act — NO caps, NO exclamations, just flat visceral truth
- fallout continues the real_talk tone — factual, forward-looking, no exclamations
- Target: 150-170 words total for ~65-70 seconds.
- ALL {num_stories} stories must be roughly equal word count (~68-78 words each). Max 15 words difference.
- GEOGRAPHIC ANCHOR: Every location on first mention MUST carry its country — plain name only, NO descriptors. Format: "City, Country". Examples: "Misrata, Libya", "New York, USA", "Lisbon, Portugal", "Tehran, Iran" (NOT "Iran, the Middle Eastern power"). When multiple cities share the same country, group them: "Libyan cities of Misrata and Benghazi". No bare location names. No regional labels ("Middle Eastern power", "European nation", "Asian giant") — always use the country name.
- CTA QUARANTINE: Subscribe, like, share, or sign-off text may ONLY appear in the dedicated "closing" field. ANY CTA-like phrasing in narration fields is FORBIDDEN. The last story MUST end with fallout — NOT a conclusion or summary.
- CLOSING: The closing bridge must echo the LAST STORY'S FALLOUT using 2-3 plain words — NEVER dump a headline or topic title. NEVER introduce topics not in the stories. The closing only echoes what was already said."""
        
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
                max_tokens=prompt_config["max_tokens"],
                task_name="script_synthesizer"
            )
        
        if not response:
            print("  [MULTI-NEWS] Local model unavailable — no cloud fallback for sensitive content")
            return None
        
        script = self._extract_json(response)
        if not script or not isinstance(script, dict):
            print(f"Failed to parse multi-news script JSON (attempt 1)")
            print(f"Raw response: {response[:300]}...")

            retry_max = prompt_config["max_tokens"] * 2
            print(f"  [MULTI-NEWS] Retrying with max_tokens={retry_max}")

            retry_prompt = prompt + "\n\nCRITICAL: Your previous response was truncated. Write SHORTER narration — 18-22 words per segment MAXIMUM. Output ONLY valid JSON."

            retry_response = self.generate(
                prompt=retry_prompt,
                model=task_model,
                system_prompt=prompt_config["system_prompt"],
                temperature=prompt_config["temperature"],
                max_tokens=retry_max,
                task_name="script_synthesizer"
            )
            if retry_response:
                script = self._extract_json(retry_response)
                if script and isinstance(script, dict):
                    response = retry_response
                    print(f"  [MULTI-NEWS] Retry succeeded")

            if not script or not isinstance(script, dict):
                print(f"Failed to parse multi-news script JSON (attempt 2)")
                return None
        
        # Ensure greeting is set correctly
        script['greeting'] = greeting
        
        # Ensure intro_hook exists
        if not script.get('intro_hook'):
            script['intro_hook'] = "Ssssmokin'! We have got TWO stories and the dance floor is on FIRE!"
        
        # Ensure stories exist
        if 'stories' not in script or not script.get('stories'):
            print(f"  [MULTI-NEWS] No stories in response — falling back")
            return None
        
        # ══════════════════════════════════════════════════════════════
        # VALIDATION 1: Ensure exactly num_stories stories with non-empty required fields
        # ══════════════════════════════════════════════════════════════
        stories = script.get('stories', [])
        
        # Retry synthesis if fewer than num_stories stories (max 3 attempts)
        if len(stories) < num_stories:
            print(f"  [VALIDATE] Only {len(stories)} stories — need {num_stories}. Re-synthesizing...")
            for retry_attempt in range(3):
                retry_prompt = (
                    f"The previous script only had {len(stories)} stories. "
                    f"You MUST produce exactly {num_stories} stories. Each story needs part_1_narration, "
                    f"part_2_narration, real_talk, and segue fields.\n\n"
                    f"{news_block}\n\n"
                    f"CRITICAL: Output ONLY valid JSON with exactly {num_stories} stories.\n"
                    f"Target: 200-280 words total."
                )
                retry_response = self.generate(
                    prompt=retry_prompt,
                    model=self.task_models.get("multi_news_synthesizer", self.default_model),
                    system_prompt=prompt_config["system_prompt"],
                    temperature=prompt_config["temperature"],
                    max_tokens=prompt_config["max_tokens"],
                    task_name="script_synthesizer"
                )
                if retry_response:
                    retry_script = self._extract_json(retry_response)
                    if retry_script and isinstance(retry_script, dict):
                        retry_stories = retry_script.get('stories', [])
                        if len(retry_stories) >= num_stories:
                            script = retry_script
                            script['greeting'] = greeting
                            stories = script.get('stories', [])
                            print(f"  [VALIDATE] Retry {retry_attempt+1} success — got {len(stories)} stories")
                            break
                print(f"  [VALIDATE] Retry {retry_attempt+1} still insufficient")
            
            if len(stories) < num_stories:
                print(f"  [VALIDATE] Failed to get {num_stories} stories after 3 retries — aborting")
                return None
        
        # ══════════════════════════════════════════════════════════════
        # VALIDATION 2: Per-field repair for empty part_1/part_2/real_talk
        # ══════════════════════════════════════════════════════════════
        for i, story in enumerate(stories):
            fields_to_check = ['part_1_narration', 'part_2_narration']
            for field in fields_to_check:
                val = story.get(field, '').strip()
                if len(val.split()) < 5:
                    print(f"  [VALIDATE] Story {i+1} '{field}' is empty/weak ({len(val.split())} words) — repairing")
                    repair_prompt = (
                        f"Story {i+1} in a {num_stories}-story news script is missing its {field}.\n"
                        f"Context — part_1: {story.get('part_1_narration', 'N/A')[:100]}\n"
                        f"Context — part_2: {story.get('part_2_narration', 'N/A')[:100]}\n"
                        f"News topic: {news_analyses[min(i, len(news_analyses)-1)].get('topic', 'N/A') if news_analyses else 'geopolitics'}\n"
                        f"Key facts: {', '.join(news_analyses[min(i, len(news_analyses)-1)].get('key_facts', [])) if news_analyses else ''}\n\n"
                        f"Generate ONLY the {field} for this story. The Mask persona: chaotic, cartoony, "
                        f"Looney Tunes metaphors. Target: 35-45 words. Dense with facts.\n"
                        f"Return ONLY a JSON object with one key: \"{field}\""
                    )
                    repaired = False
                    for repair_attempt in range(2):
                        repair_response = self.generate(
                            prompt=repair_prompt,
                            temperature=0.8,
                            max_tokens=300,
                            task_name="script_synthesizer"
                        )
                        if repair_response:
                            repair_data = self._extract_json(repair_response)
                            if repair_data and field in repair_data:
                                new_val = repair_data[field].strip()
                                if len(new_val.split()) >= 10:
                                    stories[i][field] = new_val
                                    script['stories'][i][field] = new_val
                                    print(f"  [VALIDATE] Repaired story {i+1} '{field}' ({len(new_val.split())} words)")
                                    repaired = True
                                    break
                    if not repaired:
                        fallback = f"The developments in this situation continue to unfold with significant regional and global implications that demand close attention."
                        stories[i][field] = fallback
                        script['stories'][i][field] = fallback
                        print(f"  [VALIDATE] Repair failed — using fallback for story {i+1} '{field}'")
            
            # real_talk — less critical, use generic fallback if missing
            rt = story.get('real_talk', '').strip()
            if len(rt.split()) < 5:
                fallback_rt = "But here is the thing. The stakes are real. And the credits are not rolling yet."
                stories[i]['real_talk'] = fallback_rt
                script['stories'][i]['real_talk'] = fallback_rt
                print(f"  [VALIDATE] Story {i+1} real_talk fallback injected")
        
        # ══════════════════════════════════════════════════════════════
        # VALIDATION 3: Word count enforcement (130-170 words)
        # ══════════════════════════════════════════════════════════════
        MIN_WORDS = 130
        
        # Build a preliminary full_text to count words
        _prelim_parts = []
        _prelim_parts.append(script.get('greeting', greeting))
        _prelim_parts.append(script.get('intro_hook', ''))
        for s in stories:
            _prelim_parts.append(s.get('part_1_narration', ''))
            _prelim_parts.append(s.get('part_2_narration', ''))
            _prelim_parts.append(s.get('real_talk', ''))
            _prelim_parts.append(s.get('fallout', ''))
            _prelim_parts.append(s.get('segue', ''))
        _prelim_text = ' '.join(filter(None, _prelim_parts))
        _prelim_words = len(_prelim_text.split())
        
        if _prelim_words < MIN_WORDS:
            print(f"  [VALIDATE] Script too short: {_prelim_words} words (min {MIN_WORDS}) — requesting expansion")
            
            for retry_attempt in range(3):
                expand_prompt = (
                    f"The script you generated is only {_prelim_words} words. "
                    f"It MUST be 150-170 words total. Currently it is TOO SHORT.\n\n"
                    f"Current script JSON:\n{json.dumps(script, indent=2, ensure_ascii=False)[:3000]}\n\n"
                    f"EXPAND each story's part_1_narration to 18-22 words, part_2_narration to 22-28 words. "
                    f"EXPAND each real_talk to 12-16 words, fallout to 10-14 words. "
                    f"Add MORE specific facts, names, numbers, and original metaphors.\n"
                    f"Keep the SAME story topics and angles — just make them LONGER and MORE DETAILED.\n"
                    f"Target: 150-170 words total. Currently: {_prelim_words} words. Need at least {MIN_WORDS - _prelim_words} more.\n\n"
                    f"Return ONLY the corrected JSON with all {num_stories} stories expanded."
                )
                expand_response = self.generate(
                    prompt=expand_prompt,
                    model=self.task_models.get("multi_news_synthesizer", self.default_model),
                    system_prompt=prompt_config["system_prompt"],
                    temperature=prompt_config["temperature"],
                    max_tokens=prompt_config["max_tokens"],
                    task_name="script_synthesizer"
                )
                if expand_response:
                    expand_script = self._extract_json(expand_response)
                    if expand_script and isinstance(expand_script, dict):
                        expand_stories = expand_script.get('stories', [])
                        if len(expand_stories) >= num_stories:
                            # Count expanded words
                            _exp_parts = [expand_script.get('greeting', ''), expand_script.get('intro_hook', '')]
                            for s in expand_stories:
                                _exp_parts.append(s.get('part_1_narration', ''))
                                _exp_parts.append(s.get('part_2_narration', ''))
                                _exp_parts.append(s.get('real_talk', ''))
                                _exp_parts.append(s.get('fallout', ''))
                                _exp_parts.append(s.get('segue', ''))
                            _exp_words = len(' '.join(filter(None, _exp_parts)).split())
                            
                            if _exp_words >= MIN_WORDS:
                                script = expand_script
                                script['greeting'] = greeting
                                stories = script.get('stories', [])
                                print(f"  [VALIDATE] Expansion retry {retry_attempt+1} success — {_exp_words} words")
                                break
                            else:
                                print(f"  [VALIDATE] Expansion retry {retry_attempt+1}: {_exp_words} words — still short")
                                # Keep trying with updated count
                                _prelim_words = _exp_words
        
        # Final word count (always runs, regardless of whether expansion happened)
        _final_parts = [script.get('greeting', ''), script.get('intro_hook', '')]
        for s in script.get('stories', []):
            _final_parts.append(s.get('part_1_narration', ''))
            _final_parts.append(s.get('part_2_narration', ''))
            _final_parts.append(s.get('real_talk', ''))
            _final_parts.append(s.get('fallout', ''))
            _final_parts.append(s.get('segue', ''))
        _final_words = len(' '.join(filter(None, _final_parts)).split())
        print(f"  [VALIDATE] Final word count: {_final_words} words (target: {MIN_WORDS}-170)")
        
        # ═══ VALIDATION 3b: MAX_WORDS ceiling (script too long) ═══
        MAX_WORDS = 170
        if _final_words > MAX_WORDS:
            print(f"  [VALIDATE] Script too long: {_final_words} words (max {MAX_WORDS}) — requesting compression")
            
            for retry_attempt in range(3):
                compress_prompt = (
                    f"The script you generated is {_final_words} words. "
                    f"It MUST be 150-170 words total. Currently it is TOO LONG.\n\n"
                    f"Current script JSON:\n{json.dumps(script, indent=2, ensure_ascii=False)[:3000]}\n\n"
                    f"COMPRESS each story to fit 150-170 words total:\n"
                    f"- part_1_narration: trim to 18-22 words (punchy hook only)\n"
                    f"- part_2_narration: trim to 22-28 words (mechanism, facts only)\n"
                    f"- real_talk: keep as-is (already short)\n"
                    f"- fallout: keep as-is (already short)\n"
                    f"- segue: keep as-is\n\n"
                    f"Remove filler words, redundant phrases, and any sentence that doesn't add NEW information.\n"
                    f"Keep the SAME story topics and angles — just make every sentence SHORTER and SHARPER.\n"
                    f"Target: 150-170 words total. Currently: {_final_words} words. Need to cut at least {_final_words - MAX_WORDS} words.\n\n"
                    f"Return ONLY the corrected JSON with all {num_stories} stories compressed."
                )
                compress_response = self.generate(
                    prompt=compress_prompt,
                    model=self.task_models.get("multi_news_synthesizer", self.default_model),
                    system_prompt=prompt_config["system_prompt"],
                    temperature=prompt_config["temperature"],
                    max_tokens=prompt_config["max_tokens"],
                    task_name="script_synthesizer"
                )
                if compress_response:
                    compress_script = self._extract_json(compress_response)
                    if compress_script and isinstance(compress_script, dict) and compress_script.get('stories'):
                        compress_words_parts = [compress_script.get('greeting', ''), compress_script.get('intro_hook', '')]
                        for s in compress_script.get('stories', []):
                            compress_words_parts.append(s.get('part_1_narration', ''))
                            compress_words_parts.append(s.get('part_2_narration', ''))
                            compress_words_parts.append(s.get('real_talk', ''))
                            compress_words_parts.append(s.get('fallout', ''))
                            compress_words_parts.append(s.get('segue', ''))
                        compress_words = len(' '.join(filter(None, compress_words_parts)).split())
                        
                        if compress_words <= MAX_WORDS:
                            # Fidelity check: ensure compressed script preserved key entities
                            _fidelity_ok = True
                            for ci, (orig, comp) in enumerate(zip(script.get('stories', []), compress_script.get('stories', []))):
                                orig_body = f"{orig.get('part_1_narration','')} {orig.get('part_2_narration','')} {orig.get('real_talk','')} {orig.get('fallout','')}"
                                comp_body = f"{comp.get('part_1_narration','')} {comp.get('part_2_narration','')} {comp.get('real_talk','')} {comp.get('fallout','')}"
                                if orig_body and comp_body and not self._check_content_fidelity(orig_body, comp_body):
                                    print(f"  [VALIDATE] Compression retry {retry_attempt+1}: Story {ci+1} failed fidelity check — skipping")
                                    _fidelity_ok = False
                                    break
                            if not _fidelity_ok:
                                continue
                            
                            script = compress_script
                            script['greeting'] = script.get('greeting', greeting)
                            # Re-run enforcement after compression to maintain structure
                            script = self._enforce_segues(script)
                            script = self._dedup_segue_overlap(script)
                            script = self._dedup_inter_story_phrases(script)
                            script = self._enforce_fallout(script, news_analyses)
                            script = self._enforce_greeting(script)
                            stories = script.get('stories', [])
                            # Recount after enforcement
                            _ef_parts = [script.get('greeting', ''), script.get('intro_hook', '')]
                            for s in stories:
                                for f in ('part_1_narration', 'part_2_narration', 'real_talk', 'fallout', 'segue'):
                                    _ef_parts.append(s.get(f, ''))
                            _final_words = len(' '.join(filter(None, _ef_parts)).split())
                            print(f"  [VALIDATE] Compression retry {retry_attempt+1} success — {_final_words} words after enforcement")
                            _final_words = compress_words
                            break
                        else:
                            print(f"  [VALIDATE] Compression retry {retry_attempt+1}: {compress_words} words — still over {MAX_WORDS}")
            
            if _final_words > MAX_WORDS:
                print(f"  [VALIDATE] Could not compress below {MAX_WORDS} — using best available ({_final_words} words)")
        
        # ═══ VALIDATION 3c: Per-segment word count enforcement ═══
        SEGMENT_LIMITS = {
            'part_1_narration': (15, 25),
            'part_2_narration': (20, 32),
            'real_talk': (10, 18),
            'fallout': (8, 16),
            'segue': (6, 15),
            'greeting': (2, 15),
            'intro_hook': (5, 15),
        }
        for i, story in enumerate(script.get('stories', [])):
            for field, (lo, hi) in SEGMENT_LIMITS.items():
                if field in ('greeting', 'intro_hook'):
                    continue
                text = story.get(field, '')
                wc = len(text.split()) if text else 0
                if wc > hi:
                    print(f"  [VALIDATE] Story {i+1} {field}: {wc} words (max {hi}) — over segment limit")
                elif wc > 0 and wc < lo:
                    print(f"  [VALIDATE] Story {i+1} {field}: {wc} words (min {lo}) — under segment limit")
        
        # ── Build segment timeline from part_1/part_2 format ──
        # Each segment maps to an image: [segment_text, image_index]
        segment_timeline = []
        
        # Intro segment → image 0 (first story, part 1)
        intro_hook = script.get('intro_hook', '').strip()
        greeting = script.get('greeting', '').strip()
        intro_text = f"{greeting} {intro_hook}".strip() if intro_hook else greeting
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
            img_base = i * 4  # Story 0 → images 0,1,2,3; Story 1 → images 4,5,6,7
            
            # Part 1 narration → image (img_base) — THE HOOK
            part_1 = story.get('part_1_narration', '')
            if part_1:
                segment_timeline.append({
                    'text': part_1,
                    'image_idx': img_base,
                    'label': f'story_{i+1}_part1'
                })
            
            # Part 2 narration → image (img_base + 1) — THE MECHANISM
            part_2 = story.get('part_2_narration', '')
            if part_2:
                segment_timeline.append({
                    'text': part_2,
                    'image_idx': img_base + 1,
                    'label': f'story_{i+1}_part2'
                })
            
            # Real Talk → image (img_base + 2) — THE TRUTH
            real_talk = story.get('real_talk', '')
            if real_talk:
                segment_timeline.append({
                    'text': real_talk,
                    'image_idx': img_base + 2,
                    'label': f'story_{i+1}_real_talk'
                })
            
            # Fallout → image (img_base + 3) — THE FALLOUT (what happens next)
            fallout = story.get('fallout', '')
            if fallout:
                segment_timeline.append({
                    'text': fallout,
                    'image_idx': img_base + 3,
                    'label': f'story_{i+1}_fallout'
                })
            
            # SEGUE → content bridge to next story (keep same image as fallout)
            segue = story.get('segue', story.get('transition', ''))
            if segue and i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': segue,
                    'image_idx': img_base + 3,
                    'label': f'story_{i+1}_segue'
                })
            
            # Story separator (....) except after last story
            if i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': '....',
                    'image_idx': img_base + 3,
                    'label': f'story_{i+1}_separator',
                    'is_separator': True
                })
        
        # Closing → dynamic closing referencing last story topic
        stories = script.get('stories', [])
        last_story = stories[-1] if stories else {}
        last_topic = last_story.get('topic', '') or (news_analyses[-1].get('topic', '') if news_analyses else '')
        last_fallout = last_story.get('fallout', '')
        self._last_story_topic = last_topic
        self._last_fallout = last_fallout
        closing = self._build_dynamic_closing(last_fallout=last_fallout, last_topic=last_topic)
        script['closing'] = closing
        segment_timeline.append({
            'text': closing,
            'image_idx': (len(script['stories']) - 1) * 4 + 3,
            'label': 'closing'
        })
        
        # ═══ VALIDATION 3d: Merge broken-off trailing fragments ═══
        # LLM sometimes produces fallout/segue that is a grammatical continuation
        # of the previous segment rather than a standalone thought. Detect these
        # short fragments and merge them back into the preceding segment.
        CONTINUATION_STARTERS = {
            'and', 'but', 'or', 'so', 'yet', 'which', 'that', 'who',
            'where', 'when', 'how', 'while', 'because', 'since', 'until',
            'although', 'though', 'if', 'unless', 'whether'
        }
        
        _idx = 1  # start from 1 since segment 0 is always intro
        while _idx < len(segment_timeline):
            seg = segment_timeline[_idx]
            prev_seg = segment_timeline[_idx - 1]
            
            if seg.get('is_separator') or prev_seg.get('is_separator'):
                _idx += 1
                continue
            
            _text = seg['text'].strip()
            _prev_text = prev_seg['text'].strip()
            
            if not _text or not _prev_text:
                _idx += 1
                continue
            
            _first_word = re.sub(r'[^a-zA-Z]', '', _text.split()[0]).lower()
            _wc = len(_text.split())
            _is_fragment = _first_word in CONTINUATION_STARTERS and _wc < 5
            
            _prev_tail = re.sub(r'[^a-zA-Z]', '', _prev_text).split()[-2:]
            _curr_head = re.sub(r'[^a-zA-Z]', '', _text).split()[:2]
            _prev_tail_lower = [w.lower() for w in _prev_tail]
            _curr_head_lower = [w.lower() for w in _curr_head]
            _is_overlap = (
                len(_prev_tail_lower) > 0 and len(_curr_head_lower) > 0
                and (
                    _prev_tail_lower[-1:] == _curr_head_lower[-1:]
                    or _prev_tail_lower == _curr_head_lower
                )
            )
            
            if _is_fragment or _is_overlap:
                prev_seg['text'] = f"{_prev_text} {_text}"
                segment_timeline.pop(_idx)
                print(f"  [VALIDATE] Merged {'overlap' if _is_overlap else 'fragment'} into {prev_seg['label']}: '{_text[:50]}'")
                continue
            
            _idx += 1
        
        # Build full_text from timeline (includes segues and separators)
        full_parts = []
        for seg in segment_timeline:
            full_parts.append(seg['text'])
        full_text = ' '.join(filter(None, full_parts))
        
        script['full_text'] = full_text
        script['segment_timeline'] = segment_timeline
        
        # Extract visual prompts from stories (part_1_visual, part_2_visual, real_talk_visual, fallout_visual)
        # Fallback chain: visual field → narration-to-visual conversion → narration text
        visual_prompts = []
        for i, story in enumerate(script['stories']):
            p1_visual = story.get('part_1_visual', '')
            p2_visual = story.get('part_2_visual', '')
            rt_visual = story.get('real_talk_visual', '')
            fo_visual = story.get('fallout_visual', '')

            p1_narr = story.get('part_1_narration', '')
            p2_narr = story.get('part_2_narration', '')
            rt_narr = story.get('real_talk', '')
            fo_narr = story.get('fallout', '')

            visual_prompts.append({
                'scene': f'story_{i+1}_part1',
                'description': self._ensure_visual_prompt(p1_visual, p1_narr, 'hook')
            })
            visual_prompts.append({
                'scene': f'story_{i+1}_part2',
                'description': self._ensure_visual_prompt(p2_visual, p2_narr, 'mechanism')
            })
            visual_prompts.append({
                'scene': f'story_{i+1}_real_talk',
                'description': self._ensure_visual_prompt(rt_visual, rt_narr, 'truth')
            })
            visual_prompts.append({
                'scene': f'story_{i+1}_fallout',
                'description': self._ensure_visual_prompt(fo_visual, fo_narr, 'fallout')
            })
        script['all_visual_scenes'] = visual_prompts
        
        # ── VALIDATE VISUAL FIELDS: Rewrite narration-style prompts to composition-style ──
        for i, story in enumerate(script['stories']):
            for field, narr_field, scene_type in [
                ('part_1_visual', 'part_1_narration', 'hook'),
                ('part_2_visual', 'part_2_narration', 'mechanism'),
                ('real_talk_visual', 'real_talk', 'truth'),
                ('fallout_visual', 'fallout', 'fallout'),
            ]:
                visual = story.get(field, '').strip()
                narration = story.get(narr_field, '').strip()
                if visual and not LLMInterface._validate_visual_prompt_composition(visual):
                    rewritten = LLMInterface._ensure_visual_prompt(visual, narration, scene_type)
                    story[field] = rewritten
                    if rewritten != visual:
                        print(f"  [VISUAL] Story {i+1} {field}: narration-style → composition-style")
        
        # ENFORCE SEGUES: Guarantee every non-last story has a strong segue
        script = self._enforce_segues(script)
        
        # DEDUP SEGUE ↔ NEXT PART_1 OVERLAP
        script = self._dedup_segue_overlap(script)
        
        # DEDUP INTER-STORY PHRASES: Remove 3-gram overlap across story boundaries
        script = self._dedup_inter_story_phrases(script)
        
        # ENFORCE FALLOUT: Guarantee every story has a fallout field
        script = self._enforce_fallout(script, news_analyses)
        
        # ENFORCE GREETING: Guarantee greeting and intro_hook are non-empty
        script = self._enforce_greeting(script)
        
        # Rebuild segment_timeline after enforcement (dedup may have modified part_1, real_talk, fallout)
        segment_timeline = []
        greeting_seg = script.get('greeting', '')
        if greeting_seg:
            segment_timeline.append({
                'text': greeting_seg,
                'image_idx': -1,
                'label': 'greeting'
            })
        segment_timeline.append({
            'text': '....',
            'image_idx': -1,
            'label': 'intro_pause',
            'is_separator': True
        })
        for i, story in enumerate(script['stories']):
            img_base = i * 4
            for field, suffix, img_off in [
                ('part_1_narration', 'part1', 0),
                ('part_2_narration', 'part2', 1),
                ('real_talk', 'real_talk', 2),
                ('fallout', 'fallout', 3),
            ]:
                val = story.get(field, '')
                if val:
                    segment_timeline.append({
                        'text': val,
                        'image_idx': img_base + img_off,
                        'label': f'story_{i+1}_{suffix}'
                    })
            segue = story.get('segue', story.get('transition', ''))
            if segue and i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': segue,
                    'image_idx': img_base + 3,
                    'label': f'story_{i+1}_segue'
                })
            if i < len(script['stories']) - 1:
                segment_timeline.append({
                    'text': '....',
                    'image_idx': img_base + 3,
                    'label': f'story_{i+1}_separator',
                    'is_separator': True
                })
        
        # Rebuild full_text with enforced segues
        full_parts = [seg['text'] for seg in segment_timeline]
        full_text = ' '.join(filter(None, full_parts))
        script['full_text'] = full_text
        script['segment_timeline'] = segment_timeline
        
        # GUARANTEE GREETING: full_text MUST start with greeting
        script = self._ensure_greeting_in_fulltext(script)
        
        # SCRUB CTA FROM STORY FIELDS: prevent duplicate CTA before closing injection
        script = self._scrub_closing_from_stories(script)
        
        # Rebuild full_text after field scrubbing (story fields may have changed)
        full_parts = [seg['text'] for seg in segment_timeline]
        full_text = ' '.join(filter(None, full_parts))
        script['full_text'] = full_text
        
        # VALIDATE CLOSING: Ensure full_text ends with subscribe/CTA
        script['full_text'] = self._validate_closing(script['full_text'])
        
        # Calculate accurate word count and duration
        script['word_count'] = len(script['full_text'].split())
        script['estimated_duration'] = int(script['word_count'] / 2.5)
        
        print(f"  [MULTI-NEWS] Script: {len(script['stories'])} stories, {script['word_count']} words, ~{script['estimated_duration']}s")
        print(f"  [MULTI-NEWS] Timeline: {len(segment_timeline)} segments → {len(script['stories']) * 4} images")
        for seg in segment_timeline:
            print(f"    [{seg['label']}] → img#{seg['image_idx']}: \"{seg['text'][:50]}...\"")
        
        return script
    
    _GEO_ENTITIES = {
        "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
        "Bahrain", "Bangladesh", "Belarus", "Belgium", "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria",
        "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia",
        "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Czechia", "Denmark",
        "Ecuador", "Egypt", "El Salvador", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
        "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Guatemala", "Guinea",
        "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
        "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kosovo", "Kuwait", "Kyrgyzstan",
        "Laos", "Latvia", "Lebanon", "Libya", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
        "Mali", "Malta", "Mauritania", "Mexico", "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique",
        "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
        "North Korea", "Norway", "Oman", "Pakistan", "Palestine", "Panama", "Paraguay", "Peru", "Philippines",
        "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia",
        "Singapore", "Slovakia", "Slovenia", "Somalia", "South Africa", "South Korea", "South Sudan",
        "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
        "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Tunisia", "Turkey", "Turkmenistan", "Uganda",
        "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
        "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
        "Beijing", "Berlin", "Brussels", "Cairo", "Damascus", "Geneva", "Jerusalem", "Khartoum",
        "Kiev", "Kyiv", "London", "Minsk", "Moscow", "New Delhi", "Paris", "Riyadh", "Seoul",
        "Taipei", "Tallinn", "Tehran", "Tel Aviv", "Tokyo", "Tripoli", "Vienna", "Warsaw", "Washington",
        "Brussels", "Dubai", "Gaza", "Hong Kong", "West Bank",
        "Baltic Sea", "Black Sea", "Mediterranean", "Persian Gulf", "Red Sea", "South China Sea",
        "Suez Canal", "Taiwan Strait", "Strait", "Panama Canal",
        "Antarctica", "Arctic", "Balkans", "Caucasus", "Crimea", "Donbas", "Europe", "Kurdistan",
        "Latin America", "Middle East", "Sahara", "Southeast Asia",
        "Aleppo", "Hodeidah", "Idlib", "Kurdish", "Mariupol", "Odesa", "Zaporizhzhia",
    }

    _ACRONYM_PATTERN = re.compile(r'\b([A-Z]{2,}(?:\.[A-Z]{2,})*)\b')
    _PROPER_NOUN_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    _COMPOUND_PROPER_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:-[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)+)\b')
    _NUMBER_CTX_PATTERN = re.compile(
        r'(\$?\d+(?:\.\d+)?\s*(?:billion|million|thousand|trillion|percent|%))', re.IGNORECASE
    )
    _BARE_NUMBER_PATTERN = re.compile(r'\b\d+[\d,]*\b')

    _STOPWORDS = {
        'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'For', 'Not', 'Nor', 'From',
        'With', 'It', 'Its', 'Are', 'Were', 'Has', 'Have', 'Had', 'Been', 'Would', 'Could',
        'Should', 'May', 'Might', 'They', 'Their', 'There', 'Each', 'Every', 'Which',
        'What', 'When', 'Where', 'Who', 'How', 'Why', 'More', 'Most', 'Some', 'Such',
        'Than', 'Then', 'Now', 'Just', 'Also', 'Very', 'Even', 'Still', 'Only', 'About',
        'After', 'Before', 'Between', 'Through', 'During', 'Without', 'Against',
        'Another', 'While', 'Last', 'First', 'Next', 'Both', 'All', 'Many', 'Much',
        'Own', 'Other', 'New', 'Old', 'Good', 'Great', 'Big', 'Small', 'Little',
        'So', 'If', 'Or', 'An', 'No', 'Not', 'Do', 'Did', 'Get', 'Got', 'Make',
        'Made', 'Like', 'Well', 'Back', 'Over', 'Into', 'Right', 'Because',
        'Since', 'Being', 'Having', 'Doing', 'Going', 'Coming', 'Taking', 'Give',
        'Tonight', 'Today', 'Yesterday', 'Tomorrow', 'Subscribe', 'Masker',
        'Afternoon', 'Morning', 'Evening', 'Hello', 'Look', 'Here', 'Watch',
        'Imagine', 'Behind', 'Think', 'Yeah', 'Yes', 'Okay', 'Anyway',
        'I', 'You', 'We', 'He', 'She', 'Me', 'My', 'Your', 'Our',
        'By', 'Is', 'Was', 'At', 'In', 'On', 'To',
    }

    @staticmethod
    def _extract_key_entities(text: str) -> set:
        entities = set()

        for token in LLMInterface._GEO_ENTITIES:
            if token.lower() in text.lower():
                entities.add(token)

        entities.update(LLMInterface._ACRONYM_PATTERN.findall(text))
        entities.update(LLMInterface._PROPER_NOUN_PATTERN.findall(text))
        entities.update(LLMInterface._COMPOUND_PROPER_PATTERN.findall(text))
        entities.update(m.group(0) for m in LLMInterface._NUMBER_CTX_PATTERN.finditer(text))
        entities.update(LLMInterface._BARE_NUMBER_PATTERN.findall(text))

        entities -= LLMInterface._STOPWORDS

        return entities
    
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
        
        if preservation_ratio < 0.15:
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
        
        # ── EXTRACT ONLY STORY BODIES (including fallout) ──
        story_bodies = []
        for i, story in enumerate(script.get('stories', [])):
            p1 = story.get('part_1_narration', '')
            p2 = story.get('part_2_narration', '')
            rt = story.get('real_talk', '')
            fo = story.get('fallout', '')
            body = f"{p1} {p2} {rt} {fo}".strip()
            story_bodies.append(body)
        
        if len(story_bodies) < 2:
            print(f"  [CURATOR] Not enough stories to curate ({len(story_bodies)}), using original")
            return script.get('full_text', '')
        
        # Build body text with 4-part structure markers so the curator preserves them
        body_text = "\n\n---\n\n".join(
            f"[STORY {i+1}]\n[HOOK] {story.get('part_1_narration', '')}\n[MECHANISM] {story.get('part_2_narration', '')}\n[REAL_TALK] {story.get('real_talk', '')}\n[FALLOUT] {story.get('fallout', '')}"
            for i, story in enumerate(script.get('stories', []))
        )
        
        prompt = f"""Transform these {len(story_bodies)} story narrations from written text into The Mask's manic spoken language.

You receive ONLY the story narration bodies — no greeting, no segues, no closing.
Your job is ONLY to improve the rhythm and naturalness of each story's narration.

RULES:
- NEVER change facts, numbers, or country names
- NEVER add or remove information
- NEVER change catchphrases, exclamations, or Mask personality quirks — only fix rhythm and punctuation
- Break long sentences into short punchy ones — The Mask speaks in rapid-fire BURSTS
- Use PERIODS for dramatic pauses before punchlines
  Example: 'Classic leverage play. Disguised as safety.' NOT 'Classic leverage play... disguised as safety.'
- Use em-dash for abrupt cartoon contrasts
- Move key numbers to end of sentences (punch position)
- Use contractions ALWAYS (it's, they're, won't)
- Create rhythm: alternate short punchy + longer explanatory sentences
- Before every punchline/reveal, end previous sentence with a PERIOD, start punchline as new sentence
- After rhetorical questions, use a period before the answer
- Balance all stories to roughly equal word count (75-85 words each)
- Output all {len(story_bodies)} stories, separated by --- lines

    STRUCTURE AWARENESS — CRITICAL:
Each story body has 4 distinct segments that MUST be preserved separately:
- HOOK (part_1): manic energy, short punchy sentences
- MECHANISM (part_2): speed-talk the facts, connecting the dots
- REAL TALK: the mask DROPS. NO caps. NO exclamations. One flat sentence with a period.
- FALLOUT: SAME flat tone. Forward-looking. One sentence. Period at end.
The CONTRAST between manic (HOOK/MECHANISM) and flat (REAL TALK/FALLOUT) is the comedy. Do NOT flatten them together.

Output EACH story as exactly 4 lines with these EXACT markers:
[HOOK] curated hook text here
[MECHANISM] curated mechanism text here
[REAL_TALK] curated real talk text here
[FALLOUT] curated fallout text here

Separate stories with ---. No JSON. No explanations.

ORIGINAL STORY NARRATIONS:
{body_text}"""

        # ── OLLAMA ONLY: Curation contains geopolitical narration
        # that triggers GLM-5's content filter (1301). No cloud fallback.
        response = self.generate(
            prompt=prompt,
            system_prompt=prompt_config["system_prompt"],
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"],
            task_name="script_curator"
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
        
        # ── PARSE CURATED BODIES BACK INTO 4-PART STRUCTURES ──
        curated_structures = self._parse_curated_structures(curated, len(story_bodies))
        
        if not curated_structures or len(curated_structures) < len(story_bodies):
            print(f"  [CURATOR] Could not parse {len(story_bodies)} stories from response (got {len(curated_structures) if curated_structures else 0}), using original")
            return self._reassemble_script(script, story_bodies)
        
        # ── PER-STORY FIDELITY CHECK ──
        for i in range(len(story_bodies)):
            curated_body = ' '.join(v for v in curated_structures[i].values() if v)
            if not self._check_content_fidelity(story_bodies[i], curated_body):
                print(f"  [CURATOR] ⚠️ Story {i+1} failed fidelity check — using original narration")
                curated_structures[i] = {
                    'hook': script['stories'][i].get('part_1_narration', ''),
                    'mechanism': script['stories'][i].get('part_2_narration', ''),
                    'real_talk': script['stories'][i].get('real_talk', ''),
                    'fallout': script['stories'][i].get('fallout', ''),
                }
        
        # ── UPDATE STORY NARRATIONS WITH CURATED TEXT (4-PART STRUCTURE) ──
        for i, structure in enumerate(curated_structures):
            if i < len(script.get('stories', [])):
                if structure.get('hook'):
                    script['stories'][i]['part_1_narration'] = structure['hook']
                if structure.get('mechanism'):
                    script['stories'][i]['part_2_narration'] = structure['mechanism']
                if structure.get('real_talk'):
                    script['stories'][i]['real_talk'] = structure['real_talk']
                if structure.get('fallout'):
                    script['stories'][i]['fallout'] = structure['fallout']
        script['_curated'] = True

        # ── REASSEMBLE WITH STRUCTURAL ELEMENTS ──
        curated_bodies = [' '.join(v for v in s.values() if v) for s in curated_structures]
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
    
    def _parse_curated_structures(self, curated_text: str, expected_count: int) -> Optional[List[Dict[str, str]]]:
        """
        Parse curated LLM response back into 4-part structures per story.
        Expects [HOOK], [MECHANISM], [REAL_TALK], [FALLOUT] markers.
        Falls back to quarter-split if markers are absent.
        """
        import re
        
        MARKER_RE = re.compile(r'\[(HOOK|MECHANISM|REAL_TALK|FALLOUT)\]\s*', re.IGNORECASE)
        
        stories = []
        raw_stories = []
        
        # Split by [STORY N] markers or --- separators
        story_pattern = r'\[STORY\s+\d+\]\s*\n?'
        parts = re.split(story_pattern, curated_text)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) < expected_count and '---' in curated_text:
            parts = curated_text.split('---')
            parts = [re.sub(r'^\[STORY\s+\d+\]\s*\n?', '', p).strip() for p in parts if p.strip()]
        
        if not parts:
            return None
        
        for part in parts:
            # Split by markers — capturing group includes marker names in result
            # so filter them out before mapping to field names
            segments = MARKER_RE.split(part)
            marker_names = {'HOOK', 'MECHANISM', 'REAL_TALK', 'FALLOUT'}
            content_segments = [s.strip() for s in segments if s.strip() and s.strip().upper() not in marker_names]
            
            if len(content_segments) >= 4:
                structure = {
                    'hook': content_segments[0] if len(content_segments) > 0 else '',
                    'mechanism': content_segments[1] if len(content_segments) > 1 else '',
                    'real_talk': content_segments[2] if len(content_segments) > 2 else '',
                    'fallout': content_segments[3] if len(content_segments) > 3 else '',
                }
            elif len(content_segments) >= 2:
                structure = {
                    'hook': content_segments[0] if len(content_segments) > 0 else '',
                    'mechanism': content_segments[1] if len(content_segments) > 1 else '',
                }
                # Fill missing fields from quarter-split of remaining text
                remaining_text = ' '.join(s for s in content_segments if s)
                if not structure['mechanism'] and remaining_text:
                    sents = re.split(r'(?<=[.!?])\s+', remaining_text)
                    q = max(1, len(sents) // 4)
                    structure['hook'] = structure['hook'] or ' '.join(sents[:q])
                    structure['mechanism'] = structure['mechanism'] or ' '.join(sents[q:q*2])
                    structure['real_talk'] = structure['real_talk'] or ' '.join(sents[q*2:q*3])
                    structure['fallout'] = structure['fallout'] or ' '.join(sents[q*3:])
            else:
                # No markers at all: quarter-split entire text
                all_sents = re.split(r'(?<=[.!?])\s+', part)
                q = max(1, len(all_sents) // 4)
                structure = {
                    'hook': ' '.join(all_sents[:q]),
                    'mechanism': ' '.join(all_sents[q:q*2]),
                    'real_talk': ' '.join(all_sents[q*2:q*3]),
                    'fallout': ' '.join(all_sents[q*3:]),
                }
            stories.append(structure)
        
        while len(stories) < expected_count:
            stories.append({'hook': '', 'mechanism': '', 'real_talk': '', 'fallout': ''})
        
        # Convert structures back to body strings for _reassemble_script compatibility
        curated_bodies = []
        for s in stories:
            curated_bodies.append(' '.join(v for v in s.values() if v))
        
        return stories[:expected_count]
    
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
        
        # Stories with real_talk, fallout, and segues
        stories = script.get('stories', [])
        for i in range(len(story_bodies)):
            body = story_bodies[i]
            story = stories[i] if i < len(stories) else {}
            original_rt = story.get('real_talk', '')
            original_fo = story.get('fallout', '')

            # Strip real_talk from body if already present (avoid duplication)
            if original_rt and original_rt.strip() in body:
                body = body.replace(original_rt.strip(), '').strip()
                body = re.sub(r'\s*[-—]+\s*$', '', body).strip()
            
            # Strip fallout from body if already present (avoid duplication)
            if original_fo and original_fo.strip() in body:
                body = body.replace(original_fo.strip(), '').strip()
                body = re.sub(r'\s*[-—]+\s*$', '', body).strip()
            
            parts.append(body)
            if original_rt:
                parts.append(original_rt)
            if original_fo:
                parts.append(original_fo)
            
            # Add segue + separator after non-last stories
            if i < len(story_bodies) - 1:
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
        Generate dedicated visual prompts from curated narration text.
        4 scenes per story (hook, mechanism, truth, fallout).
        
        Args:
            script: Script dict with 'stories' array
            
        Returns:
            List of dict: [{'scene': 'story_N_partM', 'description': '...'}, ...]
        """
        prompt_config = self.config["prompts"].get("visual_prompt_generator")
        if not prompt_config:
            print("  [VISUAL-GEN] No visual_prompt_generator config found, skipping")
            return None
        
        stories = script.get('stories', [])
        if not stories:
            print("  [VISUAL-GEN] No stories in script, skipping")
            return None
        
        num_scenes = len(stories) * 4
        
        narration_block = ""
        for i, story in enumerate(stories, 1):
            p1 = story.get('part_1_narration', story.get('mini_hook', ''))
            p2 = story.get('part_2_narration', story.get('body', ''))
            rt = story.get('real_talk', '')
            fo = story.get('fallout', '')
            narration_block += f"""
--- story_{i}_part1 (THE HOOK for Story {i}) ---
NARRATION: "{p1}"

--- story_{i}_part2 (THE MECHANISM for Story {i}) ---
NARRATION: "{p2}"

--- story_{i}_real_talk (THE TRUTH for Story {i}) ---
NARRATION: "{rt}"

--- story_{i}_fallout (THE FALLOUT for Story {i}) ---
NARRATION: "{fo}"
"""
        
        system_prompt = prompt_config["system_prompt"]
        
        scene_mapping_lines = []
        for i, story in enumerate(stories, 1):
            scene_mapping_lines.append(f"- story_{i}_part1 → THE HOOK — visually depict what Story {i} Part 1 narration describes")
            scene_mapping_lines.append(f"- story_{i}_part2 → THE MECHANISM — visually depict what Story {i} Part 2 narration describes")
            scene_mapping_lines.append(f"- story_{i}_real_talk → THE TRUTH — visually depict the visceral consequence from Story {i} real_talk")
            scene_mapping_lines.append(f"- story_{i}_fallout → THE FALLOUT — visually depict the forward consequence from Story {i} fallout")
        scene_mapping = "\n".join(scene_mapping_lines)
        
        scene_json_entries = []
        for i, story in enumerate(stories, 1):
            scene_json_entries.append(f'    {{"scene": "story_{i}_part1", "description": "..."}}')
            scene_json_entries.append(f'    {{"scene": "story_{i}_part2", "description": "..."}}')
            scene_json_entries.append(f'    {{"scene": "story_{i}_real_talk", "description": "..."}}')
            scene_json_entries.append(f'    {{"scene": "story_{i}_fallout", "description": "..."}}')
        scene_json = ",\n".join(scene_json_entries)
        
        user_prompt = f"""You MUST generate exactly {num_scenes} visual scene descriptions. Each scene MUST depict EXACTLY what the corresponding narration says.

CRITICAL MAPPING RULES — DO NOT shuffle or rearrange:
{scene_mapping}

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
{scene_json}
  ]
}}"""
        
        # ── TRY LOCAL GEMMA 4 FIRST, FALL BACK TO GLM-5 CLOUD ──
        response = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=prompt_config["temperature"],
            max_tokens=prompt_config["max_tokens"],
            task_name="visual_prompt_generator"
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
        if not scenes or len(scenes) < num_scenes:
            print(f"  [VISUAL-GEN] Expected {num_scenes} scenes, got {len(scenes)}")
            if len(scenes) >= len(stories):
                while len(scenes) < num_scenes:
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
            scene_types = ['part1', 'part2', 'real_talk', 'fallout']
            default_name = f'story_{(i//4)+1}_{scene_types[i%4]}'
            scenes[i]['scene'] = scenes[i].get('scene', default_name)
        
        # ── DEDUPLICATION: Detect and regenerate duplicate descriptions ──
        scenes = self._deduplicate_visual_prompts(scenes, user_prompt, system_prompt)
        
        print(f"  [VISUAL-GEN] Generated {len(scenes)} visual prompts")
        for s in scenes:
            print(f"    [{s['scene']}] {s.get('description', '')[:80]}...")
        
        return scenes[:num_scenes]
    
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
                        max_tokens=300,
                        task_name="visual_prompt_generator"
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
            print(f"  [VISUAL-GEN] \u2705 All {len(scenes)} scenes are visually distinct (no duplicates)")
        
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
            max_tokens=500,
            task_name="visual_prompt_generator"
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
