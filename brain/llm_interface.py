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
        
        response = response.replace('```json', '').replace('```', '')
        
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            return None
        
        json_str = response[json_start:json_end]
        
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
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
        explainer_response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        prompt_config = self.config["prompts"]["script_synthesizer"]
        
        prompt = f"""Create a viral short-form video script from this debate:

Topic: {news_summary.get('topic', '')}
Key Facts: {', '.join(news_summary.get('key_facts', []))}
Angle: {news_summary.get('angle', '')}

Skeptic's Critique: {skeptic_response.get('critique', '')}
Skeptic's Question: {skeptic_response.get('key_question', '')}

Explainer's Response: {explainer_response.get('explanation', '')}
Explainer's Analogy: {explainer_response.get('analogy', '')}

Synthesize this into a compelling 45-second script."""
        
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
            print(f"Raw response: {response[:300]}")
            return None
        
        if "word_count" not in script:
            def count_words(field):
                value = script.get(field, "")
                if isinstance(value, str):
                    return len(value.split())
                elif isinstance(value, list):
                    return sum(len(str(item).split()) for item in value)
                return 0
            
            total_words = count_words("hook") + count_words("body") + \
                         count_words("twist") + count_words("cta")
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
