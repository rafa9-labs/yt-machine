"""
Image Curator — validates generated pixel art images using local Ollama vision model.

Evaluates images across 5 dimensions:
  1. Topic Relevance — key subjects present
  2. Composition Grounding — elements logically placed
  3. Style Consistency — proper 16-bit pixel art
  4. Spatial Coherence — physical sense
  5. Visual Completeness — full frame utilized

Pass threshold: average >= 6.5, no single dimension < 5.0

Uses a local Ollama vision model (default: gemma4:12b) instead of cloud APIs.
Falls back to ZhipuAI GLM-4V if OLLAMA_HOST is unset and ZHIPUAI_API_KEY is set.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── Configuration ──
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "gemma4")

# Fallback: ZhipuAI cloud (used only if Ollama is unavailable)
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY", "").strip()
ZHIPUAI_VISION_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPUAI_VISION_MODEL = "glm-4v-flash"

CURATOR_CONFIG_PATH = Path(__file__).parent.parent / "config" / "system_prompts.json"

PASS_AVG_THRESHOLD = 6.5
PASS_MIN_DIMENSION = 5.0
MAX_RETRIES = 2


def _load_curator_config() -> dict:
    with open(CURATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["prompts"]["image_curator"]


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_ollama_vision(system_prompt: str, image_base64: str, visual_prompt: str) -> Optional[str]:
    if not OLLAMA_HOST:
        print("  [CURATOR] No OLLAMA_HOST configured — skipping Ollama evaluation")
        return None

    url = f"{OLLAMA_HOST}/api/chat"

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"VISUAL PROMPT this image was generated from:\n{visual_prompt}\n\nEvaluate this pixel art image against the prompt above.",
                "images": [image_base64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if content:
            print(f"  [CURATOR] Ollama ({OLLAMA_VISION_MODEL}) evaluation complete")
            return content
        print("  [CURATOR] Ollama returned empty content")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [CURATOR] Ollama not running at {OLLAMA_HOST} — trying ZhipuAI fallback")
        return None
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"  [CURATOR] Model '{OLLAMA_VISION_MODEL}' not found in Ollama — run: ollama pull {OLLAMA_VISION_MODEL}")
        else:
            print(f"  [CURATOR] Ollama API error: {e}")
        return None
    except Exception as e:
        print(f"  [CURATOR] Ollama call failed: {e}")
        return None


def _call_zhipuai(system_prompt: str, image_base64: str, visual_prompt: str) -> Optional[str]:
    if not ZHIPUAI_API_KEY:
        print("  [CURATOR] No ZHIPUAI_API_KEY set — skipping cloud evaluation")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"VISUAL PROMPT this image was generated from:\n{visual_prompt}\n\nEvaluate this pixel art image against the prompt above."
                }
            ]
        }
    ]

    payload = {
        "model": ZHIPUAI_VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {ZHIPUAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(ZHIPUAI_VISION_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            print(f"  [CURATOR] ZhipuAI ({ZHIPUAI_VISION_MODEL}) evaluation complete")
            return content
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  [CURATOR] ZhipuAI API error: {e}")
        return None
    except Exception as e:
        print(f"  [CURATOR] ZhipuAI call failed: {e}")
        return None


def _call_vision_model(system_prompt: str, image_base64: str, visual_prompt: str) -> Optional[str]:
    result = _call_ollama_vision(system_prompt, image_base64, visual_prompt)
    if result:
        return result
    return _call_zhipuai(system_prompt, image_base64, visual_prompt)


def _parse_evaluation(raw_response: str) -> Optional[dict]:
    if not raw_response:
        return None

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    json_match = re.search(r'\{[^{}]*"scores"[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    try:
        result = json.loads(cleaned)
        if "scores" in result:
            return result
    except json.JSONDecodeError:
        pass

    scores = {}
    for dim in ["topic_relevance", "composition_grounding", "style_consistency",
                 "spatial_coherence", "visual_completeness"]:
        match = re.search(rf'"{dim}"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
        if match:
            scores[dim] = float(match.group(1))

    if len(scores) >= 3:
        avg = sum(scores.values()) / len(scores)
        return {
            "scores": scores,
            "average": round(avg, 1),
            "pass": avg >= PASS_AVG_THRESHOLD,
            "issues": re.findall(r'"issues"\s*:\s*\[(.*?)\]', cleaned),
            "improvement_hint": ""
        }

    print(f"  [CURATOR] Could not parse evaluation response")
    return None


def evaluate_image(image_path: str, visual_prompt: str) -> Optional[dict]:
    config = _load_curator_config()
    image_b64 = _encode_image_base64(image_path)

    print(f"  [CURATOR] Evaluating {Path(image_path).name}...")
    raw = _call_vision_model(config["system_prompt"], image_b64, visual_prompt)

    if not raw:
        return None

    result = _parse_evaluation(raw)
    if result:
        scores = result.get("scores", {})
        avg = result.get("average", 0)
        passed = result.get("pass", False)
        status = "PASS" if passed else "FAIL"
        print(f"  [CURATOR] {status} avg={avg:.1f} | "
              f"topic={scores.get('topic_relevance', '?')} "
              f"comp={scores.get('composition_grounding', '?')} "
              f"style={scores.get('style_consistency', '?')} "
              f"spatial={scores.get('spatial_coherence', '?')} "
              f"complete={scores.get('visual_completeness', '?')}")
        if result.get("issues"):
            for issue in result["issues"]:
                print(f"  [CURATOR]   Issue: {issue}")
        if not passed and result.get("improvement_hint"):
            print(f"  [CURATOR]   Hint: {result['improvement_hint']}")
    return result


def _check_pass(evaluation: Optional[dict]) -> bool:
    if not evaluation:
        return True

    scores = evaluation.get("scores", {})
    if not scores:
        return True

    avg = sum(scores.values()) / len(scores) if scores else 0
    min_dim = min(scores.values()) if scores else 0

    return avg >= PASS_AVG_THRESHOLD and min_dim >= PASS_MIN_DIMENSION


def curate_image(
    image_path: str,
    visual_prompt: str,
    generate_fn=None,
    seed: int = 0,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, Optional[dict]]:
    evaluation = evaluate_image(image_path, visual_prompt)

    if _check_pass(evaluation):
        return image_path, evaluation

    best_path = image_path
    best_eval = evaluation
    best_avg = 0.0

    if evaluation:
        scores = evaluation.get("scores", {})
        best_avg = sum(scores.values()) / len(scores) if scores else 0.0

    current_prompt = visual_prompt

    for retry in range(1, max_retries + 1):
        if not generate_fn:
            print(f"  [CURATOR] No generate_fn provided — accepting failed image")
            break

        hint = ""
        if evaluation and evaluation.get("improvement_hint"):
            hint = evaluation["improvement_hint"]

        enriched_prompt = f"{current_prompt}. IMPORTANT: {hint}" if hint else current_prompt

        print(f"  [CURATOR] Retry {retry}/{max_retries} with enriched prompt + seed {seed + retry}")
        new_path = generate_fn(
            prompt=enriched_prompt,
            seed=seed + retry,
        )

        if not new_path or not Path(new_path).exists():
            print(f"  [CURATOR] Retry {retry} generation failed")
            continue

        evaluation = evaluate_image(new_path, enriched_prompt)

        retry_avg = 0.0
        if evaluation:
            scores = evaluation.get("scores", {})
            retry_avg = sum(scores.values()) / len(scores) if scores else 0.0

        if retry_avg > best_avg:
            best_avg = retry_avg
            best_path = new_path
            best_eval = evaluation

        if _check_pass(evaluation):
            print(f"  [CURATOR] Retry {retry} PASSED")
            return new_path, evaluation

    print(f"  [CURATOR] All retries exhausted — using best attempt (avg={best_avg:.1f})")
    return best_path, best_eval


def curate_all_images(
    image_paths: List[str],
    visual_prompts: List[dict],
    generate_fn=None,
    base_seed: int = 0,
) -> Tuple[List[str], List[Optional[dict]]]:
    final_paths = []
    all_evaluations = []

    for i, (img_path, vp) in enumerate(zip(image_paths, visual_prompts)):
        description = vp.get("description", "")
        scene = vp.get("scene", f"scene_{i}")

        print(f"\n  [CURATOR] === Image {i+1}/{len(image_paths)}: {scene} ===")

        final_path, evaluation = curate_image(
            image_path=img_path,
            visual_prompt=description,
            generate_fn=generate_fn,
            seed=base_seed + i,
        )

        final_paths.append(final_path)
        all_evaluations.append(evaluation)

    passed = sum(1 for e in (all_evaluations or []) if _check_pass(e))
    total = len(image_paths)
    print(f"\n  [CURATOR] Batch complete: {passed}/{total} passed evaluation")

    return final_paths, all_evaluations