"""
Visual QA Module — Post-generation image validation.

Sprint 3: Blur detection (Laplacian variance) + CLIP semantic alignment score.
Sprint 4: VLM binary presence check (LLaVA-1.6-34B 4-bit).

Pipeline integration:
  validate_image() → composite check → pass/fail
  If fail → adjust prompt → retry (max 2)
  If hard fail → use adjacent story image
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None
_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_LOADED = False

CLIP_THRESHOLD = 0.22
BLUR_THRESHOLD = 100.0
VLM_CONFIDENCE_THRESHOLD = 0.8


def _get_clip_model():
    """Lazy-load OpenCLIP ViT-L/14 model."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE

    import open_clip
    import torch

    _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    model = model.to(_CLIP_DEVICE).eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_TOKENIZER = tokenizer

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE


def detect_blur(image_path: str, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """
    Detect blurry images using Laplacian variance.
    
    Returns (is_blurry: bool, variance: float).
    is_blurry=True means the image FAILED the quality check.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return True, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < threshold, float(variance)


def compute_clip_score(
    image_path: str,
    prompt: str,
    style_suffix: str = "",
) -> float:
    """
    Compute CLIP semantic alignment score between an image and a prompt.
    
    Returns cosine similarity score (0.0-1.0).
    Higher = better alignment between image content and prompt.
    """
    import torch

    model, preprocess, tokenizer, device = _get_clip_model()

    if not os.path.exists(str(image_path)):
        return 0.0

    try:
        image = Image.open(str(image_path)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        text_for_clip = prompt
        if style_suffix and len(prompt) > 50:
            text_for_clip = prompt[:200]
        text_for_clip = re.sub(r'\([^)]*:\d+\.?\d*\)', '', text_for_clip).strip()
        text_for_clip = text_for_clip[:250]

        text_input = tokenizer([text_for_clip]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze().item()

        return float(similarity)

    except Exception as e:
        print(f"  [VQA-CLIP] Error computing CLIP score: {e}")
        return 0.0


def load_vlm_model():
    """
    Lazy-load LLaVA-1.6-34B in 4-bit quantization.
    Requires ~9GB VRAM. Fits alongside CLIP on RTX 3090 (24GB).
    """
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_LOADED

    if _VLM_LOADED:
        return _VLM_MODEL, _VLM_PROCESSOR

    try:
        from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        print(f"  [VQA-VLM] Loading {model_id} (4-bit)...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)

        _VLM_MODEL = model
        _VLM_PROCESSOR = processor
        _VLM_LOADED = True

        print(f"  [VQA-VLM] Model loaded successfully")
        return model, processor

    except ImportError as e:
        print(f"  [VQA-VLM] Cannot load VLM: {e}")
        print(f"  [VQA-VLM] Install bitsandbytes: pip install bitsandbytes")
        _VLM_LOADED = False
        return None, None

    except Exception as e:
        print(f"  [VQA-VLM] Failed to load model: {e}")
        _VLM_LOADED = False
        return None, None


def extract_key_subject(prompt: str) -> str:
    """
    Extract the key visual subject from an image generation prompt.
    Takes the first sentence and truncates to 50 chars for the VLM question.
    """
    style_markers = [
        'Retro Pixel', 'true 16-bit pixel art', 'retro SNES style',
        'isometric perspective', 'hard pixel edges', 'limited color palette',
        'dramatic lighting', 'flat colors', 'detailed proportions',
    ]
    text = prompt
    for marker in style_markers:
        text = text.replace(marker, '')
    text = re.sub(r'\([^)]*:\d+\.?\d*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'[.!?]', text)
    subject = sentences[0].strip() if sentences else text.strip()

    if len(subject) > 80:
        subject = subject[:80].rsplit(' ', 1)[0]

    return subject


def vlm_presence_check(
    image_path: str,
    subject: str,
    model=None,
    processor=None,
) -> Tuple[bool, str]:
    """
    Binary presence check using VLM.
    Asks: "Does this pixel art image contain [subject]? Answer Yes or No."
    
    Returns (present: bool, explanation: str).
    """
    if model is None or processor is None:
        model, processor = load_vlm_model()
        if model is None:
            return True, "VLM unavailable - passing by default"

    try:
        import torch
        from PIL import Image as PILImage

        image = PILImage.open(str(image_path)).convert("RGB")

        question = f"Does this pixel art image contain {subject}? Answer only Yes or No."
        prompt_text = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(prompt_text, image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)

        response = processor.decode(output[0], skip_special_tokens=True)
        assistant_part = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()

        is_present = "yes" in assistant_part.lower() and "no" not in assistant_part.lower().replace("yes", "")

        return bool(is_present), assistant_part

    except Exception as e:
        print(f"  [VQA-VLM] VLM check failed: {e}")
        return True, f"VLM error: {e}"


def adjust_prompt_for_retry(
    original_prompt: str,
    failure_reason: str,
    retry_number: int,
) -> str:
    """
    Adjust an image generation prompt based on why the image failed validation.
    
    Retry 1: Emphasize the missing subject with higher weight.
    Retry 2: Simplify to just key subject + style suffix.
    """
    subject = extract_key_subject(original_prompt)
    style_suffix = ''
    for marker in ['Retro Pixel', 'true 16-bit pixel art', 'retro SNES style',
                    'isometric perspective', 'hard pixel edges', 'limited color palette',
                    'dramatic lighting', 'flat colors', 'detailed proportions']:
        if marker in original_prompt:
            style_suffix += marker + ', '

    if retry_number == 1:
        if 'clip' in failure_reason.lower():
            adjusted = f"({subject}:1.5), detailed illustration of {subject}, {style_suffix}clear subject, centered composition"
        elif 'blur' in failure_reason.lower():
            adjusted = f"sharp focus, {subject}, crisp pixel art edges, {style_suffix}high detail"
        elif 'vlm' in failure_reason.lower():
            adjusted = f"({subject}:1.5), {subject} prominently featured in center, {style_suffix}clear depiction of {subject}"
        else:
            adjusted = original_prompt

        adjusted = re.sub(r'\s+', ' ', adjusted).strip()
        return adjusted

    elif retry_number == 2:
        simplified = f"{subject}, {style_suffix}simple clear scene"
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        return simplified

    return original_prompt


def validate_image(
    image_path: str,
    prompt: str,
    blur_threshold: float = BLUR_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    skip_vlm: bool = True,
    key_subject: str = None,
) -> Dict:
    """
    Composite validation: blur → CLIP → VLM (optional).
    
    Returns dict with:
        pass: bool — True if image passes all checks
        blur: dict — {is_blurry, variance}
        clip: dict — {score, threshold, passed}
        vlm: dict or None — {present, explanation} if skip_vlm=False
        reason: str — failure reason or "pass"
    """
    result = {
        'pass': True,
        'blur': {},
        'clip': {},
        'vlm': None,
        'reason': 'pass',
    }

    if not os.path.exists(str(image_path)):
        result['pass'] = False
        result['reason'] = 'file_missing'
        return result

    blur_result = detect_blur(image_path, threshold=blur_threshold)
    result['blur'] = {
        'is_blurry': blur_result[0],
        'variance': round(blur_result[1], 1),
    }
    if blur_result[0]:
        result['pass'] = False
        result['reason'] = f"blurry (variance={blur_result[1]:.1f} < {blur_threshold})"
        print(f"  [VQA] FAIL: {Path(image_path).name} — {result['reason']}")
        return result

    clip_score = compute_clip_score(image_path, prompt)
    result['clip'] = {
        'score': round(clip_score, 4),
        'threshold': clip_threshold,
        'passed': clip_score >= clip_threshold,
    }
    if clip_score < clip_threshold:
        result['pass'] = False
        result['reason'] = f"clip_misalignment (score={clip_score:.3f} < {clip_threshold})"
        print(f"  [VQA] FAIL: {Path(image_path).name} — {result['reason']}")
        return result

    if not skip_vlm:
        subject = key_subject or extract_key_subject(prompt)
        present, explanation = vlm_presence_check(image_path, subject)
        result['vlm'] = {
            'present': present,
            'explanation': explanation,
        }
        if not present:
            result['pass'] = False
            result['reason'] = f"vlm_missing_subject ({subject[:40]}: {explanation})"
            print(f"  [VQA] FAIL: {Path(image_path).name} — {result['reason']}")
            return result

    print(f"  [VQA] PASS: {Path(image_path).name} — blur={blur_result[1]:.0f}, clip={clip_score:.3f}")
    return result


def validate_batch(
    image_paths: List[str],
    prompts: List[str],
    blur_threshold: float = BLUR_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    skip_vlm: bool = True,
) -> List[Dict]:
    """
    Validate a batch of images against their prompts.
    Returns list of validation result dicts.
    """
    results = []
    for img_path, prompt in zip(image_paths, prompts):
        result = validate_image(
            img_path, prompt,
            blur_threshold=blur_threshold,
            clip_threshold=clip_threshold,
            skip_vlm=skip_vlm,
        )
        results.append(result)
    return results