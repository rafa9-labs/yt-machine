import os
import json
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional
from redfish.geopolitical_validator import GeopoliticalValidator

# Load environment variables from .env file
load_dotenv()

# Load image style config — single source of truth
_STYLE_CONFIG_PATH = Path(__file__).parent.parent / "config" / "image_style.json"
with open(_STYLE_CONFIG_PATH, 'r', encoding='utf-8') as _f:
    IMAGE_STYLE_CONFIG = json.load(_f)

server = FastMCP("pixel-art-tool")

FAL_KEY = os.getenv("FAL_KEY")
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PHASE 1: Model Re-integration - Pixel-Art Optimized Model Configuration
# ============================================================================

# Pixel-art optimized model endpoint (using existing FAL_KEY)
# This replaces the complex LoRA training workflow with a purpose-built pixel-art model
PIXEL_ART_MODEL = "fal-ai/flux-lora"  # LoRA-compatible model for true pixel art
PIXEL_ART_MODEL_ENDPOINT = "https://fal.run/fal-ai/flux-lora"

# Model-specific configuration for pixel-art generation with LoRA
PIXEL_ART_MODEL_CONFIG = {
    "model": "fal-ai/flux-lora",
    "api_endpoint": "https://fal.run/fal-ai/flux-lora",
    "auth_type": "bearer",
    "content_type": "application/json",
    "optimized_for": ["pixel_art", "isometric", "16bit", "retro_style"],
    "supports_reference": True,
    "default_params": {
        "image_size": "custom",
        "custom_width": 1088,
        "custom_height": 1152,
        "num_inference_steps": 20,
        "guidance_scale": 3.5,
        "enable_safety_checker": False,
        "output_format": "png"
    }
}

# Per-model inference step counts — each model has its own optimal range
MODEL_STEP_CONFIG = {
    "fal-ai/flux-lora": 20,
    "fal-ai/flux/schnell": 6,      # Schnell is architected for 4-8 steps
    "fal-ai/flux/dev": 20,
    "fal-ai/flux-pro/v1.1-ultra": 28,
}

# LoRA scale — 0.75 activates pixel art style without over-applying (NES look)
LORA_SCALE = 0.75

# Fallback chain — cheaper models only
FAL_MODEL = "fal-ai/flux-lora"
FAL_FALLBACK_MODELS = ["fal-ai/flux/schnell"]

# Prompt hash cache — avoids regenerating identical/near-identical scenes
_prompt_cache: Dict[str, str] = {}


def _check_prompt_cache(prompt: str) -> Optional[str]:
    """Check if we already generated an image for this exact prompt. Returns path or None."""
    import hashlib
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    return _prompt_cache.get(cache_key)


def _store_prompt_cache(prompt: str, image_path: str) -> None:
    """Store a prompt→image mapping in the cache."""
    import hashlib
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    _prompt_cache[cache_key] = image_path


def _upscale_pixel_art(input_path: str, render_size: tuple = None,
                       target_size: tuple = None) -> str:
    """
    Upscale a small pixel-art image using nearest-neighbor interpolation.
    This preserves hard pixel edges — the key to TRUE pixel art vs blurry faux-pixel.
    """
    # Resolve defaults at call time (RENDER_RESOLUTION defined later in module)
    if render_size is None:
        render_size = tuple(GENERATION_PARAMS.get('render_resolution', [256, 256]))
    if target_size is None:
        target_size = tuple(GENERATION_PARAMS.get('target_resolution', [1024, 1792]))
    img = Image.open(input_path)
    if img.size != render_size:
        # Resize to expected render size first (in case model output differs)
        img = img.resize(render_size, Image.NEAREST)
    # Nearest-neighbor upscale to target — keeps chunky pixels
    upscaled = img.resize(target_size, Image.NEAREST)
    upscaled.save(input_path, format='PNG')
    print(f"  [IMG] Upscaled {render_size} → {target_size} (nearest-neighbor)")
    return input_path


def _get_pixel_art_model_headers() -> Dict[str, str]:
    """
    Generate authentication headers for pixel-art optimized model.
    Uses existing FAL_KEY from environment.
    """
    if not FAL_KEY:
        raise ValueError("FAL_KEY environment variable not set")
    
    return {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json"
    }

# LoRA configuration — loaded from config, with custom LoRA override support
_CUSTOM_LORA_PATH = Path(__file__).parent.parent / "config" / "custom_lora.json"
_custom_lora = None
if _CUSTOM_LORA_PATH.exists():
    with open(_CUSTOM_LORA_PATH, 'r', encoding='utf-8') as _f:
        _custom_lora = json.load(_f)
    print(f"  [IMG] Custom LoRA loaded: {_custom_lora.get('trigger_word', 'N/A')}")

_lora_defaults = IMAGE_STYLE_CONFIG.get('lora_defaults', {})


def _resolve_lora_path(custom_lora: dict) -> str:
    """
    Resolve the effective LoRA path for fal.ai.
    fal.ai requires a URL or HuggingFace repo ID — it cannot load local files directly.
    If the stored path is a local .safetensors file, upload it to fal.ai CDN once
    and cache the returned URL back into custom_lora.json.
    """
    lora_url = custom_lora.get('lora_url', '')
    hub_url = custom_lora.get('hub_url')

    # Prefer HuggingFace Hub URL if available (permanent, fastest)
    if hub_url and hub_url.startswith('https://huggingface.co'):
        return hub_url

    # If it's already an HTTPS URL, use directly
    if lora_url.startswith('https://') or lora_url.startswith('http://'):
        return lora_url

    # If it's a HuggingFace repo ID (e.g. "username/sentinel-pixel-lora")
    if '/' in lora_url and not lora_url.startswith('/') and not ':' in lora_url:
        return lora_url

    # Local file path — upload to fal.ai CDN and cache the URL
    local_path = Path(lora_url) if lora_url else None
    if not local_path:
        local_path_str = custom_lora.get('lora_local_path', '')
        local_path = Path(local_path_str) if local_path_str else None

    if local_path and local_path.exists() and local_path.suffix == '.safetensors':
        try:
            import fal_client
            print(f"  [IMG] Uploading local LoRA to fal.ai CDN: {local_path.name}")
            cdn_url = fal_client.upload_file(str(local_path))
            print(f"  [IMG] LoRA CDN URL: {cdn_url}")
            # Cache URL back so we don't re-upload every run
            custom_lora['lora_url'] = cdn_url
            _CUSTOM_LORA_PATH.write_text(
                json.dumps(custom_lora, indent=2), encoding='utf-8'
            )
            return cdn_url
        except Exception as e:
            print(f"  [IMG] WARNING: Could not upload LoRA to CDN: {e}")
            print(f"  [IMG] Falling back to default HuggingFace LoRA")

    # Fallback to default
    return _lora_defaults.get('path', 'prithivMLmods/Retro-Pixel-Flux-LoRA')


_resolved_lora_path = (
    _resolve_lora_path(_custom_lora) if _custom_lora
    else _lora_defaults.get('path', 'prithivMLmods/Retro-Pixel-Flux-LoRA')
)

PIXEL_ART_LORA = {
    "path": _resolved_lora_path,
    "scale": _lora_defaults.get('scale', 0.85)
}

# ============================================================================
# PHASE 3: Accuracy Refinement - Strength, Noise, and Guidance Parameters
# ============================================================================

# Image-to-Image accuracy refinement configuration
# These parameters control how closely output matches reference vs prompt
I2I_REFINEMENT_CONFIG = {
    # Strength: How much to transform the reference image (0.0 = unchanged, 1.0 = completely new)
    "strength": {
        "low": 0.45,      # Subtle changes, keeps reference structure
        "medium": 0.75,   # Balanced transformation (default)
        "high": 0.95,     # Major transformation, keeps only composition
        "default": 0.75
    },
    
    # Guidance Scale: How closely to follow the text prompt (higher = more prompt adherence)
    "guidance_scale": {
        "low": 2.0,       # More creative freedom, less prompt adherence
        "medium": 3.0,    # Balanced (default)
        "high": 5.0,      # Strict prompt adherence
        "pixel_art_optimized": 3.5,  # Optimized for pixel-art detail preservation
        "default": 3.0
    },
    
    # Inference Steps: Quality vs speed tradeoff (more steps = higher quality, slower)
    "num_inference_steps": {
        "fast": 20,       # Quick generation, acceptable quality
        "balanced": 35,   # Good quality/speed balance (default)
        "quality": 50,    # Maximum quality, slower
        "pixel_art_max": 45,  # Optimized for pixel-art clarity
        "default": 35
    },
    
    # Control modes for different generation scenarios
    "control_modes": {
        "strict_reference": {
            "strength": 0.45,
            "guidance_scale": 2.5,
            "description": "Keep reference image mostly intact, subtle pixel-art styling"
        },
        "balanced": {
            "strength": 0.75,
            "guidance_scale": 3.0,
            "description": "Balanced reference influence and prompt adherence (default)"
        },
        "prompt_dominant": {
            "strength": 0.95,
            "guidance_scale": 5.0,
            "description": "Follow prompt closely, use reference for composition only"
        },
        "pixel_art_precise": {
            "strength": 0.70,
            "guidance_scale": 3.5,
            "num_inference_steps": 45,
            "description": "Optimized for pixel-art accuracy with reference guidance"
        }
    }
}


def _get_refinement_params(
    mode: str = "balanced",
    custom_strength: Optional[float] = None,
    custom_guidance: Optional[float] = None,
    custom_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get accuracy refinement parameters for Image-to-Image generation.
    
    Args:
        mode: Predefined control mode ("strict_reference", "balanced", "prompt_dominant", "pixel_art_precise")
        custom_strength: Override strength value (0.0-1.0)
        custom_guidance: Override guidance scale value
        custom_steps: Override inference steps
        
    Returns:
        Dict with strength, guidance_scale, and num_inference_steps
    """
    # Get base params from control mode
    mode_config = I2I_REFINEMENT_CONFIG["control_modes"].get(mode, I2I_REFINEMENT_CONFIG["control_modes"]["balanced"])
    
    params = {
        "strength": custom_strength if custom_strength is not None else mode_config.get("strength", I2I_REFINEMENT_CONFIG["strength"]["default"]),
        "guidance_scale": custom_guidance if custom_guidance is not None else mode_config.get("guidance_scale", I2I_REFINEMENT_CONFIG["guidance_scale"]["default"]),
        "num_inference_steps": custom_steps if custom_steps is not None else mode_config.get("num_inference_steps", I2I_REFINEMENT_CONFIG["num_inference_steps"]["default"]),
        "mode": mode,
        "mode_description": mode_config.get("description", "")
    }
    
    # Validate ranges
    params["strength"] = max(0.0, min(1.0, params["strength"]))
    params["guidance_scale"] = max(1.0, min(20.0, params["guidance_scale"]))
    params["num_inference_steps"] = max(1, min(100, params["num_inference_steps"]))
    
    return params


def _print_refinement_settings(params: Dict[str, Any]) -> None:
    """Print current refinement settings for debugging."""
    print(f"  [IMG] I2I Refinement: {params['mode']} - {params['mode_description']}")
    print(f"  [IMG]   Strength: {params['strength']:.2f} (transform amount)")
    print(f"  [IMG]   Guidance: {params['guidance_scale']:.1f} (prompt adherence)")
    print(f"  [IMG]   Steps: {params['num_inference_steps']} (quality level)")


# Build STYLE_LORAS from config
_lora_by_type = IMAGE_STYLE_CONFIG.get('lora_by_visual_type', {})
STYLE_LORAS = {}
for _vtype, _vcfg in _lora_by_type.items():
    STYLE_LORAS[_vtype] = {
        'path': _resolved_lora_path,
        'scale': _vcfg.get('scale', 0.85),
        'trigger': _custom_lora['trigger_word'] if _custom_lora else _vcfg.get('trigger', 'Retro Pixel'),
        'additional_prompts': _vcfg.get('additional_prompts', '')
    }

BRAND_COLORS = IMAGE_STYLE_CONFIG.get('brand_colors', {})

STYLE_SUFFIX = IMAGE_STYLE_CONFIG.get('style_suffix', 'Retro Pixel, (true 16-bit pixel art:1.5), (retro SNES style:1.3), isometric perspective, (hard pixel edges:1.2), limited color palette, detailed proportions, flat colors, dramatic lighting')
COLOR_PALETTE_PROMPT = IMAGE_STYLE_CONFIG.get('color_palette_prompt', '')
GENERATION_PARAMS = IMAGE_STYLE_CONFIG.get('generation_params', {})

# Render-small-then-upscale for TRUE pixel art (must be after GENERATION_PARAMS)
RENDER_RESOLUTION = tuple(GENERATION_PARAMS.get('render_resolution', [256, 256]))
TARGET_RESOLUTION = tuple(GENERATION_PARAMS.get('target_resolution', [1024, 1792]))
UPSCALE_METHOD = GENERATION_PARAMS.get('upscale_method', 'nearest')

# Phase 5.1: Negative prompt — loaded from config, sent to FAL API
NEGATIVE_PROMPT = IMAGE_STYLE_CONFIG.get('negative_prompt',
    "text, words, letters, watermark, signature, logo, UI elements, HUD, "
    "speech bubbles, captions, subtitles, labels, annotations, "
    "blurry, low quality, jpeg artifacts, pixelated noise, distorted proportions, "
    "photorealistic, 3D render, CGI, anime style, cartoon, sketch, pencil drawing, "
    "nsfw, gore, violence close-up, human faces distorted"
)

# Phase 5.1: Quality check keywords — prompt must contain enough specificity
MIN_SPECIFICITY_WORDS = 4  # Below this, prompt is flagged as too generic

# FAL.ai content-policy safe substitutions - MINIMAL LAYER
# Only replace extreme terms that actually trigger content policy violations
# flux-lora and flux/dev are permissive - most military/geopolitical terms are fine
_SAFE_SUBSTITUTIONS = [
    (r'\bgore\b',                     'aftermath'),
    (r'\bblood(?:y|ied)?\b',          'impact scene'),
    (r'\bcasualt(?:y|ies)\b',         'aftermath scene'),
    (r'\bdead bodies\b',              'aftermath scene'),
    (r'\bcorpse(?:s)?\b',             'aftermath scene'),
    (r'\bexecution(?:s)?\b',          'confrontation'),
]


def _sanitize_prompt_for_api(prompt: str) -> str:
    """
    Replace terms that trigger FAL.ai content policy with safe visual equivalents.
    Preserves the visual intent while avoiding safety checker rejections.
    """
    import re
    result = prompt
    for pattern, replacement in _SAFE_SUBSTITUTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _detect_visual_type(prompt: str) -> str:
    """
    16-category geopolitical visual type detection.
    Returns one of: warfare, naval, aerial, arms_defense, markets,
    trade_sanctions, energy, commodities, diplomacy, political,
    espionage, protests, humanitarian, border, cyber, megaprojects, general
    """
    prompt_lower = prompt.lower()

    CATEGORY_KEYWORDS = {
        'warfare': [
            'troops', 'infantry', 'ground forces', 'invasion', 'artillery',
            'tank', 'battle', 'frontline', 'combat', 'offensive', 'siege',
            'ground war', 'soldiers', 'march', 'advancing', 'retreating',
            'firefight', 'explosion', 'bombing', 'smoke', 'battlefield',
            'military', 'forces', 'strike', 'attack', 'war', 'weapon',
            'deployment', 'f-35', 'f-16', 's-400', 'ah-64', 'mq-9', 'uav'
        ],
        'naval': [
            'warship', 'destroyer', 'frigate', 'carrier', 'submarine',
            'naval', 'fleet', 'navy', 'strait', 'hormuz', 'blockade',
            'patrol boat', 'convoy', 'ship', 'vessel', 'port', 'harbor',
            'maritime', 'sealane', 'ocean', 'sea', 'flotilla'
        ],
        'aerial': [
            'aircraft', 'fighter jet', 'drone', 'airstrike', 'bomber',
            'air force', 'jet', 'plane', 'helicopter', 'sortie',
            'aerial', 'air defense', 'missile intercept', 'contrail',
            'uav', 'stealth', 'airspace', 'no-fly zone', 'sky', 'squadron'
        ],
        'arms_defense': [
            'missile defense', 'iron dome', 'patriot', 's-300', 's-400',
            'missile', 'defense system', 'radar', 'military hardware',
            'weapons expo', 'arms deal', 'defense contract', 'nuclear',
            'ballistic', 'hypersonic', 'warhead', 'arsenal', 'munitions'
        ],
        'markets': [
            'stock', 'trading floor', 'wall street', 'market crash',
            'ticker', 'traders', 'dow', 'nasdaq', 's&p', 'bear market',
            'bull market', 'portfolio', 'hedge fund', 'market panic',
            'price', 'economy', 'market', 'inflation', 'trading',
            'financial', 'economic', 'revenue', 'budget'
        ],
        'trade_sanctions': [
            'sanctions', 'tariff', 'trade war', 'embargo', 'import ban',
            'export control', 'cargo', 'container', 'shipping', 'port congestion',
            'customs', 'trade barrier', 'duties', 'supply chain', 'semiconductor ban',
            'oil ban', 'swift', 'decoupling'
        ],
        'energy': [
            'oil', 'gas', 'pipeline', 'refinery', 'lng', 'natural gas',
            'opec', 'energy', 'petroleum', 'crude', 'shale', 'fracking',
            'nuclear plant', 'power grid', 'electricity', 'fuel', 'diesel',
            'petrol', 'energy crisis', 'blackout', 'power plant'
        ],
        'commodities': [
            'wheat', 'grain', 'food shortage', 'famine', 'bread lines',
            'commodity', 'gold', 'copper', 'lithium', 'rare earth',
            'mining', 'agricultural', 'crop failure', 'drought',
            'supply shortage', 'queue', 'shelves', 'rationing'
        ],
        'diplomacy': [
            'diplomatic', 'summit', 'treaty', 'negotiation', 'agreement',
            'minister', 'ambassador', 'embassy', 'talks', 'officials',
            'foreign', 'envoy', 'delegation', 'ceasefire', 'accord',
            'handshake', 'meeting', 'peace deal', 'diplomacy', 'joint statement'
        ],
        'political': [
            'election', 'parliament', 'vote', 'president', 'prime minister',
            'congress', 'senate', 'government', 'regime change', 'coup',
            'political', 'opposition', 'rally', 'campaign', 'inauguration',
            'resignation', 'impeachment', 'ballot', 'podium', 'legislature'
        ],
        'espionage': [
            'spy', 'espionage', 'intelligence', 'surveillance', 'cia',
            'mi6', 'fsb', 'mossad', 'intercept', 'classified', 'leak',
            'whistleblower', 'cyber espionage', 'wiretap', 'agent',
            'covert', 'undercover', 'espionage', 'secret service'
        ],
        'protests': [
            'protest', 'demonstration', 'riot', 'unrest', 'march',
            'crowd', 'banners', 'signs', 'police crackdown', 'tear gas',
            'occupation', 'sit-in', 'strike', 'uprising', 'revolt',
            'civil disobedience', 'clash with police', 'barricades'
        ],
        'humanitarian': [
            'refugees', 'displaced', 'humanitarian', 'aid', 'un camp',
            'fleeing', 'evacuation', 'rescue', 'relief', 'crisis',
            'civilian', 'families', 'victims', 'shelter', 'tent city',
            'convoy', 'medical', 'hospital', 'wounded', 'children'
        ],
        'border': [
            'border', 'checkpoint', 'crossing', 'wall', 'fence',
            'frontier', 'immigration', 'migrant', 'deportation',
            'no-mans land', 'guard tower', 'territory', 'annexation',
            'demilitarized zone', 'dmz', 'buffer zone', 'border patrol'
        ],
        'cyber': [
            'cyber', 'hack', 'ransomware', 'data breach', 'server',
            'network', 'digital', 'online', 'internet shutdown',
            'firewall', 'malware', 'phishing', 'ddos', 'encryption',
            'code', 'holographic', 'virtual', 'ai-powered'
        ],
        'megaprojects': [
            'belt and road', 'canal', 'megaproject', 'construction',
            'infrastructure', 'high-speed rail', 'bridge', 'dam',
            'tunnel', 'crane', 'building', 'development project',
            'port expansion', 'airport', 'smart city', 'industrial zone'
        ],
    }

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return 'general'

    max_score = max(scores.values())
    winners = [cat for cat, s in scores.items() if s == max_score]

    if len(winners) == 1:
        return winners[0]

    # Tiebreaker: prefer more specific categories over general ones
    specificity_order = [
        'warfare', 'naval', 'aerial', 'arms_defense',
        'espionage', 'cyber', 'protests', 'humanitarian', 'border',
        'energy', 'commodities', 'trade_sanctions', 'markets',
        'diplomacy', 'political', 'megaprojects'
    ]
    for cat in specificity_order:
        if cat in winners:
            return cat

    return winners[0]


def _get_adaptive_enrichment(visual_type: str) -> str:
    """
    Get adaptive enrichment text based on visual type.
    Supports all 16 geopolitical visual categories with per-category emphasis.
    """
    enrichments = {
        'warfare': '(tactical positioning:1.3), (battlefield terrain:1.2), (smoke and fire:1.2), (military tension:1.2)',
        'naval': '(naval formation:1.3), (ocean waves:1.2), (maritime flags:1.2), (fleet maneuvers:1.1)',
        'aerial': '(altitude perspective:1.3), (contrails visible:1.2), (sky domination:1.2), (payload detail:1.1)',
        'arms_defense': '(hardware detail:1.3), (technical markings:1.2), (defense system:1.2), (national insignia:1.1)',
        'markets': '(market indicators visible:1.3), (price displays:1.2), (trading floor:1.2), (financial panic:1.1)',
        'trade_sanctions': '(cargo containers:1.3), (port bottleneck:1.2), (trade barrier:1.2), (nation flags:1.1)',
        'energy': '(industrial infrastructure:1.3), (flames and smoke:1.2), (pipeline terrain:1.2), (workers:1.1)',
        'commodities': '(resource scarcity:1.3), (supply queue:1.2), (raw materials:1.2), (shortage indicators:1.1)',
        'diplomacy': '(formal meeting setting:1.3), (official flags:1.2), (professional atmosphere:1.2), (summit table:1.1)',
        'political': '(government building:1.3), (political rally:1.2), (ballot or podium:1.2), (crowd energy:1.1)',
        'espionage': '(surveillance screens:1.3), (classified documents:1.2), (shadows and secrecy:1.2), (tech equipment:1.1)',
        'protests': '(demonstration crowd:1.3), (protest banners:1.2), (police barricades:1.2), (city backdrop:1.1)',
        'humanitarian': '(civilian perspective:1.3), (displacement crisis:1.2), (aid supplies:1.2), (distress indicators:1.1)',
        'border': '(border fence:1.3), (checkpoint guards:1.2), (vehicle queue:1.2), (terrain context:1.1)',
        'cyber': '(server infrastructure:1.3), (digital data:1.2), (warning alerts:1.2), (holographic displays:1.1)',
        'megaprojects': '(massive construction:1.3), (heavy machinery:1.2), (engineering scale:1.2), (route maps:1.1)',
        'general': '(dramatic composition:1.2), (strategic perspective:1.1), (balanced lighting:1.1), (professional atmosphere:1.1)'
    }
    
    return enrichments.get(visual_type, enrichments['general'])


# ============================================================================
# PHASE 2: Reference Image Pipeline - Image-to-Image Generation Support
# ============================================================================

def _upload_reference_image_to_fal(image_path: str) -> str:
    """
    Upload a local reference image to FAL.ai CDN and return the URL.
    Required for Image-to-Image generation with the pixel-art optimized model.
    
    Args:
        image_path: Local path to reference image file
        
    Returns:
        str: CDN URL of uploaded image
        
    Raises:
        ValueError: If image file doesn't exist or upload fails
    """
    import fal_client
    
    local_path = Path(image_path)
    if not local_path.exists():
        raise ValueError(f"Reference image not found: {image_path}")
    
    if not local_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        raise ValueError(f"Unsupported image format: {local_path.suffix}")
    
    try:
        print(f"  [IMG] Uploading reference image to FAL.ai CDN: {local_path.name}")
        cdn_url = fal_client.upload_file(str(local_path))
        print(f"  [IMG] Reference image CDN URL: {cdn_url}")
        return cdn_url
    except Exception as e:
        raise ValueError(f"Failed to upload reference image: {str(e)}")


def _validate_reference_image_url(url: str) -> bool:
    """
    Validate that a reference image URL is accessible and valid.
    
    Args:
        url: Image URL to validate
        
    Returns:
        bool: True if valid and accessible
    """
    import requests
    
    if not url or not isinstance(url, str):
        return False
    
    # Must be HTTP(S) URL
    if not url.startswith(('http://', 'https://')):
        return False
    
    try:
        # Head request to check accessibility without downloading
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '')
        
        # Must be image content type
        if not content_type.startswith('image/'):
            print(f"  [IMG] Warning: Reference URL is not an image: {content_type}")
            return False
        
        return response.status_code == 200
    except Exception as e:
        print(f"  [IMG] Warning: Cannot validate reference image: {e}")
        return False


def _prepare_reference_image(
    reference_image: Optional[str] = None,
    auto_upload: bool = True
) -> Optional[str]:
    """
    Prepare reference image for Image-to-Image generation.
    Handles both local file paths and remote URLs.
    
    Args:
        reference_image: Local path or URL to reference image
        auto_upload: If True, automatically upload local files to FAL CDN
        
    Returns:
        str: Valid image URL ready for API call, or None if invalid
    """
    if not reference_image:
        return None
    
    # Check if it's already a URL
    if reference_image.startswith(('http://', 'https://')):
        if _validate_reference_image_url(reference_image):
            return reference_image
        else:
            print(f"  [IMG] Warning: Invalid reference image URL, proceeding without reference")
            return None
    
    # It's a local file path
    if auto_upload:
        try:
            return _upload_reference_image_to_fal(reference_image)
        except Exception as e:
            print(f"  [IMG] Warning: Could not upload reference: {e}, proceeding without reference")
            return None
    else:
        # Don't upload, just validate path exists
        if Path(reference_image).exists():
            # Return as-is; caller must handle upload
            return reference_image
        else:
            print(f"  [IMG] Warning: Reference image not found: {reference_image}")
            return None


def _build_i2i_generation_args(
    prompt: str,
    reference_image_url: str,
    strength: float = 0.75,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 35,
    seed: Optional[int] = None,
    negative_prompt: str = None
) -> Dict[str, Any]:
    """
    Build API arguments for Image-to-Image generation.
    
    Args:
        prompt: Text prompt for generation
        reference_image_url: URL of reference image (must be accessible)
        strength: How much to transform reference (0.0-1.0, higher = more change)
        guidance_scale: How closely to follow prompt (higher = more adherence)
        num_inference_steps: Number of denoising steps
        seed: Optional seed for reproducibility
        negative_prompt: Optional negative prompt
        
    Returns:
        Dict of arguments for FAL API call
    """
    args = {
        "prompt": prompt,
        "image_url": reference_image_url,  # Reference image for I2I
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "image_size": {"width": GENERATION_PARAMS.get('custom_width', 1088), "height": GENERATION_PARAMS.get('custom_height', 1152)},
        "enable_safety_checker": PIXEL_ART_MODEL_CONFIG["default_params"]["enable_safety_checker"],
        "output_format": PIXEL_ART_MODEL_CONFIG["default_params"]["output_format"],
        "num_images": 1,
    }
    
    if seed is not None:
        args["seed"] = seed
    
    if negative_prompt:
        args["negative_prompt"] = negative_prompt
    
    return args


def _select_style_lora(visual_type: str) -> Dict[str, Any]:
    """
    Select appropriate LoRA configuration based on visual type.
    Returns the LoRA config to use for generation.
    """
    return STYLE_LORAS.get(visual_type, STYLE_LORAS['general'])


def _enhance_prompt_with_lora_trigger(prompt: str, visual_type: str) -> str:
    """
    Enhance prompt with LoRA-specific trigger words and additional prompts.
    """
    lora_config = _select_style_lora(visual_type)
    trigger = lora_config.get('trigger', 'Retro Pixel')
    additional = lora_config.get('additional_prompts', '')
    
    # Add trigger if not already present
    if trigger.lower() not in prompt.lower():
        prompt = f"{trigger}, {prompt}"
    
    # Add additional prompts if specified
    if additional and additional.lower() not in prompt.lower():
        prompt = f"{prompt}, {additional}"
    
    return prompt


def _score_prompt_specificity(prompt: str) -> int:
    """
    Score how specific / scene-relevant a prompt is (0-100).
    Based on presence of locations, actions, subjects, and era cues.
    Higher = more specific = better image relevance expected.
    """
    score = 0
    prompt_lower = prompt.lower()

    # Locations add 20 pts
    location_markers = [
        "strait", "gulf", "sea", "ocean", "desert", "city", "capital",
        "port", "base", "border", "valley", "mountain", "coast", "plain"
    ]
    if any(m in prompt_lower for m in location_markers):
        score += 20

    # Named geography adds 15 pts
    named_geo = [
        "hormuz", "ukraine", "gaza", "taiwan", "syria", "iran", "russia",
        "china", "israel", "korea", "persian", "red sea", "arctic"
    ]
    if any(g in prompt_lower for g in named_geo):
        score += 15

    # Action verbs add 20 pts
    actions = [
        "intercept", "launch", "strike", "deploy", "blockade", "patrol",
        "advancing", "retreating", "firing", "bombing", "crossing", "signing"
    ]
    if any(a in prompt_lower for a in actions):
        score += 20

    # Military/subject keywords add 15 pts
    military_subjects = [
        "missile", "tank", "warship", "aircraft", "soldier", "drone",
        "fighter", "destroyer", "submarine", "convoy", "carrier", "artillery"
    ]
    if any(s in prompt_lower for s in military_subjects):
        score += 15
    
    # Economic subjects add 15 pts (equal weight)
    economic_subjects = [
        "trading", "market", "price", "inflation", "shortage", "queue",
        "gas station", "stock", "economy", "dollar", "oil prices", "shelves"
    ]
    if any(s in prompt_lower for s in economic_subjects):
        score += 15
    
    # Diplomatic subjects add 15 pts (equal weight)
    diplomatic_subjects = [
        "summit", "treaty", "negotiation", "embassy", "minister",
        "diplomat", "agreement", "talks", "meeting", "officials"
    ]
    if any(s in prompt_lower for s in diplomatic_subjects):
        score += 15
    
    # Human impact subjects add 15 pts (equal weight)
    human_subjects = [
        "civilian", "family", "families", "protest", "refugee", "crowd",
        "people", "residents", "evacuees", "victims"
    ]
    if any(s in prompt_lower for s in human_subjects):
        score += 15

    # Lighting/atmosphere adds 10 pts
    atmosphere = [
        "dusk", "dawn", "night", "smoke", "explosion", "fire", "storm",
        "dramatic", "silhouette", "horizon", "sunset", "overcast"
    ]
    if any(a in prompt_lower for a in atmosphere):
        score += 10

    # Era/period cues add 10 pts
    eras = ["1980s", "1990s", "2000s", "2010s", "2020s", "cold war", "modern", "historical"]
    if any(e in prompt_lower for e in eras):
        score += 10

    # Penalise generic filler phrases
    generic = ["dramatic confrontation", "strategic forces", "strategic location"]
    for g in generic:
        if g in prompt_lower:
            score -= 15

    return max(0, min(100, score))


def _generate_placeholder(prompt: str, output_path: Path) -> None:
    width, height = 1024, 1792
    img = Image.new("RGB", (width, height), color=(10, 5, 25))
    draw = ImageDraw.Draw(img)

    grid_color = (0, 40, 80)
    spacing = 32
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    border_color = (0, 180, 255)
    draw.rectangle([8, 8, width - 8, height - 8], outline=border_color, width=3)
    draw.rectangle([16, 16, width - 16, height - 16], outline=(0, 100, 180), width=1)

    cx, cy = width // 2, height // 2
    draw.rectangle([cx - 180, cy - 180, cx + 180, cy + 180], outline=(0, 255, 180), width=2)
    draw.line([(cx - 200, cy), (cx + 200, cy)], fill=(0, 255, 180), width=1)
    draw.line([(cx, cy - 200), (cx, cy + 200)], fill=(0, 255, 180), width=1)

    for rx, ry, rw, rh, col in [
        (40, 40, 120, 60, (0, 120, 200)),
        (width - 160, 40, 120, 60, (200, 0, 120)),
        (40, height - 100, 120, 60, (0, 200, 120)),
        (width - 160, height - 100, 120, 60, (200, 120, 0)),
    ]:
        draw.rectangle([rx, ry, rx + rw, ry + rh], outline=col, width=2)

    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 22)
        font_tiny  = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = font_large
        font_tiny  = font_large

    draw.text((cx, cy - 280), "GENERIC PIXEL ART ASSET", font=font_large,
              fill=(0, 255, 255), anchor="mm")
    draw.text((cx, cy - 240), "[ PLACEHOLDER — NO FAL_KEY SET ]", font=font_small,
              fill=(255, 80, 80), anchor="mm")

    max_chars = 60
    words = prompt[:300].split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    for i, line in enumerate(lines[:6]):
        draw.text((cx, cy + 320 + i * 28), line, font=font_tiny,
                  fill=(160, 160, 200), anchor="mm")

    draw.text((cx, height - 60), "YT-MACHINE  |  SENTINEL v2.1", font=font_tiny,
              fill=(0, 100, 160), anchor="mm")

    img.save(str(output_path), format="PNG")


def _extract_visual_terms_from_narration(script_text: str) -> str:
    """
    Extract concrete visual terms from narration text for grounded enrichment.
    Scans for proper nouns, locations, equipment, and numbers — then builds
    weighted prompt tokens that reinforce what the narrator is talking about.
    Returns empty string if nothing concrete found (caller falls back to category enrichment).
    """
    if not script_text:
        return ''
    
    import re
    
    # Known geopolitical locations (partial match)
    geo_locations = [
        'strait of hormuz', 'hormuz', 'south china sea', 'taiwan strait',
        'persian gulf', 'red sea', 'suez canal', 'panama canal',
        'ukraine', 'gaza', 'syria', 'yemen', 'iran', 'israel',
        'russia', 'china', 'taiwan', 'korea', 'kiev', 'kyiv',
        'moscow', 'beijing', 'tehran', 'washington', 'brussels',
        'arctic', 'sahara', 'sahel', 'himalaya', 'indian ocean',
        'pacific', 'atlantic', 'mediterranean', 'black sea',
        'crimea', 'donbas', 'golan heights', 'west bank',
        'south ossetia', 'abkhazia', 'kashmir', 'kuril islands'
    ]
    
    # Known equipment/hardware terms
    equipment_terms = [
        'f-35', 'f-16', 'f-22', 'su-35', 'su-57', 'j-20',
        'missile', 'drone', 'tank', 'warship', 'destroyer',
        'submarine', 'carrier', 'aircraft carrier', 'frigate',
        'artillery', 'helicopter', 'fighter jet', 'bomber',
        'patrol boat', 'oil tanker', 'cargo ship', 'container ship',
        'pipeline', 'refinery', 'nuclear plant', 'power grid',
        's-400', 's-300', 'patriot', 'iron dome', 'thaad',
        'himars', 'm142', 'iskander', 'kinzhal', 'dagger',
        'reaper', 'mq-9', 'bayraktar', 'tb2', 'switchblade'
    ]
    
    # Known economic/infrastructure terms
    economic_terms = [
        'oil', 'gas', 'pipeline', 'lng', 'crude', 'petroleum',
        'wheat', 'grain', 'semiconductor', 'chip', 'rare earth',
        'lithium', 'copper', 'gold', 'uranium', 'cobalt',
        'sanctions', 'tariff', 'embargo', 'trade war', 'swift',
        'trading floor', 'stock market', 'inflation', 'price'
    ]
    
    text_lower = script_text.lower()
    found_terms = []
    
    # Extract locations (highest priority → weight 1.3)
    for loc in geo_locations:
        if loc in text_lower:
            found_terms.append((loc, 1.3))
            if len(found_terms) >= 3:
                break
    
    # Extract equipment (second priority → weight 1.2)
    for equip in equipment_terms:
        if equip in text_lower:
            found_terms.append((equip, 1.2))
            if len(found_terms) >= 5:
                break
    
    # Extract economic terms (third priority → weight 1.1)
    for econ in economic_terms:
        if econ in text_lower and not any(econ in ft[0] for ft in found_terms):
            found_terms.append((econ, 1.1))
            if len(found_terms) >= 6:
                break
    
    # Extract numbers (billions, millions, percentages)
    number_patterns = re.findall(r'\b(\d+)\s*(?:billion|million|percent|%|trillion)\b', text_lower)
    for num in number_patterns[:2]:
        found_terms.append((num, 1.1))
    
    if not found_terms:
        return ''
    
    # Build weighted prompt fragment
    # Take top 4 terms max to avoid overloading
    top_terms = found_terms[:4]
    weighted_parts = [f"({term}:{weight})" for term, weight in top_terms]
    
    return ', '.join(weighted_parts)


def _inject_geopolitical_context(prompt: str, issues: list, script_text: str) -> Optional[str]:
    """
    Inject geopolitical context into a prompt that failed validation.
    Tries to fix specific issues (missing location, missing country context, etc.)
    by extracting relevant terms from the narration text.
    Returns the enriched prompt, or None if no improvement could be made.
    """
    if not script_text:
        return None
    
    issues_text = ' '.join(issues).lower()
    enrichment_parts = []
    
    # If missing location context — extract from narration
    if any(word in issues_text for word in ['location', 'geography', 'region', 'country']):
        narration_terms = _extract_visual_terms_from_narration(script_text)
        if narration_terms:
            enrichment_parts.append(narration_terms)
    
    # If missing subject/equipment — extract key nouns from narration
    if any(word in issues_text for word in ['subject', 'equipment', 'specific', 'vague']):
        # Look for capitalized proper nouns in the narration
        import re
        proper_nouns = re.findall(r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)\b', script_text)
        # Filter common non-visual words
        skip_words = {'The', 'This', 'That', 'These', 'Those', 'They', 'Their',
                      'Masker', 'Today', 'Meanwhile', 'Actually', 'Believe', 'Simply',
                      'Welcome', 'Afternoon', 'Evening', 'Morning', 'Subscribe', 'Story'}
        visual_nouns = [n for n in proper_nouns if n not in skip_words][:3]
        if visual_nouns:
            enrichment_parts.append(', '.join(visual_nouns))
    
    if not enrichment_parts:
        return None
    
    # Inject before the style suffix
    enrichment = ', '.join(enrichment_parts)
    enriched = f"{prompt}, geopolitical context: {enrichment}"
    return enriched


@server.tool()
def generate_pixel_art(
    prompt: str, 
    script_text: str = None, 
    seed: int = None,
    reference_image: str = None,
    i2i_mode: str = "balanced",
    i2i_strength: float = None,
    i2i_guidance: float = None,
    i2i_steps: int = None,
    use_pixel_art_model: bool = True
) -> dict:
    """
    Generate a 16-bit cyberpunk pixel art image for a given scene prompt.
    Enhanced with visual type detection, structured prompting, and Image-to-Image support.
    
    PHASE 1-3 INTEGRATION:
    - Pixel-art optimized model (use_pixel_art_model=True)
    - Reference image guidance (reference_image parameter)
    - Accuracy refinement (i2i_mode, i2i_strength, i2i_guidance, i2i_steps)

    Args:
        prompt: Scene description (e.g. "Strait of Hormuz blockade at dusk")
        script_text: Original script text for geopolitical context (optional)
        seed: Optional seed for reproducible style. Use base_seed + scene_index for batch consistency.
        reference_image: Path or URL to reference image for I2I generation (optional)
        i2i_mode: Refinement mode ("strict_reference", "balanced", "prompt_dominant", "pixel_art_precise")
        i2i_strength: Override strength (0.0-1.0) - how much to transform reference
        i2i_guidance: Override guidance scale - how closely to follow prompt
        i2i_steps: Override inference steps
        use_pixel_art_model: If True, use pixel-art optimized model endpoint

    Returns:
        dict with success status, image path, and metadata including i2i_params
    """
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty"}

    # Geopolitical accuracy validation — 3-tier: "strict", "log_only", "disabled"
    VALIDATION_MODE = "log_only"
    geo_validator = GeopoliticalValidator()
    geo_validator.validation_rules['strict_mode'] = (VALIDATION_MODE == "strict")
    geo_validator.validation_rules['min_accuracy_score'] = 50  # Lower threshold — let enrichment fix it
    
    should_proceed, geo_error = geo_validator.validate_before_generation(script_text or "", prompt)
    
    if not should_proceed and VALIDATION_MODE == "strict":
        return {
            "success": False, 
            "error": f"Geopolitical accuracy validation failed: {geo_error}",
            "validation_type": "geopolitical",
            "accuracy_blocked": True
        }
    elif not should_proceed:
        print(f"  [IMG] ⚠️ Pre-generation validation: {geo_error} (proceeding anyway)")

    # Enhanced visual type detection
    visual_type = _detect_visual_type(prompt)
    print(f"  [IMG] Visual type detected: {visual_type}")

    # Phase 5.1: Score and optionally enrich low-specificity prompts
    specificity = _score_prompt_specificity(prompt)
    print(f"  [IMG] Specificity score: {specificity}/100")

    # If too generic, try narration-grounded enrichment first, then category fallback
    enriched_prompt = prompt.strip()
    if specificity < 35:
        # FIRST: Extract concrete terms from narration for grounded enrichment
        narration_enrichment = _extract_visual_terms_from_narration(script_text) if script_text else ''
        if narration_enrichment:
            enriched_prompt += f", {narration_enrichment}"
            print(f"  [IMG] Low specificity — narration-grounded enrichment applied")
        else:
            # FALLBACK: Category-based enrichment only if narration yields nothing
            enrichment = _get_adaptive_enrichment(visual_type)
            enriched_prompt += f", {enrichment}"
            print(f"  [IMG] Low specificity — {visual_type} category enrichment (no narration)")

    # Sanitize for FAL.ai content policy before building final prompt
    sanitized_prompt = _sanitize_prompt_for_api(enriched_prompt)
    
    # Enhance with LoRA trigger and additional prompts
    enhanced_prompt = _enhance_prompt_with_lora_trigger(sanitized_prompt, visual_type)
    
    # Build final prompt: style suffix + brand color palette
    full_prompt = f"{enhanced_prompt}, {STYLE_SUFFIX}"
    if COLOR_PALETTE_PROMPT:
        full_prompt = f"{full_prompt}, {COLOR_PALETTE_PROMPT}"
    
    # Final geopolitical validation — log warnings, retry enrichment if score is low
    final_geo_validation = geo_validator.validate_prompt_geopolitical_accuracy(full_prompt, script_text)
    print(f"  [IMG] Geopolitical accuracy score: {final_geo_validation['accuracy_score']}%")
    
    if not final_geo_validation['passed']:
        if VALIDATION_MODE == "strict":
            return {
                "success": False,
                "error": f"Final prompt failed geopolitical validation: {'; '.join(final_geo_validation['issues'][:2])}",
                "validation_type": "geopolitical_final",
                "accuracy_score": final_geo_validation['accuracy_score'],
                "issues": final_geo_validation['issues']
            }
        elif VALIDATION_MODE == "log_only":
            print(f"  [IMG] ⚠️ Geopolitical issues: {'; '.join(final_geo_validation['issues'][:3])}")
            # Try to inject geopolitical context as a retry
            if final_geo_validation['accuracy_score'] < 50 and script_text:
                retry_enrichment = _inject_geopolitical_context(full_prompt, final_geo_validation['issues'], script_text)
                if retry_enrichment:
                    full_prompt = retry_enrichment
                    retry_validation = geo_validator.validate_prompt_geopolitical_accuracy(full_prompt, script_text)
                    print(f"  [IMG] 🔄 Retry accuracy: {retry_validation['accuracy_score']}%")
    
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    filename = f"pixel_art_{safe_name}_{hash(prompt) % 100000}.png"
    output_path = OUTPUT_DIR / filename

    # ============================================================================
    # PHASE 2 & 3: Reference Image Preparation and Refinement Parameters
    # ============================================================================
    
    reference_image_url = None
    i2i_params = None
    
    if reference_image:
        print(f"  [IMG] Preparing reference image for I2I generation...")
        reference_image_url = _prepare_reference_image(reference_image, auto_upload=True)
        
        if reference_image_url:
            # Get refinement parameters (Phase 3)
            i2i_params = _get_refinement_params(
                mode=i2i_mode,
                custom_strength=i2i_strength,
                custom_guidance=i2i_guidance,
                custom_steps=i2i_steps
            )
            _print_refinement_settings(i2i_params)
        else:
            print(f"  [IMG] Proceeding with text-to-image generation (no reference)")

    if FAL_KEY:
        try:
            import fal_client
            import requests

            os.environ["FAL_KEY"] = FAL_KEY

            # Determine model chain based on configuration
            if use_pixel_art_model and PIXEL_ART_MODEL_CONFIG["supports_reference"] and reference_image_url:
                # Use pixel-art optimized model with I2I
                models_to_try = [PIXEL_ART_MODEL_CONFIG["model"]]
                print(f"  [IMG] Using pixel-art optimized model with Image-to-Image: {PIXEL_ART_MODEL_CONFIG['model']}")
            elif use_pixel_art_model:
                # Use pixel-art optimized model (text-to-image)
                models_to_try = [PIXEL_ART_MODEL_CONFIG["model"]] + [FAL_MODEL] + FAL_FALLBACK_MODELS
                print(f"  [IMG] Using pixel-art optimized model (T2I): {PIXEL_ART_MODEL_CONFIG['model']}")
            else:
                # Legacy fallback chain
                models_to_try = [FAL_MODEL] + FAL_FALLBACK_MODELS
                
            result = None
            model_used = None
            last_error = None
            
            for model in models_to_try:
                try:
                    print(f"  [IMG] Trying model: {model}")
                    
                    # Build arguments based on model type and I2I status
                    if reference_image_url and i2i_params:
                        # Image-to-Image generation (Phase 2 & 3)
                        arguments = _build_i2i_generation_args(
                            prompt=full_prompt,
                            reference_image_url=reference_image_url,
                            strength=i2i_params["strength"],
                            guidance_scale=i2i_params["guidance_scale"],
                            num_inference_steps=i2i_params["num_inference_steps"],
                            seed=seed,
                            negative_prompt=NEGATIVE_PROMPT
                        )
                        print(f"  [IMG] I2I params: strength={i2i_params['strength']:.2f}, guidance={i2i_params['guidance_scale']:.1f}")
                    else:
                        # Standard text-to-image generation
                        _gp = GENERATION_PARAMS
                        custom_w = _gp.get('custom_width', 1088)
                        custom_h = _gp.get('custom_height', 1152)
                        base_args = {
                            "prompt": full_prompt,
                            "negative_prompt": NEGATIVE_PROMPT,
                            "image_size": {"width": custom_w, "height": custom_h},
                            "num_images": 1,
                            "num_inference_steps": _gp.get('num_inference_steps', 28),
                            "guidance_scale": _gp.get('guidance_scale', 3.5),
                            "enable_safety_checker": _gp.get('enable_safety_checker', False),
                            "output_format": _gp.get('output_format', 'png'),
                        }
                        print(f"  [IMG] Custom dimensions: {custom_w}×{custom_h}")
                        if seed is not None:
                            base_args["seed"] = seed
                        
                        if model == "fal-ai/flux-lora":
                            lora_config = _select_style_lora(visual_type)
                            arguments = {
                                **base_args,
                                "num_inference_steps": MODEL_STEP_CONFIG.get(model, 20),
                                "loras": [{
                                    "path": lora_config['path'],
                                    "scale": LORA_SCALE
                                }],
                            }
                        else:
                            arguments = {k: v for k, v in base_args.items()
                                         if k not in ('negative_prompt', 'loras')}
                            arguments["num_inference_steps"] = MODEL_STEP_CONFIG.get(model, 20)
                    
                    result = fal_client.run(model, arguments=arguments)
                    model_used = model
                    print(f"  [IMG] ✓ Success with {model}")
                    break
                    
                except Exception as model_error:
                    last_error = str(model_error)
                    print(f"  [IMG] ✗ {model} failed: {last_error[:100]}")
                    if model == models_to_try[-1]:
                        raise
                    continue
            
            if not result:
                raise Exception(f"All models failed. Last error: {last_error}")

            image_url = result["images"][0]["url"]

            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(img_response.content)

            # Upscale with nearest-neighbor for TRUE pixel art
            try:
                _upscale_pixel_art(str(output_path))
            except Exception as upscale_err:
                print(f"  [IMG] Warning: Upscale failed, using raw output: {upscale_err}")

            # Cache the prompt→image mapping
            _store_prompt_cache(full_prompt, str(output_path))

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "specificity_score": specificity,
                "visual_type": visual_type,
                "source": model_used,
                "lora_used": model_used == "fal-ai/flux-lora" and not reference_image_url,
                "lora_config": _select_style_lora(visual_type) if model_used == "fal-ai/flux-lora" and not reference_image_url else None,
                "image_url": image_url,
                "output_directory": str(OUTPUT_DIR),
                # NEW: I2I metadata
                "i2i_used": reference_image_url is not None,
                "i2i_params": i2i_params,
                "reference_image_url": reference_image_url,
                "pixel_art_model_used": use_pixel_art_model,
                # NEW: Geopolitical accuracy metadata
                "geopolitical_validation": final_geo_validation,
                "accuracy_score": final_geo_validation['accuracy_score'],
                "countries_detected": list(final_geo_validation['country_analysis'].keys()),
                "equipment_validated": not any(analysis['issues'] for analysis in final_geo_validation['equipment_analysis'].values())
            }

        except Exception as e:
            error_msg = str(e)
            # Always fall back to placeholder on any FAL error
            print(f"⚠️  FAL.ai generation failed: {error_msg[:120]}")
            print(f"   Generating placeholder image as fallback.")
            
            _generate_placeholder(prompt, output_path)
            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "visual_type": visual_type,
                "source": "placeholder",
                "note": f"FAL.ai fallback: {error_msg[:80]}",
                "size": "1024x1792",
                "output_directory": str(OUTPUT_DIR),
                "geopolitical_validation": final_geo_validation,
                "i2i_used": reference_image_url is not None,
                "i2i_params": i2i_params
            }

    else:
        try:
            _generate_placeholder(prompt, output_path)

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "visual_type": visual_type,
                "source": "placeholder",
                "note": "FAL_KEY not set — placeholder image generated",
                "size": "1024x1792",
                "output_directory": str(OUTPUT_DIR),
                "geopolitical_validation": final_geo_validation,
                "i2i_used": False,
                "i2i_params": None
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Placeholder generation failed: {str(e)}",
            }
