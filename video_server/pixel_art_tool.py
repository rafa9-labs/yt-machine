import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont

server = FastMCP("pixel-art-tool")

FAL_KEY = os.getenv("FAL_KEY")
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PHASE 2.4: Brand color palette for visual consistency
BRAND_COLORS = {
    "primary": "dark navy blue (#0A1628)",
    "accent": "amber orange (#FFA500)",
    "highlight": "cyan blue (#00D4FF)",
    "neutral": "slate gray (#4A5568)"
}

STYLE_SUFFIX = "true 16-bit pixel art, retro SNES style, isometric perspective, hard pixel edges, limited color palette with dark navy blues, amber accents, and cyan highlights, realistic military equipment proportions, flat colors, dramatic lighting, NO blur, NO text"

# Phase 5.1: Negative prompt — explicit exclusions sent to the model
NEGATIVE_PROMPT = (
    "text, words, letters, watermark, signature, logo, UI elements, HUD, "
    "speech bubbles, captions, subtitles, labels, annotations, "
    "blurry, low quality, jpeg artifacts, pixelated noise, distorted proportions, "
    "photorealistic, 3D render, CGI, anime style, cartoon, sketch, pencil drawing, "
    "nsfw, gore, violence close-up, human faces distorted"
)

# Phase 5.1: Quality check keywords — prompt must contain enough specificity
MIN_SPECIFICITY_WORDS = 4  # Below this, prompt is flagged as too generic


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
    subjects = [
        "missile", "tank", "warship", "aircraft", "soldier", "drone",
        "fighter", "destroyer", "submarine", "convoy", "carrier", "artillery"
    ]
    if any(s in prompt_lower for s in subjects):
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


@server.tool()
def generate_pixel_art(prompt: str) -> dict:
    """
    Generate a 16-bit cyberpunk pixel art image for a given scene prompt.

    Args:
        prompt: Scene description (e.g. "Strait of Hormuz blockade at dusk")

    Returns:
        dict with success status, image path, and metadata
    """
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty"}

    # Phase 5.1: Score and optionally enrich low-specificity prompts
    specificity = _score_prompt_specificity(prompt)
    print(f"  [IMG] Specificity score: {specificity}/100")

    # If too generic, append a scene-grounding fallback
    enriched_prompt = prompt.strip()
    if specificity < 35:
        enriched_prompt += (
            ", aerial isometric view of military forces in dramatic confrontation, "
            "dark navy sky, amber explosion light, high tension"
        )
        print(f"  [IMG] Low specificity — enriched prompt applied")

    full_prompt = f"{enriched_prompt}, {STYLE_SUFFIX}"
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    filename = f"pixel_art_{safe_name}_{hash(prompt) % 100000}.png"
    output_path = OUTPUT_DIR / filename

    if FAL_KEY:
        try:
            import fal_client
            import requests

            os.environ["FAL_KEY"] = FAL_KEY

            result = fal_client.run(
                "fal-ai/flux-2-pro",
                arguments={
                    "prompt": full_prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "image_size": "portrait_4_3",
                    "num_images": 1,
                }
            )

            image_url = result["images"][0]["url"]

            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(img_response.content)

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "specificity_score": specificity,
                "source": "fal-ai/flux-2-pro",
                "image_url": image_url,
                "output_directory": str(OUTPUT_DIR),
            }

        except Exception as e:
            error_msg = str(e)
            if "balance" in error_msg.lower() or "locked" in error_msg.lower():
                print(f"⚠️  FAL.ai account balance exhausted. Using placeholder image.")
                print(f"   Top up at: https://fal.ai/dashboard/billing")
                
                _generate_placeholder(prompt, output_path)
                return {
                    "success": True,
                    "filename": filename,
                    "path": str(output_path),
                    "prompt_used": full_prompt,
                    "source": "placeholder",
                    "note": "FAL.ai balance exhausted - placeholder generated. Top up at fal.ai/dashboard/billing",
                    "size": "1024x1792",
                    "output_directory": str(OUTPUT_DIR),
                }
            
            return {
                "success": False,
                "error": f"Fal.ai generation failed: {error_msg}",
                "fallback_used": False,
            }

    else:
        try:
            _generate_placeholder(prompt, output_path)

            return {
                "success": True,
                "filename": filename,
                "path": str(output_path),
                "prompt_used": full_prompt,
                "source": "placeholder",
                "note": "OPENAI_API_KEY not set — placeholder image generated",
                "size": "1024x1792",
                "output_directory": str(OUTPUT_DIR),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Placeholder generation failed: {str(e)}",
            }
