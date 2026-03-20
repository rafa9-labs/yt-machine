import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont

server = FastMCP("pixel-art-tool")

FAL_KEY = os.getenv("FAL_KEY")
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_SUFFIX = "true 16-bit pixel art, retro SNES style, isometric perspective, hard pixel edges, limited color palette, realistic military equipment proportions, flat colors, NO blur"

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

    full_prompt = f"{prompt.strip()}, {STYLE_SUFFIX}"
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    filename = f"pixel_art_{safe_name}_{hash(prompt) % 100000}.png"
    output_path = OUTPUT_DIR / filename

    if FAL_KEY:
        try:
            import fal_client
            import requests

            os.environ["FAL_KEY"] = FAL_KEY

            lora_prompt = full_prompt

            result = fal_client.run(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": lora_prompt,
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
                "prompt_used": lora_prompt,
                "source": "fal-ai/flux/schnell",
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
