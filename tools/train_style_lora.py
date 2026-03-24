"""
Train a custom style LoRA on fal.ai for permanent visual style lock.

Usage:
    python tools/train_style_lora.py <image_folder> [--trigger TRIGGER_WORD] [--steps STEPS]

Examples:
    python tools/train_style_lora.py training_data/
    python tools/train_style_lora.py training_data/ --trigger sentinel_pixel --steps 1200

Requirements:
    - FAL_KEY environment variable set
    - 15-30 high-quality reference images in the image folder
    - Images should be consistent in style (same color palette, pixel density, perspective)

Output:
    - Saves trained LoRA URL to config/custom_lora.json
    - Once saved, pixel_art_tool.py will automatically use the custom LoRA
"""

import os
import sys
import json
import zipfile
import argparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).parent.parent / "config"
CUSTOM_LORA_PATH = CONFIG_DIR / "custom_lora.json"
TRAINING_DATA_DIR = Path(__file__).parent.parent / "training_data"


def validate_images(image_folder: Path) -> list:
    """Validate training images and return list of valid paths."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    images = []
    
    for f in sorted(image_folder.iterdir()):
        if f.suffix.lower() in valid_extensions and f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb < 0.01:
                print(f"  SKIP {f.name} (too small: {size_mb:.2f}MB)")
                continue
            images.append(f)
    
    return images


def create_training_zip(images: list) -> Path:
    """Create a ZIP file from training images."""
    zip_path = Path(tempfile.mktemp(suffix='.zip'))
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, img_path.name)
    
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ZIP created: {zip_path} ({size_mb:.1f}MB, {len(images)} images)")
    return zip_path


def upload_zip_to_fal(zip_path: Path) -> str:
    """Upload ZIP to fal.ai and return the URL."""
    import fal_client
    
    print("  Uploading training data to fal.ai...")
    url = fal_client.upload_file(str(zip_path))
    print(f"  Uploaded: {url}")
    return url


def train_lora(images_url: str, trigger_word: str, steps: int) -> dict:
    """Submit LoRA training job to fal.ai and wait for completion."""
    import fal_client
    
    print(f"\n  Starting LoRA training...")
    print(f"    Trigger word: {trigger_word}")
    print(f"    Steps: {steps}")
    print(f"    Model: flux-lora-fast-training")
    print(f"    This will take ~5-15 minutes...\n")
    
    result = fal_client.subscribe(
        "fal-ai/flux-lora-fast-training",
        arguments={
            "images_data_url": images_url,
            "trigger_word": trigger_word,
            "is_style": True,
            "steps": steps,
        },
        with_logs=True,
        on_queue_update=lambda update: _handle_queue_update(update),
    )
    
    return result


def _handle_queue_update(update):
    """Handle training progress updates."""
    if hasattr(update, 'logs') and update.logs:
        for log in update.logs:
            msg = log.get('message', '') if isinstance(log, dict) else str(log)
            if msg:
                print(f"    [TRAIN] {msg}")


def save_lora_config(lora_url: str, trigger_word: str, steps: int, image_count: int):
    """Save trained LoRA config to config/custom_lora.json."""
    from datetime import datetime
    
    config = {
        "lora_url": lora_url,
        "trigger_word": trigger_word,
        "training_steps": steps,
        "training_images": image_count,
        "trained_at": datetime.now().isoformat(),
        "model": "fal-ai/flux-lora-fast-training",
        "notes": "Custom style LoRA. Delete this file to revert to default HuggingFace LoRA."
    }
    
    # Backup existing config if present
    if CUSTOM_LORA_PATH.exists():
        backup_path = CUSTOM_LORA_PATH.with_suffix('.json.bak')
        CUSTOM_LORA_PATH.rename(backup_path)
        print(f"  Previous config backed up to {backup_path.name}")
    
    CUSTOM_LORA_PATH.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f"\n  Config saved to {CUSTOM_LORA_PATH}")
    print(f"  pixel_art_tool.py will automatically use this LoRA on next run.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom style LoRA on fal.ai"
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Folder containing 15-30 reference images"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="sentinel_pixel",
        help="Trigger word to activate the LoRA (default: sentinel_pixel)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps (default: 1000, recommended: 800-1500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate images and show what would be trained without actually training"
    )
    
    args = parser.parse_args()
    
    # Validate FAL_KEY
    fal_key = os.getenv("FAL_KEY")
    if not fal_key and not args.dry_run:
        print("ERROR: FAL_KEY environment variable not set.")
        print("  Set it in your .env file or export FAL_KEY=your_key")
        sys.exit(1)
    
    if not args.dry_run:
        os.environ["FAL_KEY"] = fal_key
    
    # Validate image folder
    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        print(f"ERROR: Folder not found: {image_folder}")
        sys.exit(1)
    
    print("=" * 60)
    print("  CUSTOM STYLE LORA TRAINING")
    print("=" * 60)
    
    # Validate images
    print(f"\nScanning {image_folder}...")
    images = validate_images(image_folder)
    
    if len(images) < 10:
        print(f"\nWARNING: Only {len(images)} valid images found.")
        print("  Recommended: 15-30 images for best results.")
        print("  Minimum: 10 images.")
        if len(images) < 5:
            print("  Too few images. Aborting.")
            sys.exit(1)
    
    print(f"\n  Valid images: {len(images)}")
    for img in images:
        size_mb = img.stat().st_size / (1024 * 1024)
        print(f"    {img.name} ({size_mb:.1f}MB)")
    
    print(f"\n  Trigger word: {args.trigger}")
    print(f"  Training steps: {args.steps}")
    print(f"  Estimated cost: ~$2-5")
    
    if args.dry_run:
        print("\n  [DRY RUN] No training will be performed.")
        print("  Remove --dry-run to actually train.")
        return
    
    # Confirm
    print(f"\n  Press Enter to start training, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return
    
    # Create ZIP
    print("\nPreparing training data...")
    zip_path = create_training_zip(images)
    
    try:
        # Upload
        images_url = upload_zip_to_fal(zip_path)
        
        # Train
        result = train_lora(images_url, args.trigger, args.steps)
        
        # Extract LoRA URL from result
        lora_url = None
        if isinstance(result, dict):
            lora_url = result.get('diffusers_lora_file', {}).get('url')
            if not lora_url:
                lora_url = result.get('config_file', {}).get('url')
            if not lora_url:
                # Try to find any URL in the result
                for key, value in result.items():
                    if isinstance(value, dict) and 'url' in value:
                        lora_url = value['url']
                        break
        
        if not lora_url:
            print(f"\n  ERROR: Could not extract LoRA URL from result.")
            print(f"  Raw result: {json.dumps(result, indent=2)}")
            sys.exit(1)
        
        print(f"\n  LoRA trained successfully!")
        print(f"  URL: {lora_url}")
        
        # Save config
        save_lora_config(lora_url, args.trigger, args.steps, len(images))
        
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Your custom LoRA is ready to use.")
        print(f"  Next video generation will use: {args.trigger}")
        print(f"  To revert: delete config/custom_lora.json")
        
    finally:
        # Clean up ZIP
        if zip_path.exists():
            zip_path.unlink()


if __name__ == "__main__":
    main()
