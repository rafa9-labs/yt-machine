"""
Collect best generated images for LoRA training data.

Scans output/projects/*/images/ for generated images, displays them sorted
by project, and lets you pick favorites to copy into training_data/ folder.

Usage:
    python tools/collect_best_images.py [--output training_data]
    python tools/collect_best_images.py --list-only

Once you have 15-30 good images in training_data/, run:
    python tools/train_style_lora.py training_data/
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

PROJECTS_DIR = Path(__file__).parent.parent / "output" / "projects"
IMAGES_DIR = Path(__file__).parent.parent / "output" / "images"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "training_data"


def scan_project_images() -> list:
    """Scan all project folders for generated images."""
    images = []
    
    # Scan project folders
    if PROJECTS_DIR.exists():
        for project_dir in sorted(PROJECTS_DIR.iterdir(), reverse=True):
            img_dir = project_dir / "images"
            if img_dir.exists():
                for img_file in sorted(img_dir.iterdir()):
                    if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                        images.append({
                            'path': img_file,
                            'project': project_dir.name,
                            'scene': img_file.stem.split('_')[0] if '_' in img_file.stem else 'unknown',
                            'size_mb': img_file.stat().st_size / (1024 * 1024),
                        })
    
    # Scan flat images directory
    if IMAGES_DIR.exists():
        for img_file in sorted(IMAGES_DIR.iterdir(), reverse=True):
            if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                images.append({
                    'path': img_file,
                    'project': 'output/images',
                    'scene': 'standalone',
                    'size_mb': img_file.stat().st_size / (1024 * 1024),
                })
    
    return images


def display_images(images: list):
    """Display found images grouped by project."""
    if not images:
        print("No images found in output/projects/ or output/images/")
        return
    
    print(f"\nFound {len(images)} images total:\n")
    
    current_project = None
    for idx, img in enumerate(images):
        if img['project'] != current_project:
            current_project = img['project']
            print(f"\n  [{current_project}]")
        
        print(f"    {idx:3d}. {img['path'].name}  ({img['size_mb']:.1f}MB, scene: {img['scene']})")


def collect_images(images: list, output_dir: Path):
    """Interactive selection of images to copy to training folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(output_dir.iterdir())
    if existing:
        print(f"\n  Output folder already has {len(existing)} files.")
    
    print("\nEnter image numbers to collect (comma-separated), 'all' for everything, or 'q' to quit:")
    print("Example: 0,3,5,7,12")
    print("You can also enter ranges: 0-5,10-15")
    
    try:
        selection = input("\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return
    
    if selection.lower() == 'q':
        return
    
    # Parse selection
    indices = set()
    if selection.lower() == 'all':
        indices = set(range(len(images)))
    else:
        for part in selection.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = part.split('-')
                    indices.update(range(int(start), int(end) + 1))
                except ValueError:
                    print(f"  Invalid range: {part}")
            else:
                try:
                    indices.add(int(part))
                except ValueError:
                    print(f"  Invalid number: {part}")
    
    # Copy selected images
    copied = 0
    for idx in sorted(indices):
        if idx < 0 or idx >= len(images):
            print(f"  SKIP index {idx} (out of range)")
            continue
        
        img = images[idx]
        dst = output_dir / img['path'].name
        
        # Avoid overwriting
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(img['path'], dst)
        copied += 1
        print(f"  Copied: {img['path'].name} -> {dst.name}")
    
    total = len(list(output_dir.iterdir()))
    print(f"\n  Copied {copied} images. Total in {output_dir.name}/: {total}")
    
    if total >= 15:
        print(f"\n  You have enough images for training!")
        print(f"  Run: python tools/train_style_lora.py {output_dir}")
    elif total >= 10:
        print(f"\n  Almost enough. 15-30 images recommended for best results.")
    else:
        print(f"\n  Need more images. Aim for 15-30 total.")


def main():
    parser = argparse.ArgumentParser(description="Collect best images for LoRA training")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output folder for collected images (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available images, don't collect"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  IMAGE COLLECTOR FOR LORA TRAINING")
    print("=" * 60)
    
    images = scan_project_images()
    display_images(images)
    
    if not images:
        print("\nGenerate some videos first, then come back to collect the best images.")
        return
    
    if args.list_only:
        return
    
    output_dir = Path(args.output)
    collect_images(images, output_dir)


if __name__ == "__main__":
    main()
