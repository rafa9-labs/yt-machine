"""
Local Flux LoRA Training — RTX 3090 / 24GB VRAM

Trains a custom style LoRA on your own GPU using HuggingFace diffusers + peft.
Designed for geopolitical pixel art style lock-in across all visual categories.

Prerequisites:
    pip install diffusers peft accelerate bitsandbytes transformers safetensors
    pip install huggingface_hub torch torchvision
    accelerate config  (run once to set up device)

Usage:
    python tools/train_lora_local.py training_data/
    python tools/train_lora_local.py training_data/ --steps 1500 --upload-to-hub
    python tools/train_lora_local.py training_data/ --rank 16 --steps 1000

Output:
    output/lora/sentinel_pixel.safetensors   (trained LoRA weights)
    config/custom_lora.json                  (auto-updated for pixel_art_tool.py)
    Optional: uploaded to HuggingFace Hub as username/sentinel-pixel-lora
"""

import os
import sys
import json
import math
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
LORA_OUTPUT_DIR = ROOT / "output" / "lora"
CONFIG_PATH = ROOT / "config" / "image_style.json"
CUSTOM_LORA_PATH = ROOT / "config" / "custom_lora.json"

# ── Constants ──────────────────────────────────────────────────────────────
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
TRIGGER_WORD = "sentinel_pixel"
DEFAULT_STEPS = 1200
DEFAULT_RANK = 16          # LoRA rank — higher = more expressive, more VRAM
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 1          # 1 image per step — safe for 24GB with Flux
DEFAULT_GRAD_ACCUM = 4     # Effective batch = 4
SAVE_EVERY = 400           # Save checkpoint every N steps
VALIDATE_EVERY = 200       # Generate validation image every N steps
MAX_GRAD_NORM = 1.0
MIXED_PRECISION = "bf16"   # RTX 3090 supports bf16 natively


# ── Dataset ────────────────────────────────────────────────────────────────

def _build_dataset(data_dir: Path, tokenizer, tokenizer_2, image_size: int = 768):
    """
    Build a PyTorch dataset from image/caption pairs in data_dir.
    Expects: image_001.png + image_001.txt, image_002.png + image_002.txt, ...
    """
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    class StyleDataset(Dataset):
        def __init__(self, data_dir: Path, tokenizer, tokenizer_2, size: int):
            self.pairs = []
            for img_path in sorted(data_dir.glob("*.png")):
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    caption = txt_path.read_text(encoding="utf-8").strip()
                    # Prepend trigger word to every caption
                    full_caption = f"{TRIGGER_WORD}, {caption}"
                    self.pairs.append((img_path, full_caption))

            if not self.pairs:
                raise ValueError(f"No image/caption pairs found in {data_dir}. "
                                 f"Expected image_NNN.png + image_NNN.txt pairs.")

            self.tokenizer = tokenizer
            self.tokenizer_2 = tokenizer_2
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=0.1),  # Minimal flip — preserve directionality
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, caption = self.pairs[idx]
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)

            # Tokenize with both CLIP tokenizers (Flux uses dual encoder)
            tokens_1 = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokens_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "pixel_values": pixel_values,
                "input_ids": tokens_1.input_ids.squeeze(0),
                "input_ids_2": tokens_2.input_ids.squeeze(0),
                "caption": caption,
            }

    return StyleDataset(data_dir, tokenizer, tokenizer_2, image_size)


# ── LoRA setup ─────────────────────────────────────────────────────────────

def _apply_lora(transformer, rank: int, alpha: float):
    """
    Apply LoRA adapters to the Flux transformer attention layers.
    Targets Q/K/V projection and output projection in all attention blocks.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Self-attention
            "add_q_proj", "add_k_proj", "add_v_proj",  # Cross-attention (Flux MM-DiT)
            "ff.net.0.proj", "ff.net.2",              # MLP layers (optional, improves style)
        ],
        lora_dropout=0.05,  # Small dropout to prevent overfitting
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    return transformer


# ── Training loop ──────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    output_dir: Path,
    steps: int,
    rank: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    upload_to_hub: bool,
    hub_repo: Optional[str],
    hf_token: Optional[str],
):
    """Main training function."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPTokenizer, T5TokenizerFast
    from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import compute_snr
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from safetensors.torch import save_file

    print("=" * 65)
    print("  FLUX LORA LOCAL TRAINING")
    print("=" * 65)
    print(f"  Base model    : {BASE_MODEL}")
    print(f"  Training data : {data_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Steps         : {steps}")
    print(f"  LoRA rank     : {rank}")
    print(f"  Learning rate : {lr}")
    print(f"  Batch size    : {batch_size} x {grad_accum} grad accum = {batch_size * grad_accum} effective")
    print(f"  Precision     : {MIXED_PRECISION}")
    print(f"  Trigger word  : {TRIGGER_WORD}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Accelerator ──────────────────────────────────────────────────────
    project_cfg = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=MIXED_PRECISION,
        project_config=project_cfg,
        log_with="tensorboard",
    )
    device = accelerator.device
    print(f"  Device        : {device}")
    print(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ── Load tokenizers (lightweight — no GPU needed) ────────────────────
    print("  Loading tokenizers...")
    tokenizer = CLIPTokenizer.from_pretrained(
        BASE_MODEL, subfolder="tokenizer", token=hf_token
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        BASE_MODEL, subfolder="tokenizer_2", token=hf_token
    )

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    print("  Building dataset...")
    dataset = _build_dataset(data_dir, tokenizer, tokenizer_2)
    print(f"  Found {len(dataset)} image/caption pairs")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── Load VAE (fp16 — frozen, inference only) ──────────────────────────
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL, subfolder="vae", token=hf_token, torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)

    # ── Load text encoders (frozen) ───────────────────────────────────────
    print("  Loading text encoders...")
    from transformers import CLIPTextModel, T5EncoderModel
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder", token=hf_token, torch_dtype=torch.float16
    ).to(device)
    text_encoder.requires_grad_(False)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder_2", token=hf_token, torch_dtype=torch.float16
    ).to(device)
    text_encoder_2.requires_grad_(False)

    # ── Load transformer (trainable) ──────────────────────────────────────
    print("  Loading Flux transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer", token=hf_token, torch_dtype=torch.bfloat16
    )
    transformer.requires_grad_(False)

    # ── Apply LoRA ────────────────────────────────────────────────────────
    print(f"  Applying LoRA (rank={rank}, alpha={rank * 2})...")
    transformer = _apply_lora(transformer, rank=rank, alpha=rank * 2)
    transformer = transformer.to(device)

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        BASE_MODEL, subfolder="scheduler", token=hf_token
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Cosine learning rate schedule with warmup
    warmup_steps = max(50, steps // 20)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Prepare with accelerator ──────────────────────────────────────────
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  Starting training for {steps} steps...")
    print(f"  Checkpoints every {SAVE_EVERY} steps")
    print()

    global_step = 0
    losses = []
    data_iter = iter(dataloader)

    while global_step < steps:
        transformer.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(transformer):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            # Encode text
            with torch.no_grad():
                enc_out_1 = text_encoder(
                    batch["input_ids"].to(device),
                    output_hidden_states=False,
                )
                prompt_embeds = enc_out_1.last_hidden_state.to(dtype=torch.bfloat16)

                enc_out_2 = text_encoder_2(
                    batch["input_ids_2"].to(device),
                    output_hidden_states=False,
                )
                pooled_embeds = enc_out_2.last_hidden_state.mean(dim=1).to(dtype=torch.bfloat16)

            # Sample random timestep
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()

            # Add flow matching noise
            noise = torch.randn_like(latents, dtype=torch.bfloat16)
            noisy_latents = scheduler.scale_noise(latents.to(dtype=torch.bfloat16), timesteps, noise)

            # Prepare image ids (required by Flux)
            h, w = latents.shape[2], latents.shape[3]
            image_ids = _prepare_image_ids(latents.shape[0], h // 2, w // 2, device)

            # Forward pass
            model_pred = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=timesteps,
                img_ids=image_ids,
                return_dict=False,
            )[0]

            # Flow matching loss (predict velocity field)
            target = noise - latents.to(dtype=torch.bfloat16)
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            losses.append(loss.detach().item())
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            current_lr = optimizer.param_groups[0]["lr"]

            if global_step % 50 == 0 or global_step == 1:
                print(f"  Step {global_step:4d}/{steps}  "
                      f"loss={avg_loss:.4f}  lr={current_lr:.2e}")

            # Save checkpoint
            if global_step % SAVE_EVERY == 0 or global_step == steps:
                ckpt_path = output_dir / f"checkpoint_{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                _save_lora(accelerator, transformer, ckpt_path, global_step)
                print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Save final LoRA ───────────────────────────────────────────────────
    print("\n  Saving final LoRA...")
    final_path = output_dir / "sentinel_pixel.safetensors"
    _save_lora(accelerator, transformer, output_dir, steps, final=True)
    print(f"  ✓ LoRA saved: {final_path}")

    # ── Upload to HuggingFace Hub ─────────────────────────────────────────
    hub_url = None
    if upload_to_hub and hf_token:
        print(f"\n  Uploading to HuggingFace Hub...")
        hub_url = _upload_to_hub(final_path, hub_repo, hf_token)

    # ── Update config/custom_lora.json ───────────────────────────────────
    lora_ref = hub_url if hub_url else str(final_path)
    _save_lora_config(lora_ref, hub_url, final_path, steps, len(dataset))

    # ── Done ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  LoRA file     : {final_path}")
    if hub_url:
        print(f"  Hub URL       : {hub_url}")
    print(f"  Config updated: {CUSTOM_LORA_PATH}")
    print(f"  Trigger word  : {TRIGGER_WORD}")
    print()
    print("  Next video run will automatically use your custom LoRA.")
    print("  To revert to default: delete config/custom_lora.json")
    print("=" * 65)


def _prepare_image_ids(batch_size: int, height: int, width: int, device):
    """Prepare image position IDs required by Flux's attention mechanism."""
    import torch
    ids = torch.zeros(height, width, 3, device=device)
    ids[..., 1] = ids[..., 1] + torch.arange(height, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width, device=device)[None, :]
    ids = ids.reshape(-1, 3)
    ids = ids.unsqueeze(0).repeat(batch_size, 1, 1)
    return ids


def _save_lora(accelerator, transformer, output_dir: Path, step: int, final: bool = False):
    """Extract and save only the LoRA weights (not full model)."""
    from safetensors.torch import save_file

    unwrapped = accelerator.unwrap_model(transformer)

    # Extract only LoRA parameters
    lora_state_dict = {}
    for name, param in unwrapped.named_parameters():
        if "lora_" in name or "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.cpu().clone().to(dtype=param.dtype)

    filename = "sentinel_pixel.safetensors" if final else f"lora_step_{step}.safetensors"
    save_path = output_dir / filename
    save_file(lora_state_dict, str(save_path))
    return save_path


def _upload_to_hub(lora_path: Path, repo_id: Optional[str], token: str) -> Optional[str]:
    """Upload trained LoRA to HuggingFace Hub. Returns the file URL."""
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        whoami = api.whoami(token=token)
        username = whoami["name"]

        if not repo_id:
            repo_id = f"{username}/sentinel-pixel-lora"

        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, token=token, exist_ok=True, private=False)
            print(f"  Repository: https://huggingface.co/{repo_id}")
        except Exception:
            pass

        # Upload the LoRA file
        print(f"  Uploading {lora_path.name} ({lora_path.stat().st_size / 1e6:.1f} MB)...")
        url = api.upload_file(
            path_or_fileobj=str(lora_path),
            path_in_repo=lora_path.name,
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload sentinel_pixel LoRA trained {datetime.now().strftime('%Y-%m-%d')}",
        )

        # Also upload a README
        readme_content = f"""# Sentinel Pixel LoRA

Custom style LoRA for geopolitical pixel art news videos.

## Usage

```python
loras=[{{"path": "{repo_id}", "scale": 0.85}}]
```

## Trigger Word

`{TRIGGER_WORD}`

## Style

Isometric 16-bit pixel art with dark navy blue (#0A1628), amber orange (#FFA500), 
cyan blue (#00D4FF) color palette. Optimised for military, economic, diplomatic,
and geopolitical news imagery.

## Training

- Base model: {BASE_MODEL}
- Training images: diverse geopolitical scenes across 8 categories
- Framework: HuggingFace diffusers + peft
- Trained: {datetime.now().strftime('%Y-%m-%d')}
"""
        readme_path = lora_path.parent / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )

        hub_url = f"https://huggingface.co/{repo_id}/resolve/main/{lora_path.name}"
        print(f"  ✓ Uploaded: {hub_url}")
        return hub_url

    except Exception as e:
        print(f"  ✗ Hub upload failed: {e}")
        print(f"    LoRA still saved locally: {lora_path}")
        return None


def _save_lora_config(lora_ref: str, hub_url: Optional[str], local_path: Path,
                      steps: int, image_count: int):
    """Update config/custom_lora.json so pixel_art_tool.py auto-loads the new LoRA."""
    # Backup existing
    if CUSTOM_LORA_PATH.exists():
        bak = CUSTOM_LORA_PATH.with_suffix(".json.bak")
        shutil.copy2(CUSTOM_LORA_PATH, bak)

    config = {
        "lora_url": lora_ref,
        "lora_local_path": str(local_path),
        "hub_url": hub_url,
        "trigger_word": TRIGGER_WORD,
        "training_steps": steps,
        "training_images": image_count,
        "trained_at": datetime.now().isoformat(),
        "base_model": BASE_MODEL,
        "rank": DEFAULT_RANK,
        "trained_locally": True,
        "notes": "Custom geopolitical pixel art LoRA. Delete to revert to HuggingFace default.",
    }
    CUSTOM_LORA_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _check_dependencies():
    """Check all required packages are installed before starting training."""
    missing = []
    packages = {
        "torch": "torch",
        "diffusers": "diffusers",
        "peft": "peft",
        "accelerate": "accelerate",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "torchvision": "torchvision",
        "huggingface_hub": "huggingface_hub",
    }
    for pkg, import_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print()
        print("Install all at once:")
        print("  pip install torch torchvision diffusers peft accelerate transformers safetensors huggingface_hub")
        sys.exit(1)

    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Local training requires a GPU.")
        print("  For CPU-only environments, use fal.ai training instead:")
        print("  python tools/train_style_lora.py training_data/")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"  GPU detected  : {gpu_name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 16:
        print(f"  WARNING: {vram_gb:.1f}GB VRAM detected. Flux LoRA training needs 16GB+.")
        print("  Training may OOM. Consider fal.ai training instead.")
        answer = input("  Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Train custom Flux LoRA locally on your GPU"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory with image_NNN.png + image_NNN.txt training pairs",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK,
                        help=f"LoRA rank (default: {DEFAULT_RANK}. Higher = more expressive)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--output", type=str, default=str(LORA_OUTPUT_DIR),
                        help="Output directory for trained LoRA")
    parser.add_argument("--upload-to-hub", action="store_true",
                        help="Upload trained LoRA to HuggingFace Hub after training")
    parser.add_argument("--hub-repo", type=str, default=None,
                        help="HuggingFace repo ID (e.g. username/sentinel-pixel-lora). "
                             "Auto-generated from HF username if not specified.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                        help=f"Base model to train from (default: {BASE_MODEL})")
    args = parser.parse_args()

    # Check packages and GPU before loading anything heavy
    print("  Checking dependencies...")
    _check_dependencies()

    # Get HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print()
        print("  HF_TOKEN not found in .env")
        print("  Get your token at: https://huggingface.co/settings/tokens")
        print("  Add to .env file:  HF_TOKEN=hf_xxxxxxxxxxxx")
        if args.upload_to_hub:
            print("  --upload-to-hub requires HF_TOKEN. Disabling upload.")
            args.upload_to_hub = False

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"  Run first: python tools/auto_generate_training_set.py")
        sys.exit(1)

    png_count = len(list(data_dir.glob("*.png")))
    txt_count = len(list(data_dir.glob("*.txt")))
    print(f"  Training images: {png_count} PNG, {txt_count} captions")

    if png_count < 10:
        print(f"  WARNING: Only {png_count} images found.")
        print(f"  Recommended: 60+ for diverse style training.")
        print(f"  Run: python tools/auto_generate_training_set.py")
        if png_count < 5:
            sys.exit(1)

    print()
    print(f"  Steps         : {args.steps}")
    print(f"  LoRA rank     : {args.rank}")
    print(f"  Upload to Hub : {args.upload_to_hub}")
    print()
    print("  Press Enter to start training, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("  Cancelled.")
        sys.exit(0)

    train(
        data_dir=data_dir,
        output_dir=Path(args.output),
        steps=args.steps,
        rank=args.rank,
        lr=args.lr,
        batch_size=DEFAULT_BATCH,
        grad_accum=DEFAULT_GRAD_ACCUM,
        upload_to_hub=args.upload_to_hub,
        hub_repo=args.hub_repo,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
