"""Capability checks for the repository's local CUDA LoRA trainer."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


TRAINING_PACKAGES = (
    "torch",
    "diffusers",
    "peft",
    "accelerate",
    "transformers",
    "safetensors",
    "torchvision",
    "huggingface_hub",
)


@dataclass(frozen=True)
class LocalTrainingStatus:
    """Result of checking whether ``train_lora_local.py`` can run."""

    available: bool
    reason: str
    gpu: str | None = None
    vram_gb: float | None = None
    missing_packages: tuple[str, ...] = ()

    def menu_label(self) -> str:
        if self.available:
            details = self.gpu or "CUDA GPU"
            if self.vram_gb is not None:
                details += f", {self.vram_gb:.1f} GB"
            return f"Train a LoRA locally ({details})"
        return f"Train a LoRA locally [unavailable: {self.reason}]"


def check_local_training(min_vram_gb: float = 16.0) -> LocalTrainingStatus:
    """Check dependencies and CUDA without importing the training stack."""
    missing = tuple(
        package
        for package in TRAINING_PACKAGES
        if importlib.util.find_spec(package) is None
    )
    if missing:
        return LocalTrainingStatus(
            available=False,
            reason="missing packages: " + ", ".join(missing),
            missing_packages=missing,
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - find_spec normally catches it
        return LocalTrainingStatus(False, f"torch could not be imported: {exc}")

    if not torch.cuda.is_available():
        return LocalTrainingStatus(
            available=False,
            reason="CUDA is unavailable; the local trainer does not support Apple MPS",
        )

    try:
        device = torch.cuda.get_device_properties(0)
        vram_gb = device.total_memory / (1024 ** 3)
        gpu = str(device.name)
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime
        return LocalTrainingStatus(False, f"CUDA device could not be inspected: {exc}")

    if vram_gb < min_vram_gb:
        return LocalTrainingStatus(
            available=False,
            reason=f"CUDA GPU has {vram_gb:.1f} GB; at least {min_vram_gb:.0f} GB is recommended",
            gpu=gpu,
            vram_gb=vram_gb,
        )

    return LocalTrainingStatus(True, "ready", gpu=gpu, vram_gb=vram_gb)
