"""Tests for the local CUDA-training capability gate."""

from src.video import lora_training


def test_missing_training_dependency_is_reported(monkeypatch):
    real_find_spec = lora_training.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "diffusers":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(lora_training.importlib.util, "find_spec", fake_find_spec)

    status = lora_training.check_local_training()

    assert status.available is False
    assert "diffusers" in status.missing_packages
    assert "unavailable" in status.menu_label()
