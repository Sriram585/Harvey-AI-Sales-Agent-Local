"""Hardware detection and local model recommendation."""

import os
import shutil
import subprocess
from typing import Literal
from pydantic import BaseModel, Field
import psutil


class HardwareProfile(BaseModel):
    tier: Literal["standard_laptop", "high_performance"]
    recommended_model: str
    fallback_models: list[str] = Field(default_factory=list)
    ram_gb: float
    available_ram_gb: float = 0.0
    cpu_cores: int
    gpu_name: str | None = None
    vram_gb: float = 0.0
    rationale: str


def _detect_gpu() -> tuple[str | None, float]:
    """Detect GPU name and VRAM in GB without crashing on CPU-only systems."""
    # 1. Try torch.cuda if available
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                vram_gb = round(mem / (1024 ** 3), 2)
                return name, vram_gb
    except Exception:
        pass

    # 2. Fallback to nvidia-smi if binary exists
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            line = out.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                vram_mb = float(parts[1])
                vram_gb = round(vram_mb / 1024.0, 2)
                return name, vram_gb
    except Exception:
        pass

    return None, 0.0


def get_hardware_profile() -> HardwareProfile:
    """Analyze host system hardware and determine optimal local model configuration."""
    cpu_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 4
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    avail_ram_gb = round(vm.available / (1024 ** 3), 2)

    gpu_name, vram_gb = _detect_gpu()

    # Classification logic:
    # high_performance: dedicated GPU >= 8GB VRAM or system RAM >= 32GB
    # standard_laptop: CPU-only or <16GB RAM or <8GB VRAM
    if vram_gb >= 8.0 or total_ram_gb >= 32.0:
        tier = "high_performance"
        recommended_model = "qwen3:8b"
        fallback_models = ["qwen3:14b", "qwen3:4b"]
        rationale = (
            f"Detected {total_ram_gb} GB RAM and {vram_gb} GB VRAM ({gpu_name or 'System RAM'}). "
            f"High-performance tier qualifies for 8B-14B parameter models with full quantization in memory."
        )
    else:
        tier = "standard_laptop"
        if total_ram_gb >= 16.0:
            recommended_model = "qwen3:4b"
            fallback_models = ["qwen3:1.7b"]
            rationale = (
                f"Detected standard setup with {total_ram_gb} GB RAM and {vram_gb} GB VRAM. "
                "qwen3:4b recommended for balanced reasoning without memory pressure."
            )
        else:
            recommended_model = "qwen3:1.7b"
            fallback_models = ["qwen3:4b"]
            rationale = (
                f"Detected standard laptop setup with {total_ram_gb} GB RAM (CPU-only / low VRAM). "
                "qwen3:1.7b recommended to ensure low latency and negligible CPU/RAM footprint."
            )

    return HardwareProfile(
        tier=tier,
        recommended_model=recommended_model,
        fallback_models=fallback_models,
        ram_gb=total_ram_gb,
        available_ram_gb=avail_ram_gb,
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        rationale=rationale,
    )
