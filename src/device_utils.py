"""Helpers for selecting and reporting the available inference device."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Optional


def resolve_device(device: Optional[str] = "auto") -> str:
    """Resolve ``auto`` to CUDA when PyTorch can use it, otherwise CPU."""
    requested = (device or "auto").strip().lower()
    try:
        import torch
    except Exception as exc:
        if requested.startswith("cuda"):
            raise RuntimeError("CUDA was selected, but PyTorch is not available in this Python environment.") from exc
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if requested in {"", "auto", "default"}:
        return "cuda:0" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            "CUDA was selected, but this PyTorch build cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build in the environment used to start the app."
        )
    return requested


def get_device_status() -> Dict[str, Any]:
    """Return PyTorch CUDA details and, when available, the NVIDIA driver view."""
    status: Dict[str, Any] = {
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "gpu_names": [],
        "gpu_memory_gib": [],
        "driver_gpus": [],
    }
    try:
        import torch

        status["torch_version"] = str(torch.__version__)
        status["cuda_version"] = torch.version.cuda
        status["cuda_available"] = bool(torch.cuda.is_available())
        if status["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                status["gpu_names"].append(str(properties.name))
                status["gpu_memory_gib"].append(round(float(properties.total_memory) / (1024**3), 1))
    except Exception as exc:
        status["torch_error"] = str(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",", maxsplit=1)]
                    if not parts or not parts[0]:
                        continue
                    memory_gib = None
                    if len(parts) > 1:
                        try:
                            memory_gib = round(float(parts[1]) / 1024, 1)
                        except ValueError:
                            pass
                    status["driver_gpus"].append({"name": parts[0], "memory_gib": memory_gib})
        except (OSError, subprocess.SubprocessError):
            pass
    return status


def available_devices(status: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return selectable devices with CUDA first when usable by PyTorch."""
    status = status or get_device_status()
    if not status.get("cuda_available"):
        return ["cpu"]
    return [f"cuda:{index}" for index in range(max(len(status.get("gpu_names", [])), 1))] + ["cpu"]
