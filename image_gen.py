"""
image_gen.py - Local AI Image Generation for Agent OTG

Uses the diffusers library with the "stabilityai/sd-turbo" model.
This model is lightweight and fast enough to run without a high-end GPU,
though generation will be noticeably slower on CPU-only machines.

Note on offline operation:
The model weights (approx 2-3 GB) download once on first use and require
internet access for that initial download only. After that, the pipeline
runs completely offline from the local Hugging Face cache.
"""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Any

# Cached pipeline instance at module level (loaded once)
_PIPELINE: Any = None


def _clean_path(path: str) -> str:
    """Clean quotes, whitespace, and resolve relative path into output directory."""
    if not isinstance(path, (str, os.PathLike)) or not path:
        return ""
    p = str(path).strip().strip("<>").strip("&").strip()
    if not p or "\x00" in p:
        return ""
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()

    # If it is a bare filename or relative path, save to ~/Downloads/AgentOTG or generated_files
    otg_dir = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / "AgentOTG"
    otg_dir.mkdir(parents=True, exist_ok=True)

    # Image generation is an output-only operation, so discard caller-supplied
    # directories rather than allowing writes outside the artifact folder.
    return str(otg_dir / Path(p).name)


def _get_pipeline():
    """Load and cache the SD-Turbo pipeline on GPU if available, else CPU."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    missing = [name for name in ("torch", "diffusers", "transformers", "accelerate")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Local image generation is unavailable because this Python environment is missing: "
            f"{', '.join(missing)}. Install the image dependencies from requirements.txt "
            "in the same environment used to run ask.py, then restart the app."
        )

    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        # Select torch device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            # Fallback to CPU (generation will be slower on CPU-only machines)
            device = "cpu"
            torch_dtype = torch.float32

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None,
        )
        pipe.to(device)
        _PIPELINE = pipe
        return _PIPELINE
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize image generation pipeline: {exc}. "
            "Ensure diffusers, torch, transformers, and accelerate are installed."
        ) from exc


def generate_image(prompt: str, filename: str = "generated.png") -> str:
    """
    Generate an image from a text prompt using local SD-Turbo.
    Saves the image file and returns the saved file path.
    """
    if not prompt or not str(prompt).strip():
        return "❌ Error: Missing image description prompt."

    target_path = _clean_path(filename or "generated.png")
    if not target_path:
        return "❌ Error: Invalid output filename."
    if not target_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        target_path += ".png"

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        pipe = _get_pipeline()
        # SD-Turbo operates in 1 step with guidance_scale=0.0
        result = pipe(prompt=str(prompt).strip(), num_inference_steps=1, guidance_scale=0.0)
        image = result.images[0]
        image.save(target_path)
        return f"✅ Image saved to: {target_path}"
    except Exception as exc:
        return f"❌ Failed to generate image: {exc}"
