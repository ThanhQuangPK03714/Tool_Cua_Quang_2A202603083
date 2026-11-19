"""Download the gated Meta SAM 3 checkpoint into the tool's default location."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    checkpoint = model_dir / "sam3.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        print(f"SAM 3 checkpoint already exists: {checkpoint}")
        return 0

    try:
        from huggingface_hub import get_token, hf_hub_download
    except ImportError:
        print("Missing huggingface_hub. Run .\\setup_gpu.ps1 first.", file=sys.stderr)
        return 1

    token = get_token()
    if not token:
        print(
            "No Hugging Face login is configured. Request access to facebook/sam3, "
            "then run .\\.venv-sam3\\Scripts\\hf.exe auth login and retry .\\run_gpu.ps1."
        )
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_dir=str(model_dir),
                token=token,
            )
        )
    except Exception as exc:
        print(
            "Could not download facebook/sam3. Confirm that the Hugging Face account "
            "has approved checkpoint access and a read token. "
            f"Details: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        print(f"Hugging Face did not return a usable checkpoint at {downloaded}.", file=sys.stderr)
        return 1
    if downloaded.resolve() != checkpoint.resolve():
        downloaded.replace(checkpoint)
    print(f"SAM 3 checkpoint is ready: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
