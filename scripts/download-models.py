from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "istupakov/parakeet-ctc-0.6b-onnx"
REQUIRED_FILES = ["model.onnx", "vocab.txt"]
OPTIONAL_FILES = ["model.onnx.data"]


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download(models_dir: Path, revision: str | None) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        print("Missing dependency: huggingface_hub.", file=sys.stderr)
        print("Install it with: python -m pip install huggingface_hub", file=sys.stderr)
        return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir=str(models_dir),
                revision=revision,
            )
            print(f"Downloaded {filename} -> {path}")
        except Exception as exc:
            if filename in OPTIONAL_FILES:
                print(f"Optional file not downloaded ({filename}): {exc}", file=sys.stderr)
                continue
            print(f"Failed to download required file ({filename}): {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model files into the local models cache."
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Models cache directory (default: <repo>/models).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision tag/commit.",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    models_dir = Path(args.models_dir) if args.models_dir else repo_root / "models"

    exit_code = download(models_dir, args.revision)
    if exit_code == 0:
        print(f"Done. Models cached under: {models_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
