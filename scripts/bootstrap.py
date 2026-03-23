from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False).returncode


def ensure_tool(name: str) -> bool:
    return shutil.which(name) is not None


def ensure_hf_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        print("Missing dependency: huggingface_hub. Installing...")
        code = run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap Symbiote: install deps, download models, optional run."
    )
    parser.add_argument(
        "--skip-npm",
        action="store_true",
        help="Skip npm install.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model download.",
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
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the app after setup (npm run tauri dev).",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()

    if not args.skip_npm:
        if not ensure_tool("npm"):
            print("Missing tool: npm. Install Node.js first.")
            return 1
        code = run(["npm", "install"], cwd=repo_root)
        if code != 0:
            return code

    if not args.skip_models:
        if not ensure_hf_hub():
            print("Failed to install huggingface_hub. Please install it and retry.")
            return 1
        cmd = [sys.executable, str(repo_root / "scripts" / "download-models.py")]
        if args.models_dir:
            cmd += ["--models-dir", args.models_dir]
        if args.revision:
            cmd += ["--revision", args.revision]
        code = run(cmd, cwd=repo_root)
        if code != 0:
            return code

    if args.run:
        if not ensure_tool("npm"):
            print("Missing tool: npm. Install Node.js first.")
            return 1
        return run(["npm", "run", "tauri", "dev"], cwd=repo_root)

    print("Setup complete.")
    print("Next steps:")
    print("  npm run tauri dev")
    print("Optional voice service:")
    print("  python voice_service_v2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
