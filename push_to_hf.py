"""
Push the Adaptive Enterprise Autopilot to HuggingFace Spaces.

Usage:
    cd autopilot-env
    HF_TOKEN=hf_xxx python push_to_hf.py

Creates a Docker Space and uploads all project files.
"""

import os
import sys

try:
    from huggingface_hub import HfApi, whoami
except ImportError:
    print("ERROR: pip install huggingface-hub")
    sys.exit(1)

REPO_ID   = os.getenv("HF_REPO", "your-username/adaptive-enterprise-autopilot")
REPO_TYPE = "space"

api = HfApi()

try:
    info = whoami()
    print(f"Logged in as: {info['name']}")
    # Auto-fill username
    if REPO_ID.startswith("your-username"):
        REPO_ID = REPO_ID.replace("your-username", info["name"])
except Exception:
    print("ERROR: Not logged in. Run: huggingface-cli login")
    sys.exit(1)

print(f"Creating Space: {REPO_ID} ...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print("  Space ready.")
except Exception as e:
    print(f"  Note: {e}")

SKIP_DIRS = {"__pycache__", ".git", "dist", "build", ".venv", "venv",
             "grpo_output", "sft_warmup", "trained_adapter"}
SKIP_EXTS = {".pyc", ".pyo", ".zip"}

uploaded = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    for fname in files:
        ext = os.path.splitext(fname)[1]
        if ext in SKIP_EXTS:
            continue
        local_path = os.path.join(root, fname)
        path_in_repo = local_path.replace("\\", "/").lstrip("./")
        print(f"  Uploading: {path_in_repo}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            uploaded += 1
        except Exception as e:
            print(f"  WARNING: Could not upload {path_in_repo}: {e}")

print(f"\nDone! {uploaded} files uploaded.")
print(f"Space URL : https://huggingface.co/spaces/{REPO_ID}")
base = REPO_ID.replace("/", "-").replace("_", "-").lower()
print(f"API URL   : https://{base}.hf.space")
print("\nSpace will build in ~2-3 minutes.")
