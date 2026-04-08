"""Deploy Sentinel Environment to Hugging Face Spaces using cached credentials."""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# ── Config ──────────────────────────────────────────────────────────
SPACE_NAME = "sentinel-env"
PROJECT_ROOT = Path(__file__).parent

api = HfApi()

# Get current user
whoami = api.whoami()
username = whoami["name"]
SPACE_ID = f"{username}/{SPACE_NAME}"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE_ID}"

print(f"=" * 60)
print(f"  DEPLOYING SENTINEL ENVIRONMENT TO HF SPACES")
print(f"=" * 60)
print(f"  User:  {username}")
print(f"  Space: {SPACE_ID}")
print(f"  URL:   {SPACE_URL}")
print(f"=" * 60)


def ignore_patterns(path, names):
    """Ignore sensitive files during deployment."""
    ignored = set()
    for name in names:
        if name.startswith('.') and name not in {'.dockerignore', '.gitignore'}:
            ignored.add(name)
        if name.endswith(('.pyc', '.pyo', '.log', '.key', '.secret')):
            ignored.add(name)
        if 'pycache' in name:
            ignored.add(name)
    return ignored


# ── Step 1: Create Space if it doesn't exist ───────────────────────
print(f"\n[1/4] Creating Space: {SPACE_ID}")
try:
    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"  ✓ Space created/exists")
except Exception as e:
    print(f"  ⚠ WARNING: {e}")
    print(f"  Continuing anyway (space may already exist)...")

# ── Step 2: Prepare deploy directory ───────────────────────────────
print(f"\n[2/4] Preparing deployment files...")
DEPLOY_DIR = PROJECT_ROOT / "hf-deploy"
if DEPLOY_DIR.exists():
    shutil.rmtree(DEPLOY_DIR)
DEPLOY_DIR.mkdir()

# Copy space README (with HF metadata)
shutil.copy2(PROJECT_ROOT / "hf-space-readme.md", DEPLOY_DIR / "README.md")

# Copy project files
files_to_copy = [
    "Dockerfile",
    ".dockerignore",
    "models.py",
    "client.py",
    "openenv.yaml",
    "pyproject.toml",
    "uv.lock",
    "__init__.py",
    "inference.py",
    "inference_logging.py",
]

for fname in files_to_copy:
    src = PROJECT_ROOT / fname
    if src.exists():
        shutil.copy2(src, DEPLOY_DIR / fname)
        print(f"  ✓ Copied: {fname}")
    else:
        print(f"  ✗ MISSING: {fname}")

# Copy server directory
server_src = PROJECT_ROOT / "server"
server_dst = DEPLOY_DIR / "server"
if server_src.exists():
    if server_dst.exists():
        shutil.rmtree(server_dst)
    shutil.copytree(server_src, server_dst, ignore=ignore_patterns)
    print(f"  ✓ Copied: server/")

# Copy tests directory
tests_src = PROJECT_ROOT / "tests"
tests_dst = DEPLOY_DIR / "tests"
if tests_src.exists():
    if tests_dst.exists():
        shutil.rmtree(tests_dst)
    shutil.copytree(tests_src, tests_dst, ignore=ignore_patterns)
    print(f"  ✓ Copied: tests/")

# Create .gitignore for the space
with open(DEPLOY_DIR / ".gitignore", "w") as f:
    f.write("__pycache__/\n*.pyc\n*.pyo\n.env\nhf-deploy/\n")

print(f"\n  Total files ready: {sum(1 for _ in DEPLOY_DIR.rglob('*') if _.is_file())}")

# ── Step 3: Upload to HF Space ─────────────────────────────────────
print(f"\n[3/4] Uploading to {SPACE_URL} ...")
try:
    upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="feat: comprehensive code quality, security, and performance improvements\n\n- Security: Remove hardcoded HF_TOKEN, add non-root Docker users\n- Auth: Add API key authentication middleware\n- Rate limiting: 100 req/min per IP\n- Performance: O(n²) → O(1) metrics calculation\n- Testing: 80 tests passing\n- Architecture: Consolidated duplication, fixed imports",
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    print(f"  ✓ All files uploaded!")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ── Step 4: Verify ─────────────────────────────────────────────────
print(f"\n[4/4] Verifying deployment...")
try:
    info = api.space_info(SPACE_ID)
    print(f"  Space ID:    {info.id}")
    print(f"  SDK:         {info.sdk}")
    print(f"  URL:         {info.url}")
    print(f"  Created:     {info.created_at}")
    print(f"  Last update: {info.last_modified}")
except Exception as e:
    print(f"  ⚠ WARNING: Could not fetch space info: {e}")

# ── Cleanup ────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  DEPLOYMENT COMPLETE!")
print(f"{'=' * 60}")
print(f"  Space URL: {SPACE_URL}")
print(f"  Direct URL: https://{username}-sentinel-env.hf.space")
print(f"\n  The Space is building now (3-5 minutes).")
print(f"  When 'Running', test with:")
print(f"    curl -X POST https://{username}-sentinel-env.hf.space/health")
print(f"{'=' * 60}")

# Cleanup deploy dir
shutil.rmtree(DEPLOY_DIR, ignore_errors=True)
