#!/usr/bin/env python3
"""
Fix HF Space README and clean up unwanted files.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "PranavaKumar09/sentinel-env"


def main():
    api = HfApi()

    # Read the HF Space README
    readme_path = Path(__file__).parent / "hf-space-readme.md"
    with open(readme_path, encoding="utf-8") as f:
        readme_content = f.read()

    # Upload proper README.md
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="fix: update README with proper HF Space metadata",
    )

    # Delete unwanted directories
    dirs_to_delete = [
        ".code-review-graph",
    ]

    for dir_name in dirs_to_delete:
        print(f"Deleting {dir_name}/...")
        # Get all files in that directory
        files = api.list_repo_files(repo_id=SPACE_ID, repo_type="space")
        files_to_delete = [f for f in files if f.startswith(f"{dir_name}/")]

        for file_path in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    commit_message=f"chore: remove {file_path}",
                )
            except Exception as e:
                print(f"  Warning: Could not delete {file_path}: {e}")

    # Delete other unwanted files
    files_to_delete = [
        "__init__.py",  # Was deleted locally
        "HYPERIONRL_SUMMARY.md",
        "IMPLEMENTATION_COMPLETE.md",
        "PRE_SUBMISSION_CHECKLIST.md",
        "QUICK_REFERENCE.md",
        "prepare_for_submission.bat",
        "deploy-hf.py",
    ]

    for file_name in files_to_delete:
        print(f"Deleting {file_name}...")
        try:
            api.delete_file(
                path_in_repo=file_name,
                repo_id=SPACE_ID,
                repo_type="space",
                commit_message=f"chore: remove {file_name}",
            )
        except Exception as e:
            print(f"  Warning: Could not delete {file_name}: {e}")

    print("\n✅ Space cleanup complete!")
    print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
    print("The Space will rebuild automatically.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
