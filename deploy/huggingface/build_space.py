"""Assemble (and optionally upload) the Hugging Face Space for Fairview.

    python deploy/huggingface/build_space.py                      # build only
    python deploy/huggingface/build_space.py --upload OWNER/NAME  # build, create the Space, upload

The app comes from files tracked by git only: no enrolled faces, logs, keys,
models or virtual environments. The RFW scores for the Fairness report are
added from a local, git-ignored FaVFA-main/ folder when one exists. Uploading needs `huggingface_hub` and a prior
`huggingface-cli login` with a token that can write to OWNER.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# What the container needs from the repository, by path prefix.
APP_FILES = ("run.py", "requirements.txt", "backend/", "frontend/")
FAVFA_FILES = ("FaVFA-main/LICENCE", "FaVFA-main/README.md", "FaVFA-main/data/rfw.csv",
               "FaVFA-main/model_results/BUPT_RFW.csv")


def tracked_files() -> list[str]:
    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    out = subprocess.run([git, "-C", str(REPO), "ls-files", "-z"], check=True, capture_output=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = []

    def copy(source: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / source, target)
        written.append(target)

    for path in tracked_files():
        if path.startswith(APP_FILES) and "__pycache__" not in path:
            copy(path, out / "app" / path)
    favfa = [path for path in FAVFA_FILES if (REPO / path).is_file()]
    for path in favfa:
        copy(path, out / "app" / path)
    if not favfa:
        print("note: no FaVFA-main/ found, the Space's Fairness report will skip the RFW section")
    for name in ("Dockerfile", "README.md"):
        copy(str((HERE / name).relative_to(REPO)), out / name)
    return written


def upload(out: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    kwargs = dict(folder_path=str(out), repo_id=repo_id, repo_type="space", commit_message="Deploy Fairview")
    try:  # remove files the new build no longer has (huggingface_hub 0.16+)
        api.upload_folder(**kwargs, delete_patterns=["*"])
    except TypeError:
        api.upload_folder(**kwargs)
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "build" / "hf-space", help="where to assemble the Space")
    parser.add_argument("--upload", metavar="OWNER/NAME", help="create (if needed) and upload to this Space")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    files = build(args.out)
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"assembled {len(files)} files ({size:.1f} MB) in {args.out}")
    if args.upload:
        url = upload(args.out, args.upload, args.private)
        owner, name = args.upload.split("/", 1)
        print(f"uploaded: {url}")
        print(f"app address once built: https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    sys.exit(main())
