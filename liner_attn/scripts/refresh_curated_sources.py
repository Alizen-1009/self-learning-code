#!/usr/bin/env python3
"""Refresh curated GDN/KDA sources using disposable shallow GitHub clones.

The script clones each upstream repository into a temporary directory, copies the
selected files into a staged tree, records provenance and checksums, atomically
replaces ``code/``, and then removes all temporary clones.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "code"

SOURCES: dict[str, tuple[str, list[str]]] = {
    "flashinfer": (
        "flashinfer-ai/flashinfer",
        [
            "LICENSE",
            "NOTICE",
            "docs/api/gdn_decode.rst",
            "docs/api/gdn_prefill.rst",
            "docs/api/kda_decode.rst",
            "flashinfer/gdn_decode.py",
            "flashinfer/gdn_prefill.py",
            "flashinfer/gdn_kernels",
            "flashinfer/kda_decode.py",
            "flashinfer/kda_kernels",
            "flashinfer/trace/templates/gdn.py",
            "flashinfer/trace/templates/kda.py",
            "benchmarks/bench_gdn_decode.py",
            "benchmarks/bench_gdn_prefill.py",
            "benchmarks/bench_recurrent_kda.py",
            "benchmarks/gdn_triton_reference.py",
            "benchmarks/routines/gdn.py",
            "tests/gdn",
            "tests/kda",
        ],
    ),
    "fla": (
        "fla-org/flash-linear-attention",
        [
            "LICENSE",
            ".agents/skills/fla-kda/SKILL.md",
            "fla/layers/gated_deltanet.py",
            "fla/layers/gdn2.py",
            "fla/layers/kda.py",
            "fla/models/gated_deltanet",
            "fla/models/kda",
            "fla/ops/gated_delta_rule",
            "fla/ops/gdn2",
            "fla/ops/kda",
            "benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py",
            "benchmarks/cp/test_gdn_with_cp.py",
            "tests/context_parallel/test_cp_gdn.py",
            "tests/context_parallel/test_cp_kda.py",
            "tests/layers/test_gated_deltanet.py",
            "tests/models/test_modeling_gated_deltanet.py",
            "tests/models/test_modeling_kda.py",
            "tests/ops/test_gdn.py",
            "tests/ops/test_gdn2.py",
            "tests/ops/test_gdn_kernels.py",
            "tests/ops/test_kda.py",
        ],
    ),
    "flashkda": (
        "MoonshotAI/FlashKDA",
        [
            "LICENSE",
            "README.md",
            "BENCHMARK_GB200.md",
            "BENCHMARK_H20.md",
            "config.yaml",
            "setup.py",
            "flash_kda",
            "csrc",
            "docs",
            "benchmarks",
            "tests",
        ],
    ),
    "cula": (
        "inclusionAI/cuLA",
        [
            "LICENSE",
            "CITATION.cff",
            "README.md",
            "USAGE.md",
            "REPO_LAYOUT.md",
            "RECOMMENDED_CODING_STYLE.md",
            "BENCHMARK_GB200.md",
            "BENCHMARK_GB200_CUDA_130.md",
            "BENCHMARK_GB300.md",
            "BENCHMARK_H200.md",
            "BENCHMARK_KDA_DECODE_GB200.md",
            "BENCHMARK_KDA_DECODE_H203E.md",
            "pyproject.toml",
            "setup.py",
            "cula",
            "csrc",
            "docs",
            "benchmarks",
            "tests",
            "scripts",
        ],
    ),
}


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def clone_repository(slug: str, destination: Path) -> None:
    if shutil.which("gh"):
        run("gh", "repo", "clone", slug, str(destination), "--", "--depth=1")
    else:
        run("git", "clone", "--depth=1", f"https://github.com/{slug}.git", str(destination))


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Selected upstream path does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def write_revision(destination: Path, repository: Path, slug: str) -> None:
    revision = git_output(repository, "rev-parse", "HEAD")
    committed = git_output(repository, "show", "-s", "--format=%cI", "HEAD")
    destination.write_text(
        "# Upstream revision\n\n"
        f"- Repository: `https://github.com/{slug}`\n"
        f"- Commit: `{revision}`\n"
        f"- Commit time: `{committed}`\n"
        "- This directory is an unmodified, path-preserving source selection.\n"
        "- See the adjacent upstream `LICENSE` before reuse or redistribution.\n",
        encoding="utf-8",
    )


def write_checksums(destination: Path) -> None:
    rows: list[str] = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", "UPSTREAM_REVISION.md"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(destination).as_posix()}")
    (destination / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def replace_destination(staged: Path) -> None:
    backup = ROOT / ".code-refresh-backup"
    if backup.exists():
        shutil.rmtree(backup)
    if DESTINATION.exists():
        DESTINATION.rename(backup)
    try:
        staged.rename(DESTINATION)
    except Exception:
        if backup.exists() and not DESTINATION.exists():
            backup.rename(DESTINATION)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix=".source-refresh-", dir=ROOT) as temp_name:
        temp_root = Path(temp_name)
        clones = temp_root / "clones"
        staged = temp_root / "generated-code"
        clones.mkdir()
        staged.mkdir()

        for destination_name, (slug, selected_paths) in SOURCES.items():
            repository = clones / destination_name
            clone_repository(slug, repository)
            destination = staged / destination_name
            destination.mkdir()
            for relative_path in selected_paths:
                copy_path(repository / relative_path, destination / relative_path)
            write_revision(destination / "UPSTREAM_REVISION.md", repository, slug)
            write_checksums(destination)
            count = sum(1 for path in destination.rglob("*") if path.is_file())
            print(f"{destination_name}: copied {count} files")

        replace_destination(staged)


if __name__ == "__main__":
    main()
