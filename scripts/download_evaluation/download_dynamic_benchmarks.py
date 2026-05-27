#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download LHMPP dynamic evaluation benchmark from ModelScope and extract locally.

ModelScope repo: ``Damo_XR_Lab/LHMPP-Evaluation-Benchmark``

Default layout after extraction::

    {output_dir}/dynamic_benchmark/
        neuman/
        selfcapture/
        vid2avatar/

Usage (from LHM-plusplus root)::

    python scripts/download_evaluation/download_dynamic_benchmarks.py
    python scripts/download_evaluation/download_dynamic_benchmarks.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from typing import Iterable, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MODELSCOPE_MODEL_ID = "Damo_XR_Lab/LHMPP-Evaluation-Benchmark"
BENCHMARK_DIRNAME = "dynamic_benchmark"
EXPECTED_DATASETS = ("neuman", "selfcapture", "vid2avatar")
ARCHIVE_NAMES = (
    "dynamic_benchmark.tar.gz",
    "dynamic_benchmark.tgz",
    "dynamic_benchmark.tar",
)
DEFAULT_OUTPUT_DIR = os.path.join(".", "evaluation")


def _abs_under_repo(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(ROOT, path))


def benchmark_root(output_dir: str) -> str:
    """``{output_dir}/dynamic_benchmark``."""
    return os.path.normpath(os.path.join(_abs_under_repo(output_dir), BENCHMARK_DIRNAME))


def has_dynamic_benchmark(output_dir: str) -> bool:
    """Return True if ``dynamic_benchmark/`` looks populated."""
    root = benchmark_root(output_dir)
    if not os.path.isdir(root):
        return False
    present = {
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    }
    return any(name in present for name in EXPECTED_DATASETS)


def _find_archive(search_root: str) -> Optional[str]:
    preferred = [os.path.join(search_root, name) for name in ARCHIVE_NAMES]
    for path in preferred:
        if os.path.isfile(path):
            return path
    for dirpath, _, filenames in os.walk(search_root):
        for name in filenames:
            lower = name.lower()
            if lower in ARCHIVE_NAMES:
                return os.path.join(dirpath, name)
            if lower.endswith((".tar.gz", ".tgz", ".tar")) and "dynamic_benchmark" in lower:
                return os.path.join(dirpath, name)
    return None


def _open_tar(archive_path: str) -> tarfile.TarFile:
    if archive_path.endswith((".gz", ".tgz")):
        return tarfile.open(archive_path, "r:gz")
    return tarfile.open(archive_path, "r:")


def _extract_archive(archive_path: str, output_dir: str) -> str:
    """Extract archive so ``{output_dir}/dynamic_benchmark`` exists."""
    dest_parent = _abs_under_repo(output_dir)
    os.makedirs(dest_parent, exist_ok=True)
    target = benchmark_root(output_dir)

    print(f"Extracting {os.path.basename(archive_path)} -> {dest_parent} ...")
    with _open_tar(archive_path) as tf:
        members = tf.getmembers()
        top_names = {
            m.name.split("/", 1)[0]
            for m in members
            if m.name and m.name.split("/", 1)[0]
        }
        has_wrapper = BENCHMARK_DIRNAME in top_names

        if has_wrapper:
            if os.path.isdir(target):
                shutil.rmtree(target)
            tf.extractall(dest_parent)
        else:
            os.makedirs(target, exist_ok=True)
            tf.extractall(target)

    if not has_dynamic_benchmark(output_dir):
        raise FileNotFoundError(
            f"Archive extracted but benchmark not found at {target}. "
            f"Expected subdirs like {EXPECTED_DATASETS}."
        )
    return target


def _copy_tree(src: str, output_dir: str) -> str:
    dest_parent = _abs_under_repo(output_dir)
    target = benchmark_root(output_dir)
    os.makedirs(dest_parent, exist_ok=True)
    if os.path.isdir(target):
        shutil.rmtree(target)
    print(f"Copying {src} -> {target} ...")
    shutil.copytree(src, target)
    return target


def download_from_modelscope(cache_dir: str) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as ex:
        raise ImportError(
            "modelscope is required. Install with: pip install modelscope"
        ) from ex

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {MODELSCOPE_MODEL_ID} (ModelScope) -> cache {cache_dir} ...")
    local_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=cache_dir)
    print(f"ModelScope snapshot: {local_dir}")
    return local_dir


def download_dynamic_benchmark(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    force: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """Download and extract benchmark; return ``.../dynamic_benchmark`` path."""
    if has_dynamic_benchmark(output_dir) and not force:
        target = benchmark_root(output_dir)
        print(f"dynamic_benchmark already exists at {target}, skip.")
        return target

    cache = cache_dir or os.path.join(ROOT, ".cache", "lhmpp_evaluation_benchmark")
    downloaded = download_from_modelscope(cache)

    archive_path = _find_archive(downloaded)
    if archive_path:
        return _extract_archive(archive_path, output_dir)

    src_tree = os.path.join(downloaded, BENCHMARK_DIRNAME)
    if os.path.isdir(src_tree):
        return _copy_tree(src_tree, output_dir)

    raise FileNotFoundError(
        f"Could not find {ARCHIVE_NAMES} or {BENCHMARK_DIRNAME}/ under {downloaded}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Parent directory for extracted data (default: ./evaluation). "
            "Creates {output-dir}/dynamic_benchmark/."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="ModelScope download cache (default: .cache/lhmpp_evaluation_benchmark under repo root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if dynamic_benchmark/ already exists.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        target = download_dynamic_benchmark(
            args.output_dir,
            force=args.force,
            cache_dir=args.cache_dir,
        )
    except Exception as ex:
        print(f"Download failed: {ex}", file=sys.stderr)
        return 1

    print(f"Dynamic benchmark ready: {target}")
    present = sorted(
        name
        for name in os.listdir(target)
        if os.path.isdir(os.path.join(target, name)) and not name.startswith(".")
    )
    print(f"Datasets on disk: {', '.join(present)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
