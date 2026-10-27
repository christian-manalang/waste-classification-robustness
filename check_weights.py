#!/usr/bin/env python3
"""
Verify checkpoint weights for the waste classification robustness models.
Loads ResNet-50, CBAM-ResNet50, and EdgeNeXt-Base checkpoints with strict=True
and reports parameter counts against expected thesis benchmarks.
Also verifies each checkpoint's SHA-256 against checkpoint_hashes.json
(thesis Phase 2 checkpoints, seeds 42, 43, 44).
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

CHECKPOINT_CONFIGS = [
    ("resnet50", "weights/resnet50_best.pth", 23.52),
    ("cbam_resnet50", "weights/cbam_resnet50_best.pth", 26.04),
    ("edgenext_base", "weights/edgenext_base_best.pth", 17.92),
]

HASHES_PATH = "checkpoint_hashes.json"


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def match_seed(digest: str, size: int, filename: str, expected: Dict[str, Any]) -> Optional[str]:
    """
    Return the seed ("seed42", ...) whose entry for this filename has the same
    SHA-256 and byte size, or None. A hash that only matches a different
    model's filename does not count as a match.
    """
    for key, entry in expected.items():
        seed, name = key.split("/", 1)
        if name == filename and entry.get("sha256") == digest and entry.get("bytes") == size:
            return seed
    return None


def verify_hashes(hashes_path: str = HASHES_PATH) -> Tuple[bool, Optional[str], Dict[str, Dict[str, Any]], str]:
    """
    Hash every checkpoint in CHECKPOINT_CONFIGS and compare with hashes_path.
    Returns (ok, seed, per-arch results, message). ok is True only if every file
    matches and all three match the same seed.
    """
    if not os.path.isfile(hashes_path):
        return False, None, {}, f"'{hashes_path}' not found"

    with open(hashes_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    results: Dict[str, Dict[str, Any]] = {}
    for arch, path, _ in CHECKPOINT_CONFIGS:
        if not os.path.isfile(path):
            results[arch] = {"path": path, "sha256": None, "seed": None}
            continue
        digest = sha256_file(path)
        size = os.path.getsize(path)
        results[arch] = {
            "path": path,
            "sha256": digest,
            "seed": match_seed(digest, size, os.path.basename(path), expected),
        }

    unmatched = [arch for arch, r in results.items() if r["seed"] is None]
    seeds = {r["seed"] for r in results.values() if r["seed"] is not None}

    if unmatched:
        return False, None, results, f"no match for: {', '.join(unmatched)}"
    if len(seeds) != 1:
        return False, None, results, f"checkpoints match different seeds: {', '.join(sorted(seeds))}"
    seed = seeds.pop()
    return True, seed, results, f"all checkpoints match {seed}"


def check_all_checkpoints() -> bool:
    from models import load_weights

    print("=" * 68)
    print("  Waste Classification Robustness Demo - Checkpoint Verification")
    print("=" * 68)

    all_passed = True

    for arch, path, expected_params in CHECKPOINT_CONFIGS:
        print(f"\nEvaluating Architecture: {arch}")
        print(f"  Checkpoint Path : {path}")

        if not os.path.isfile(path):
            print(f"  [ERROR] Status: File not found at '{path}'")
            all_passed = False
            continue

        try:
            model = load_weights(arch=arch, ckpt_path=path, device="cpu")
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  [PASS] Status: Loaded successfully with strict=True!")
            print(f"  Parameters: {param_count:.2f} M (expected ~{expected_params:.2f} M)")
        except Exception as e:
            print(f"  [ERROR] Status: Checkpoint loading failed: {e}")
            all_passed = False

    print("\n" + "-" * 68)
    print(f"  SHA-256 Verification (against {HASHES_PATH})")
    print("-" * 68)

    hashes_ok, seed, hash_results, message = verify_hashes()
    for arch, r in hash_results.items():
        if r["sha256"] is None:
            print(f"  {arch:15s} [ERROR] File not found at '{r['path']}'")
            continue
        status = "[PASS]" if r["seed"] else "[ERROR]"
        print(f"  {arch:15s} {status} sha256 {r['sha256'][:8]}  ->  {r['seed'] or 'NO MATCH'}")
    if hashes_ok:
        print(f"  [PASS] {message}")
    else:
        print(f"  [ERROR] {message}")
        all_passed = False

    print("\n" + "=" * 68)
    if all_passed:
        print("  All model checkpoints loaded and verified successfully.")
    else:
        print("  Warning: One or more checkpoints failed verification.")
    print("=" * 68)

    return all_passed


if __name__ == "__main__":
    success = check_all_checkpoints()
    sys.exit(0 if success else 1)
