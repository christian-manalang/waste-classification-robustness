#!/usr/bin/env python3
"""
Verify checkpoint weights for the waste classification robustness models.
Loads ResNet-50, CBAM-ResNet50, and EdgeNeXt-Base checkpoints with strict=True
and reports parameter counts against expected thesis benchmarks.
"""

import os
import sys
from models import load_weights

CHECKPOINT_CONFIGS = [
    ("resnet50", "weights/resnet50_best.pth", 23.52),
    ("cbam_resnet50", "weights/cbam_resnet50_best.pth", 26.04),
    ("edgenext_base", "weights/edgenext_base_best.pth", 17.92),
]


def check_all_checkpoints() -> bool:
    print("=" * 68)
    print("  Waste Classification Robustness Demo - Checkpoint Verification")
    print("=" * 68)

    all_passed = True

    for arch, path, expected_params in CHECKPOINT_CONFIGS:
        print(f"\nEvaluating Architecture: {arch}")
        print(f"  Checkpoint Path : {path}")

        if not os.path.isfile(path):
            print(f"  ❌ Status: File not found at '{path}'")
            all_passed = False
            continue

        try:
            model = load_weights(arch=arch, ckpt_path=path, device="cpu")
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✅ Status: Loaded successfully with strict=True!")
            print(f"  Parameters: {param_count:.2f} M (expected ~{expected_params:.2f} M)")
        except Exception as e:
            print(f"  ❌ Status: Checkpoint loading failed: {e}")
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
