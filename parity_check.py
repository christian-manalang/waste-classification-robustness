#!/usr/bin/env python3
"""
Parity check: compare the demo's preprocessing and model outputs with the
thesis notebook's outputs stored in parity_expected.csv.

Two paths are run for every row:
  Notebook path : Image.open(...).convert("RGB") -> eval_transform (no downscale),
                  exactly as the thesis notebook. Must match the expected row.
  App path      : downscale_long_side(img, 512) -> eval_transform, exactly as app.py.
                  Must match for TrashNet images (<= 512 px, downscale is a no-op);
                  for larger images a difference is reported as DRIFT.

Images are looked up in parity_images/ first, then parity_local_only/.
Missing images are reported as SKIPPED. Exits non-zero only on a FAIL.
"""

import csv
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from models import CLASSES, load_weights
from degrade import eval_transform, downscale_long_side

EXPECTED_CSV = "parity_expected.csv"
TEMPERATURES_PATH = "temperatures.json"
IMAGE_DIRS = ["parity_images", "parity_local_only"]
APP_MAX_SIDE = 512

CHECKPOINTS = {
    "resnet50": "weights/resnet50_best.pth",
    "cbam_resnet50": "weights/cbam_resnet50_best.pth",
    "edgenext_base": "weights/edgenext_base_best.pth",
}

CONF_TOL = 1e-3


def find_image(filename: str) -> Optional[str]:
    for d in IMAGE_DIRS:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    return None


def forward(model: torch.nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(eval_transform(img).unsqueeze(0).to(device))
    return logits.squeeze(0).double().cpu().numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logit_tol = 1e-2 if device.type == "cuda" else 1e-3

    with open(TEMPERATURES_PATH, "r", encoding="utf-8") as f:
        temperatures: Dict[str, float] = {k: float(v) for k, v in json.load(f)["temperature"].items()}

    with open(EXPECTED_CSV, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    models = {arch: load_weights(arch, path, device) for arch, path in CHECKPOINTS.items()}

    print("=" * 118)
    print(f"  Parity check vs {EXPECTED_CSV}  |  device: {device.type.upper()}  |  "
          f"logit tol {logit_tol:g}, conf tol {CONF_TOL:g}")
    print("=" * 118)
    header = (f"{'file':34s} {'model':14s} {'size':>9s} | {'NB pred':10s} {'max|dlogit|':>11s} "
              f"{'d_raw':>8s} {'d_ts':>8s} {'NB':5s}| {'App pred':10s} {'max|dlogit|':>11s} {'App':s}")
    print(header)
    print("-" * 118)

    images: Dict[str, Image.Image] = {}
    counts = {"PASS": 0, "FAIL": 0, "DRIFT": 0, "SKIPPED": 0}
    flips = 0
    failures = []

    for row in rows:
        fname, arch = row["file"], row["model"]
        label = f"{fname[:34]:34s} {arch:14s}"
        path = find_image(fname)
        if path is None:
            print(f"{label} {'':>9s} | SKIPPED (image not found in {' or '.join(IMAGE_DIRS)})")
            counts["SKIPPED"] += 1
            continue

        if path not in images:
            images[path] = Image.open(path).convert("RGB")
        img = images[path]
        long_side = max(img.size)
        is_trashnet = row["pool"].startswith("tn") or row["folder"] == "parity_images"

        exp_logits = np.array([float(row[f"logit_{i}"]) for i in range(len(CLASSES))])
        exp_pred = row["pred_class"]
        T = temperatures[arch]
        problems = []

        # Sanity: the CSV's own temperature and argmax must agree with temperatures.json and CLASSES order
        if abs(float(row["temperature"]) - T) > 1e-9:
            problems.append(f"CSV temperature {row['temperature']} != temperatures.json {T}")
        if CLASSES[int(np.argmax(exp_logits))] != exp_pred:
            problems.append("CSV pred_class does not match argmax of CSV logits")

        # Notebook path
        nb_logits = forward(models[arch], img, device)
        nb_pred = CLASSES[int(np.argmax(nb_logits))]
        nb_dlogit = float(np.max(np.abs(nb_logits - exp_logits)))
        d_raw = abs(float(softmax(nb_logits).max()) - float(row["conf_raw"]))
        d_ts = abs(float(softmax(nb_logits / T).max()) - float(row["conf_ts"]))
        nb_ok = nb_pred == exp_pred and nb_dlogit <= logit_tol and d_raw <= CONF_TOL and d_ts <= CONF_TOL
        if not nb_ok:
            problems.append("notebook path mismatch")

        # App path
        app_logits = forward(models[arch], downscale_long_side(img, APP_MAX_SIDE), device)
        app_pred = CLASSES[int(np.argmax(app_logits))]
        app_dlogit = float(np.max(np.abs(app_logits - exp_logits)))
        app_match = app_pred == exp_pred and app_dlogit <= logit_tol

        if app_match:
            app_status = "PASS"
        elif is_trashnet or long_side <= APP_MAX_SIDE:
            app_status = "FAIL"
            problems.append("app path mismatch on an image <= 512 px / TrashNet row")
        else:
            flipped = app_pred != exp_pred
            flips += flipped
            app_status = f"DRIFT (class flipped: {'yes' if flipped else 'no'})"

        row_status = "FAIL" if problems else ("DRIFT" if app_status.startswith("DRIFT") else "PASS")
        counts[row_status] += 1
        if problems:
            failures.append(f"{fname} [{arch}]: " + "; ".join(problems))

        size = f"{img.size[0]}x{img.size[1]}"
        print(f"{label} {size:>9s} | {nb_pred:10s} {nb_dlogit:11.2e} {d_raw:8.1e} {d_ts:8.1e} "
              f"{'PASS' if nb_ok else 'FAIL':5s}| {app_pred:10s} {app_dlogit:11.2e} {app_status}")

    print("-" * 118)
    for msg in failures:
        print(f"  FAIL: {msg}")
    print(f"SUMMARY: {len(rows)} rows | PASS {counts['PASS']} | DRIFT {counts['DRIFT']} "
          f"(class flipped in {flips}) | FAIL {counts['FAIL']} | SKIPPED {counts['SKIPPED']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
