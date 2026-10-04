import glob, os, re
import numpy as np, torch
from PIL import Image
from models import CLASSES, load_weights
from degrade import eval_transform, apply_brightness_darkening

ALPHA = 0.3
MODELS = {
    "ResNet-50": ("resnet50", "weights/resnet50_best.pth"),
    "CBAM-ResNet50": ("cbam_resnet50", "weights/cbam_resnet50_best.pth"),
    "EdgeNeXt-Base": ("edgenext_base", "weights/edgenext_base_best.pth"),
}
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = {n: load_weights(a, p, dev) for n, (a, p) in MODELS.items()}

def predict(m, pil):
    with torch.no_grad():
        return CLASSES[int(m(eval_transform(pil).unsqueeze(0).to(dev)).argmax(1))]

files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(f"samples/*.{ext}"))
clean_ok = {n: 0 for n in models}
dark_ok = {n: 0 for n in models}

for f in files:
    label = re.match(r"[a-z]+", os.path.basename(f).lower()).group(0)
    img = Image.open(f).convert("RGB")
    dark = Image.fromarray(apply_brightness_darkening(np.array(img), ALPHA))
    row = []
    for n, m in models.items():
        pc, pd = predict(m, img), predict(m, dark)
        clean_ok[n] += pc == label
        dark_ok[n] += pd == label
        row.append(f"{n[:5]}: {pc}->{pd}")
    print(f"{os.path.basename(f):22s} true={label:10s} | " + " | ".join(row))

N = len(files)
if N > 0:
    print(f"\n{'model':15s} clean   alpha={ALPHA}")
    for n in models:
        print(f"{n:15s} {clean_ok[n]}/{N}   {dark_ok[n]}/{N}")
else:
    print("\nNo sample files found in samples/ directory.")
