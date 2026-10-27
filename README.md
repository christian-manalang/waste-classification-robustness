# Visual Degradation Robustness in Waste Classification

A Streamlit-based interactive demonstration evaluating the robustness and inference latency of Convolutional Neural Networks (CNNs) and Vision Transformers under real-world visual perturbations for automated recycling systems.

Official repository, interactive evaluation dashboard, and artifact archive for the undergraduate thesis:  
**"A Comparative Study on the Robustness of Residual and Attention-Based Architectures for Waste Classification Under Degraded Conditions"**  
*School of Information Technology, Mapúa University (October 2026)*  
**Authors:** Nicko Gabriel A. Baldo, Jorge Christian B. Manalang, Liandro E. Refulle

**Live Application:** [waste-classification-robustness.streamlit.app](https://waste-classification-robustness.streamlit.app/)

---

## Interactive & Reproducible Colab Notebooks

To ensure experimental transparency and multi-seed variance validation across all methodological phases, the complete execution pipelines are available via view-only Google Colab links:

| Execution Run | Target Seed | Description | Notebook Access |
| :--- | :---: | :--- | :---: |
| **1st Run (Representative)** | `Seed 42` | Primary baseline, Phase 5 retraining, Phase 6 ablation, and Phase 7 RW-TS evaluations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uJbjeF5og_6YRoJ6LedLHtmLDIFNF_48?usp=sharing) |
| **2nd Run (Variance Replication 1)** | `Seed 44` | Independent model re-initialization and optimization run across identical partitions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RQLwQtvX5GVjXUPQ16XgJ8HML6olVGfY?usp=sharing) |
| **3rd Run (Variance Replication 2)** | `Seed 43` | Independent model re-initialization and optimization run across identical partitions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XXxbjIcpSi3OGD01RQCkNPatp4C-dguR?usp=sharing) |
| **Post-Defense Revision** | `Seeds 42, 43, 44` | Phases 8 to 12: audit, training-change screens, calibration, one-time test evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lkB2qseokrA--yiLdx7vl-lFFpo9ua91?usp=sharing) |

---

## Project Architecture & Files

```text
waste-classification-robustness/
├── .gitignore             # Git ignore exclusions (includes parity_local_only/)
├── app.py                 # Streamlit wide-layout interactive robustness dashboard
├── cam.py                 # Lightweight Grad-CAM implementation and visualization helpers
├── check_weights.py       # Checkpoint verification (strict loading, parameter counts, SHA-256)
├── checkpoint_hashes.json # SHA-256 and byte size of the thesis Phase 2 checkpoints (seeds 42, 43, 44)
├── temperatures.json      # Temperature scaling constants (seed 42 used by the app)
├── degrade.py             # Preprocessing pipeline, 512 px downscale and visual degradation functions
├── headline_test.py       # Automated batch evaluation under darkening (alpha=0.3)
├── models.py              # Neural network architectures, CBAM wrappers, strict weight loaders
├── parity_check.py        # Compares the demo's outputs with the thesis notebook's outputs
├── parity_expected.csv    # Notebook logits, predictions and confidences for the parity images
├── parity_images/         # TrashNet test images used by parity_check.py
├── requirements.txt       # Pinned Python project dependencies (Python 3.14.4)
├── weights/               # Thesis Phase 2 checkpoints, seed 42 (Git LFS)
│   ├── resnet50_best.pth
│   ├── cbam_resnet50_best.pth
│   └── edgenext_base_best.pth
├── samples/               # Directory for real TrashNet test images (*.jpg, *.png)
└── README.md              # Complete project documentation and guide
```

---

## Model Weights & Verification

### Checkpoint Placement
Trained PyTorch checkpoints must be placed in the `weights/` directory:

| Architecture | Checkpoint Path | Expected Parameters | FLOPs | Type |
| :--- | :--- | :---: | :---: | :--- |
| **ResNet-50** | `weights/resnet50_best.pth` | `~23.52 M` | `4.132 GFLOPs` | Standard Residual Baseline |
| **CBAM-ResNet50** | `weights/cbam_resnet50_best.pth` | `~26.04 M` | `4.144 GFLOPs` | Channel & Spatial Attention CNN |
| **EdgeNeXt-Base** | `weights/edgenext_base_best.pth` | `~17.92 M` | `2.925 GFLOPs` | Hybrid ConvNeXt & Transformer |

*Note: If any required checkpoint is missing for the active mode, `app.py` halts execution with a missing weight notification to guarantee that evaluations only run on verified trained weights.*

### Verifying Checkpoints
To verify all checkpoints against the exact architectural definitions, confirm that weights load with `strict=True`, and confirm that each file is the exact checkpoint reported in the thesis (SHA-256):

```bash
python check_weights.py
```

Expected output:
```text
====================================================================
  Waste Classification Robustness Demo - Checkpoint Verification
====================================================================

Evaluating Architecture: resnet50
  Checkpoint Path : weights/resnet50_best.pth
  [PASS] Status: Loaded successfully with strict=True!
  Parameters: 23.52 M (expected ~23.52 M)

Evaluating Architecture: cbam_resnet50
  Checkpoint Path : weights/cbam_resnet50_best.pth
  [PASS] Status: Loaded successfully with strict=True!
  Parameters: 26.04 M (expected ~26.04 M)

Evaluating Architecture: edgenext_base
  Checkpoint Path : weights/edgenext_base_best.pth
  [PASS] Status: Loaded successfully with strict=True!
  Parameters: 17.93 M (expected ~17.92 M)

--------------------------------------------------------------------
  SHA-256 Verification (against checkpoint_hashes.json)
--------------------------------------------------------------------
  resnet50        [PASS] sha256 d401280f  ->  seed42
  cbam_resnet50   [PASS] sha256 ed7ac1b6  ->  seed42
  edgenext_base   [PASS] sha256 09e54ab2  ->  seed42
  [PASS] all checkpoints match seed42

====================================================================
  All model checkpoints loaded and verified successfully.
====================================================================
```

---

## Thesis Weights, Calibration & Parity

### Thesis checkpoints and SHA-256 verification
`weights/` holds the thesis Phase 2 checkpoints for seed 42 (stored with Git LFS). `checkpoint_hashes.json` lists the SHA-256 hash and byte size of the Phase 2 checkpoints for seeds 42, 43 and 44. `check_weights.py` hashes each file in `weights/`, prints the first 8 hex characters and the seed it matches (`seed42`, `seed43`, `seed44` or `NO MATCH`), and exits non-zero if any file has no match or the three models do not all match the same seed. The app runs the same check once at startup and shows the verified hashes under the title, or a warning if verification fails.

### Calibrated confidence (`temperatures.json`)
`temperatures.json` holds one temperature scaling constant per model; the app uses the seed 42 values under `"temperature"` (`"all_seeds"` is for reference only). Calibrated probabilities are `softmax(logits / T)`. This changes how confident a model is, never which class it predicts. The constants were fitted on 88 internet photos (Wild-Dev). The sidebar checkbox *Show calibrated confidence (temperature scaling)* is on by default; turning it off shows the raw softmax. The in-app expander *About calibrated confidence* summarizes the thesis results (Section 4.11.2).

### Parity check (`parity_check.py`)
`parity_check.py` confirms that the demo reproduces the thesis notebook. For each row of `parity_expected.csv` it loads the image (from `parity_images/`, then `parity_local_only/`; missing images are reported as `SKIPPED`) and runs two paths with the app's own model loader, `eval_transform`, temperatures and downscale function:

- **Notebook path**: the original image resized directly to 224x224. The predicted class must match, every logit must be within `1e-3` (CPU) or `1e-2` (GPU), and raw and calibrated confidence within `1e-3`.
- **App path**: the 512 px downscale followed by the same transform, exactly as in `app.py`. TrashNet images (at most 512 px) must match exactly; for larger photos a difference is reported as `DRIFT`, together with whether the predicted class flipped.

```bash
python parity_check.py
```

The script exits non-zero only on a `FAIL`. With only the committed `parity_images/`, the RW-TS and internet rows are reported as `SKIPPED`.

### Why images are downscaled to 512 px
TrashNet images are stored at 512 px or less on the long side, and the thesis applied blur and occlusion at that stored resolution before the 224x224 resize. The app therefore downscales every image to a 512 px long side (Lanczos, `degrade.downscale_long_side`) before degradation and inference, so that a given blur sigma or occlusion coverage has the same strength as in the thesis. This step is deliberate. For photos larger than 512 px, predictions can differ slightly from the thesis notebook, which resizes from the full-resolution image; `parity_check.py` reports this as `DRIFT`.

### Images that are never committed
`parity_local_only/` (RW-TS photos and internet images) and any other internet images are kept locally only and are never committed; the folder is listed in `.gitignore`. Only the TrashNet images in `parity_images/` are part of the repository.

---

## Model Architectures (`models.py`)

- **Classes**: `["cardboard", "glass", "metal", "paper", "plastic", "trash"]` (6 categories).
- **ResNet-50**: Standard torchvision ResNet-50 with 6-class linear classification head.
- **CBAM-ResNet50**: Matches the training notebook wrapper architecture:
  - `ChannelAttention`: `mlp = nn.Sequential(Conv2d(c, c//r, 1, bias=False), ReLU(), Conv2d(c//r, c, 1, bias=False))`, forward applies sigmoid over summed average-pooled and max-pooled representations.
  - `SpatialAttention`: `conv = Conv2d(2, 1, kernel_size=7, padding=3, bias=False)`, forward applies sigmoid over concatenated channel-pooled maps.
  - `CBAM`: Sequential application of `self.channel_attention` and `self.spatial_attention`.
  - `CBAMBottleneck`: Wraps an underlying torchvision `Bottleneck` block (`self.bottleneck = bottleneck`), inserting CBAM refinement after `conv3 -> bn3` before residual identity addition and final ReLU activation.
  - `build_cbam_resnet50`: Wraps every block in `layer1`-`layer4` of `models.resnet50(weights=None)` with `CBAMBottleneck`.
- **EdgeNeXt-Base**: Built via `timm.create_model("edgenext_base.in21k_ft_in1k", num_classes=6)`.
- **Weight Loader (`load_weights`)**: Strips any DistributedDataParallel `module.` prefixes and loads state dicts strictly (`strict=True`).

---

## Visual Explainability via Grad-CAM (`cam.py`)

The application integrates class activation mapping to inspect visual focus regions across clean and perturbed inputs:

- **Purpose**: Diagnoses whether prediction shifts under perturbations stem from feature erasure, contrast loss, or attention drift away from the target object.
- **Target Layer Resolution (`get_target_layer`)**:
  - **ResNet-50**: `model.layer4[-1]` (final residual bottleneck block).
  - **CBAM-ResNet50**: `model.layer4[-1].bottleneck.conv3` (final convolutional feature map prior to attention pooling and skip addition).
  - **EdgeNeXt-Base**: `model.stages[-1]` (final convolutional/transformer feature extraction stage).
- **Implementation**:
  - Dependency-free implementation using PyTorch forward (`register_forward_hook`) and full backward (`register_full_backward_hook`) hooks.
  - Computes global average-pooled gradients as channel importance weights, combined linearly with forward activation maps.
  - Applies positive rectification (ReLU) and min-max normalization to `[0.0, 1.0]`.
  - Overlays heatmaps on RGB inputs using OpenCV's JET colormap blending (`overlay_cam_on_image`) at `alpha = 0.5`.
  - Automatic hook cleanup in `finally` blocks prevents GPU/CPU memory accumulation.
  - Grad-CAM passes are computed strictly outside the timed inference pass to maintain latency benchmark integrity.

---

## Degradation Pipeline & Experiment Design

### Preprocessing (`degrade.py`)
- Standard evaluation transform: `Resize((224, 224))`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.
- Non-square input handling: `pad_to_square` centers images with white padding `(255, 255, 255)` to a 1:1 aspect ratio prior to evaluation (disabled by default).
- Downscaling (`downscale_long_side`): Images with a long side $> 512\text{ px}$ are resized proportionally (Lanczos) so the max dimension is $512\text{ px}$, maintaining consistent degradation intensity with the thesis test set (see *Why images are downscaled to 512 px*).

### Degradation Families & Presets
The app isolates evaluations by **Degradation Family**:
1. **Blur**: Gaussian blur with dynamic kernel size $k = 2 \cdot \lceil 3\sigma \rceil + 1$.
   - Clean: $\sigma = 0.0$ | Mild: $\sigma = 1.0$ | Moderate: $\sigma = 2.0$ | Severe: $\sigma = 3.0$
2. **Occlusion**: Random rectangular patch with gray fill value `128` (aspect ratio uniform in $[0.5, 2.0]$).
   - Clean: $0.00$ | Mild: $0.10$ | Moderate: $0.20$ | Severe: $0.30$
   - Includes a **Re-roll box** button to generate a new random seed and reposition occlusion without changing coverage.
3. **Darkening**: Illumination scaling $\alpha \in [0.30, 1.00]$ via $\text{clip}(I \cdot \alpha, 0, 255)$.
   - Clean: $\alpha = 1.00$ | Mild: $\alpha = 0.70$ | Moderate: $\alpha = 0.50$ | Severe: $\alpha = 0.30$
4. **Custom (combined)**: Allows multi-perturbation combinations (captioned with illustrative notice).

---

## Interactive Dashboard (`app.py`)

The Streamlit dashboard supports two core analytical modes:

- **Single Architecture Analysis**:
  - Side-by-side inspection of Clean input vs Degraded input.
  - High-contrast prediction badges with confidence percentages (calibrated with the raw value alongside, or raw only) and shift alerts when degradation flips classification.
  - Isolated forward-pass latency measurement with warmup pass and hardware synchronization.
  - Complete 6-class probability distribution bars.
  - Optional Grad-CAM attention map display for both clean and degraded states.
  - Degradation impact summary metrics (prediction stability, confidence delta, latency difference).

- **Cross-Architecture Robustness Comparison ("Compare All Three")**:
  - Simultaneous evaluation of EdgeNeXt-Base, CBAM-ResNet50, and ResNet-50 under identical degradation.
  - Direct side-by-side display of inputs, architecture complexity benchmarks, prediction badges, latency metrics, and optional Grad-CAM heatmaps.
  - Comparative summary table detailing parameters, FLOPs, clean top-1, degraded top-1, prediction retention status, confidence drop, and forward latency.

---

## Quickstart & Verification

> **Environment Note:** Verified on **Python 3.14.4** with pinned package versions specified in `requirements.txt`.

1. **Install Pinned Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Checkpoints**:
   ```bash
   python check_weights.py
   ```

   Then confirm parity with the thesis notebook:
   ```bash
   python parity_check.py
   ```

3. **Run Automated Batch Benchmark (`headline_test.py`)**:
   Evaluates all three architectures side-by-side across samples under clean vs severe darkening ($\alpha = 0.30$):
   ```bash
   python headline_test.py
   ```

4. **Add Demo Test Images (Optional)**:
   Place real TrashNet test images in the `samples/` directory:
   ```bash
   # Supports *.jpg, *.jpeg, *.png
   cp /path/to/trashnet/test/*.jpg samples/
   ```

5. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Alternatively, access the hosted deployment online at [waste-classification-robustness.streamlit.app](https://waste-classification-robustness.streamlit.app/).

---

## Thesis Contextual Note (Section 4.8)

> Models trained on TrashNet only; real-world photos are out-of-distribution. Thesis Section 4.8: EdgeNeXt-Base retained 38.99% of its clean accuracy on RW-TS (Seed 42); across three seeds, clean RW-TS accuracy was 36.51% (EdgeNeXt-Base), 23.72% (ResNet-50) and 19.22% (CBAM-ResNet50), against 94.91%, 93.77% and 91.40% on the clean TrashNet test set.
