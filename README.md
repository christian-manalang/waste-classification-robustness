# Visual Degradation Robustness in Waste Classification

A Streamlit-based interactive demonstration evaluating the robustness and inference latency of Convolutional Neural Networks (CNNs) and Vision Transformers under real-world visual perturbations for automated recycling systems.

Official repository, interactive evaluation dashboard, and artifact archive for the undergraduate thesis:  
**"A Comparative Study on the Robustness of Residual and Attention-Based Architectures for Waste Classification Under Degraded Conditions"**  
*School of Information Technology, Mapúa University (October 2026)*  
**Authors:** Nicko Gabriel A. Baldo, Jorge Christian B. Manalang, Liandro E. Refulle

**Live Application:** [waste-classification-robustness.streamlit.app](https://waste-classification-robustness.streamlit.app/)

---

## Interactive & Reproducible Colab Notebooks

To ensure experimental transparency and multi-seed variance validation across all seven methodological phases, the complete execution pipelines are available via view-only Google Colab links:

| Execution Run | Target Seed | Description | Notebook Access |
| :--- | :---: | :--- | :---: |
| **1st Run (Representative)** | `Seed 42` | Primary baseline, Phase 5 retraining, Phase 6 ablation, and Phase 7 RW-TS evaluations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uJbjeF5og_6YRoJ6LedLHtmLDIFNF_48?usp=sharing) |
| **2nd Run (Variance Replication 1)** | `Seed 44` | Independent model re-initialization and optimization run across identical partitions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RQLwQtvX5GVjXUPQ16XgJ8HML6olVGfY?usp=sharing) |
| **3rd Run (Variance Replication 2)** | `Seed 43` | Independent model re-initialization and optimization run across identical partitions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XXxbjIcpSi3OGD01RQCkNPatp4C-dguR?usp=sharing) |

---

## Project Architecture & Files

```text
waste-robustness-demo/
├── .gitignore           # Git ignore exclusions
├── app.py               # Streamlit wide-layout interactive robustness dashboard
├── cam.py               # Lightweight Grad-CAM implementation and visualization helpers
├── check_weights.py     # Checkpoint verification script (strict loading & parameter counts)
├── degrade.py           # Preprocessing pipeline and visual degradation functions
├── headline_test.py     # Automated batch evaluation under darkening (alpha=0.3)
├── models.py            # Neural network architectures, CBAM wrappers, strict weight loaders
├── requirements.txt     # Pinned Python project dependencies (Python 3.14.4)
├── weights/             # Directory for trained .pth checkpoints
│   ├── resnet50_best.pth
│   ├── cbam_resnet50_best.pth
│   └── edgenext_base_best.pth
├── samples/             # Directory for real TrashNet test images (*.jpg, *.png)
└── README.md            # Complete project documentation and guide
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
To verify all checkpoints against the exact architectural definitions and confirm that weights load with `strict=True`:

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

====================================================================
  All model checkpoints loaded and verified successfully.
====================================================================
```

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
- Upload downscaling: Uploaded images with a long side $> 512\text{ px}$ are resized proportionally so the max dimension is $512\text{ px}$, maintaining consistent degradation intensity with the thesis test set.

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
  - High-contrast prediction badges with confidence percentages and shift alerts when degradation flips classification.
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

## Thesis Contextual Note (§4.8)

> *"Models trained on TrashNet only; real-world photos are out-of-distribution (see thesis §4.8: EdgeNeXt retains 38.99% of its clean accuracy on RW-TS)."*
