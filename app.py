import glob
import os
import random
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image

from models import CLASSES, load_weights
from degrade import (
    eval_transform,
    apply_gaussian_blur,
    apply_rectangle_occlusion,
    apply_brightness_darkening,
    pad_to_square,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Visual Degradation Robustness in Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper to render images using width='stretch' where supported or fallback without deprecation
def render_image(img: Image.Image, caption: Optional[str] = None) -> None:
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


# -----------------------------------------------------------------------------
# Model Specifications & Complexity Benchmarks
# -----------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "EdgeNeXt-Base": {
        "arch": "edgenext_base",
        "ckpt_path": "weights/edgenext_base_best.pth",
        "params": "17.92 M",
        "gflops": "2.925 GFLOPs",
        "badge_color": "#2563eb",
        "tag": "Hybrid CNN-Transformer",
    },
    "CBAM-ResNet50": {
        "arch": "cbam_resnet50",
        "ckpt_path": "weights/cbam_resnet50_best.pth",
        "params": "26.04 M",
        "gflops": "4.144 GFLOPs",
        "badge_color": "#7c3aed",
        "tag": "Attention-Augmented CNN",
    },
    "ResNet-50": {
        "arch": "resnet50",
        "ckpt_path": "weights/resnet50_best.pth",
        "params": "23.52 M",
        "gflops": "4.132 GFLOPs",
        "badge_color": "#059669",
        "tag": "Standard Residual Network",
    },
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(arch_key: str, ckpt_path: str) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load model weights cached on the available device (CUDA or CPU).
    Loads from the specific checkpoint path defined in MODEL_CONFIGS with strict=True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(arch=arch_key, ckpt_path=ckpt_path, device=device)
    return model, device


# Warmup cache in session state
if "warmed_up_models" not in st.session_state:
    st.session_state.warmed_up_models = set()


def run_inference(
    model: torch.nn.Module,
    device: torch.device,
    pil_image: Image.Image,
    model_name: str,
) -> Dict[str, Any]:
    """
    Execute forward pass with precise latency measurement.
    Warms up once per model (tracked in session state) and times only the raw forward pass.
    Softmax and CPU conversions occur strictly AFTER the timer stops.
    """
    tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    # Warmup pass once per model
    if model_name not in st.session_state.warmed_up_models:
        with torch.no_grad():
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        st.session_state.warmed_up_models.add(model_name)

    # Measured inference pass: time only the raw forward pass logits = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Compute softmax and .cpu() AFTER the timer stops
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = int(np.argmax(probs))

    return {
        "probs": probs,
        "top_idx": top_idx,
        "top_class": CLASSES[top_idx],
        "top_prob": float(probs[top_idx]),
        "latency_ms": latency_ms,
    }


def render_prediction_badge(
    top_class: str,
    top_prob: float,
    is_degraded: bool = False,
    baseline_class: Optional[str] = None,
) -> None:
    """
    Render high-contrast styled top prediction badge with class icon and confidence.
    """
    pct = top_prob * 100.0
    icon = CLASS_ICONS.get(top_class, "🏷️")

    if is_degraded and baseline_class is not None and top_class != baseline_class:
        base_icon = CLASS_ICONS.get(baseline_class, "🏷️")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
                        color: white; padding: 12px 16px; border-radius: 8px; margin: 10px 0;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">
                    ⚠️ Prediction Shifted Under Degradation
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; margin-top: 2px;">
                    {icon} {top_class.upper()} ({pct:.1f}%)
                </div>
                <div style="font-size: 0.85rem; opacity: 0.92; margin-top: 4px;">
                    Baseline clean was: <b>{base_icon} {baseline_class.upper()}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        bg_gradient = (
            "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)"
            if is_degraded
            else "linear-gradient(135deg, #059669 0%, #047857 100%)"
        )
        subtitle = "Degraded Input Prediction" if is_degraded else "Original Input Baseline"
        st.markdown(
            f"""
            <div style="background: {bg_gradient};
                        color: white; padding: 12px 16px; border-radius: 8px; margin: 10px 0;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">
                    {subtitle}
                </div>
                <div style="font-size: 1.35rem; font-weight: 700; margin-top: 2px;">
                    {icon} {top_class.upper()} ({pct:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_bars(probs: np.ndarray, top_idx: int) -> None:
    """
    Render class confidence probability bars across all 6 waste categories.
    """
    for idx, cls_name in enumerate(CLASSES):
        prob = float(probs[idx])
        pct = prob * 100.0
        icon = CLASS_ICONS.get(cls_name, "🏷️")
        is_top = (idx == top_idx)

        weight = "700" if is_top else "400"
        color = "#10b981" if is_top else "inherit"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                        margin-bottom: 2px; font-weight: {weight}; color: {color};">
                <span>{icon} {cls_name.capitalize()}</span>
                <span style="font-family: monospace;">{pct:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("Visual Degradation Robustness in Waste Classification")
st.markdown(
    "Evaluating neural network robustness and inference latency under real-world visual perturbations "
    "(sensor blur, occlusions, and severe illumination changes) for automated recycling sorting systems."
)

# -----------------------------------------------------------------------------
# Static Complexity Badges
# -----------------------------------------------------------------------------
st.markdown("#### 📐 Architecture Complexity Benchmarks")
badge_col1, badge_col2, badge_col3 = st.columns(3)

with badge_col1:
    st.markdown(
        """
        <div style="background-color: rgba(37, 99, 235, 0.08); border-left: 4px solid #2563eb;
                    padding: 12px; border-radius: 6px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #2563eb;">EdgeNeXt-Base</div>
            <div style="margin-top: 4px; font-size: 0.95rem;">
                <b>Parameters:</b> <code>17.92 M</code> &nbsp;|&nbsp; <b>FLOPs:</b> <code>2.925 GFLOPs</code>
            </div>
            <div style="font-size: 0.82rem; opacity: 0.75; margin-top: 2px;">
                Hybrid CNN & Vision Transformer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with badge_col2:
    st.markdown(
        """
        <div style="background-color: rgba(124, 58, 237, 0.08); border-left: 4px solid #7c3aed;
                    padding: 12px; border-radius: 6px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #7c3aed;">CBAM-ResNet50</div>
            <div style="margin-top: 4px; font-size: 0.95rem;">
                <b>Parameters:</b> <code>26.04 M</code> &nbsp;|&nbsp; <b>FLOPs:</b> <code>4.144 GFLOPs</code>
            </div>
            <div style="font-size: 0.82rem; opacity: 0.75; margin-top: 2px;">
                Channel & Spatial Attention CNN
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with badge_col3:
    st.markdown(
        """
        <div style="background-color: rgba(5, 150, 105, 0.08); border-left: 4px solid #059669;
                    padding: 12px; border-radius: 6px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #059669;">ResNet-50</div>
            <div style="margin-top: 4px; font-size: 0.95rem;">
                <b>Parameters:</b> <code>23.52 M</code> &nbsp;|&nbsp; <b>FLOPs:</b> <code>4.132 GFLOPs</code>
            </div>
            <div style="font-size: 0.82rem; opacity: 0.75; margin-top: 2px;">
                Standard Deep Residual Baseline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Controls & Degradation Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Experiment Controls")

# 1. Model Selection
model_choice = st.sidebar.selectbox(
    "Architecture Selection",
    options=["EdgeNeXt-Base", "CBAM-ResNet50", "ResNet-50", "Compare All Three"],
    index=0,  # EdgeNeXt-Base default
    help="Select an individual architecture or evaluate all three models simultaneously",
)

st.sidebar.markdown("---")

# 2. Image Source
st.sidebar.subheader("📷 Image Input")
image_source = st.sidebar.radio(
    "Input Method",
    options=["Upload Image", "Use Demo Sample"],
    index=0,
    horizontal=True,
)

raw_image: Optional[Image.Image] = None
if image_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload an image of recyclable waste",
    )
    if uploaded_file is not None:
        try:
            raw_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded image: {e}")
    else:
        st.sidebar.info("💡 Upload an image or switch to 'Use Demo Sample' to test.")
else:
    sample_patterns = ["samples/*.jpg", "samples/*.jpeg", "samples/*.png", "samples/*.JPG", "samples/*.PNG"]
    sample_files = []
    for pat in sample_patterns:
        sample_files.extend(glob.glob(pat))
    sample_files = sorted(list(set(sample_files)))

    if not sample_files:
        st.sidebar.info("Place real TrashNet test images in samples/ to use demo presets.")
    else:
        sample_choice = st.sidebar.selectbox(
            "Choose Demo Sample",
            options=sample_files,
            format_func=lambda x: os.path.basename(x),
        )
        try:
            raw_image = Image.open(sample_choice).convert("RGB")
        except Exception as e:
            st.sidebar.error(f"Failed to load sample image: {e}")

# Downscale uploaded images if long side exceeds 512 px
was_downscaled = False
if raw_image is not None:
    max_side = max(raw_image.width, raw_image.height)
    if max_side > 512:
        scale = 512.0 / max_side
        new_w = max(1, int(round(raw_image.width * scale)))
        new_h = max(1, int(round(raw_image.height * scale)))
        raw_image = raw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        was_downscaled = True

# 3. Preprocessing Toggle (Default: False)
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Preprocessing")
pad_to_square_toggle = st.sidebar.checkbox(
    "Pad to Square (1:1 Aspect Ratio)",
    value=False,
    help="Center non-square images with white padding (255, 255, 255) prior to resizing to 224x224",
)

# 4. Perturbation Settings & Degradation Families
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Visual Degradation Settings")

degradation_family = st.sidebar.selectbox(
    "Degradation Family",
    options=["Blur", "Occlusion", "Darkening", "Custom (combined)"],
    index=0,
    key="degradation_family",
    help="Select an isolated thesis degradation family or custom combined perturbations",
)

if degradation_family == "Custom (combined)":
    st.sidebar.caption("Combined degradations were not evaluated in the thesis; results here are illustrative only.")

# Initialize degradation values in session state
if "blur_val" not in st.session_state:
    st.session_state.blur_val = 0.0
if "occlusion_val" not in st.session_state:
    st.session_state.occlusion_val = 0.00
if "brightness_alpha" not in st.session_state:
    st.session_state.brightness_alpha = 1.00
if "occlusion_seed" not in st.session_state:
    st.session_state.occlusion_seed = 42


def apply_family_preset(level: str) -> None:
    """Set ONLY the parameter for the active family and reset the other two."""
    family = st.session_state.get("degradation_family", "Blur")

    if family == "Blur":
        st.session_state.occlusion_val = 0.00
        st.session_state.brightness_alpha = 1.00
        st.session_state.blur_val = {"Clean": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}[level]
    elif family == "Occlusion":
        st.session_state.blur_val = 0.0
        st.session_state.brightness_alpha = 1.00
        st.session_state.occlusion_val = {"Clean": 0.00, "Mild": 0.10, "Moderate": 0.20, "Severe": 0.30}[level]
    elif family == "Darkening":
        st.session_state.blur_val = 0.0
        st.session_state.occlusion_val = 0.00
        st.session_state.brightness_alpha = {"Clean": 1.00, "Mild": 0.70, "Moderate": 0.50, "Severe": 0.30}[level]
    else:  # Custom (combined)
        st.session_state.blur_val = {"Clean": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}[level]
        st.session_state.occlusion_val = {"Clean": 0.00, "Mild": 0.10, "Moderate": 0.20, "Severe": 0.30}[level]
        st.session_state.brightness_alpha = {"Clean": 1.00, "Mild": 0.70, "Moderate": 0.50, "Severe": 0.30}[level]


# Preset Buttons with State Synchronization
st.sidebar.markdown("**Perturbation Presets**")
preset_c1, preset_c2, preset_c3, preset_c4 = st.sidebar.columns(4)

with preset_c1:
    if st.button("Clean", use_container_width=True, help="Baseline clean input"):
        apply_family_preset("Clean")
with preset_c2:
    if st.button("Mild", use_container_width=True, help="Mild degradation preset"):
        apply_family_preset("Mild")
with preset_c3:
    if st.button("Moderate", use_container_width=True, help="Moderate degradation preset"):
        apply_family_preset("Moderate")
with preset_c4:
    if st.button("Severe", use_container_width=True, help="Severe degradation preset"):
        apply_family_preset("Severe")

blur_val = st.sidebar.slider(
    "Gaussian Blur (σ)",
    min_value=0.0,
    max_value=3.0,
    step=0.1,
    key="blur_val",
    help="Gaussian blur standard deviation σ. Kernel size = 2*ceil(3*σ)+1",
)

occlusion_val = st.sidebar.slider(
    "Occlusion Coverage",
    min_value=0.00,
    max_value=0.30,
    step=0.01,
    format="%.2f",
    key="occlusion_val",
    help="Random rectangular occlusion with gray fill (value 128) covering 0.00 to 0.30 of the area",
)

# Occlusion Re-roll Button
if st.sidebar.button("Re-roll box", use_container_width=True, help="Regenerate random seed and re-trigger occlusion placement"):
    st.session_state.occlusion_seed = random.randint(1, 1_000_000)

st.sidebar.caption(f"🎲 Occlusion Seed: `{st.session_state.occlusion_seed}`")

brightness_alpha = st.sidebar.slider(
    "Brightness Darkening (α)",
    min_value=0.30,
    max_value=1.00,
    step=0.05,
    format="%.2f",
    key="brightness_alpha",
    help="Darkening scalar factor α. Scaled as np.clip(img * α, 0, 255)",
)

# -----------------------------------------------------------------------------
# Check Missing Weights: Halt Execution if Checkpoint Missing
# -----------------------------------------------------------------------------
if model_choice != "Compare All Three":
    active_ckpt = MODEL_CONFIGS[model_choice]["ckpt_path"]
    if not os.path.isfile(active_ckpt):
        st.error(
            f"❌ **Missing Weight File**: Checkpoint file for **{model_choice}** was not found at `{active_ckpt}`. "
            "Execution halted. Please place the required `.pth` checkpoint in the `weights/` directory."
        )
        st.stop()
else:
    missing_ckpts = [
        f"{name} (`{cfg['ckpt_path']}`)"
        for name, cfg in MODEL_CONFIGS.items()
        if not os.path.isfile(cfg["ckpt_path"])
    ]
    if missing_ckpts:
        st.error(
            f"❌ **Missing Weight File(s)**: Checkpoint(s) missing for comparison mode: {', '.join(missing_ckpts)}. "
            "Execution halted. Please place all required `.pth` checkpoints in the `weights/` directory."
        )
        st.stop()

# -----------------------------------------------------------------------------
# Image Degradation Pipeline
# -----------------------------------------------------------------------------
if raw_image is None:
    st.info("👋 Please upload an image from the sidebar or select a Demo Sample to begin analysis.")
    st.stop()

if was_downscaled:
    st.info("ℹ️ Image downscaled to 512 px (long side) so degradation strength matches the thesis test set.")

# Apply pad-to-square if enabled
if pad_to_square_toggle:
    clean_image_pil = pad_to_square(raw_image, fill_color=(255, 255, 255))
else:
    clean_image_pil = raw_image.copy()

# Apply degradations sequentially
img_np = np.array(clean_image_pil.convert("RGB"))
degraded_np = img_np.copy()

if blur_val > 0.0:
    degraded_np = apply_gaussian_blur(degraded_np, sigma=blur_val)

if occlusion_val > 0.0:
    degraded_np = apply_rectangle_occlusion(
        degraded_np,
        coverage=float(occlusion_val),
        prng=st.session_state.occlusion_seed,
    )

if brightness_alpha < 1.0:
    degraded_np = apply_brightness_darkening(degraded_np, alpha=brightness_alpha)

degraded_image_pil = Image.fromarray(degraded_np)

# -----------------------------------------------------------------------------
# Single Model Mode vs Compare All Three Mode
# -----------------------------------------------------------------------------
if model_choice != "Compare All Three":
    # -------------------------------------------------------------------------
    # Mode A: Single Architecture Analysis
    # -------------------------------------------------------------------------
    selected_cfg = MODEL_CONFIGS[model_choice]
    model, device = get_model(selected_cfg["arch"], selected_cfg["ckpt_path"])

    # Run inferences with warmup tracking and timer around raw forward pass
    clean_res = run_inference(model, device, clean_image_pil, model_name=model_choice)
    degraded_res = run_inference(model, device, degraded_image_pil, model_name=model_choice)

    st.subheader(f"🔍 Model Evaluation: {model_choice}")
    st.caption(
        f"Complexity: **{selected_cfg['params']}** Parameters | **{selected_cfg['gflops']}** | "
        f"Device: `{device.type.upper()}` | Checkpoint: `{selected_cfg['ckpt_path']}`"
    )

    col_orig, col_deg = st.columns(2)

    with col_orig:
        st.markdown("### 📷 Original (Clean) Input")
        render_image(clean_image_pil, caption=f"Size: {clean_image_pil.size[0]}x{clean_image_pil.size[1]}")
        render_prediction_badge(
            top_class=clean_res["top_class"],
            top_prob=clean_res["top_prob"],
            is_degraded=False,
        )
        st.markdown(
            f"""
            <div style="background-color: rgba(100, 116, 139, 0.12); padding: 6px 12px;
                        border-radius: 6px; display: inline-block; margin-bottom: 12px; font-size: 0.92rem;">
                ⚡ <b>Forward Latency:</b> <code>{clean_res['latency_ms']:.2f} ms</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Class Probabilities")
        render_probability_bars(clean_res["probs"], clean_res["top_idx"])

    with col_deg:
        st.markdown("### 🧪 Degraded Input")
        degradation_desc = []
        if blur_val > 0.0:
            degradation_desc.append(f"Blur σ={blur_val:.1f}")
        if occlusion_val > 0.0:
            degradation_desc.append(f"Occlusion {occlusion_val:.2f}")
        if brightness_alpha < 1.0:
            degradation_desc.append(f"Brightness α={brightness_alpha:.2f}")
        caption_text = f"Degradations: {', '.join(degradation_desc)}" if degradation_desc else "No degradations applied (Clean)"

        render_image(degraded_image_pil, caption=caption_text)
        render_prediction_badge(
            top_class=degraded_res["top_class"],
            top_prob=degraded_res["top_prob"],
            is_degraded=True,
            baseline_class=clean_res["top_class"],
        )
        st.markdown(
            f"""
            <div style="background-color: rgba(100, 116, 139, 0.12); padding: 6px 12px;
                        border-radius: 6px; display: inline-block; margin-bottom: 12px; font-size: 0.92rem;">
                ⚡ <b>Forward Latency:</b> <code>{degraded_res['latency_ms']:.2f} ms</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Class Probabilities")
        render_probability_bars(degraded_res["probs"], degraded_res["top_idx"])

    # Degradation Impact Summary
    st.markdown("---")
    st.markdown("#### 📈 Degradation Impact Summary")
    conf_delta = (degraded_res["top_prob"] - clean_res["top_prob"]) * 100.0
    is_preserved = clean_res["top_class"] == degraded_res["top_class"]

    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.metric(
            label="Prediction Robustness",
            value="Preserved" if is_preserved else "Shifted",
            delta="Stable" if is_preserved else f"Flipped to {degraded_res['top_class']}",
            delta_color="normal" if is_preserved else "inverse",
        )
    with sum_col2:
        st.metric(
            label="Top Confidence Delta",
            value=f"{degraded_res['top_prob'] * 100:.1f}%",
            delta=f"{conf_delta:+.1f}% vs Clean",
            delta_color="normal" if conf_delta >= 0 else "off",
        )
    with sum_col3:
        st.metric(
            label="Forward Pass Latency",
            value=f"{degraded_res['latency_ms']:.2f} ms",
            delta=f"{degraded_res['latency_ms'] - clean_res['latency_ms']:+.2f} ms",
            delta_color="off",
        )

else:
    # -------------------------------------------------------------------------
    # Mode B: "Compare All Three" Models Mode
    # -------------------------------------------------------------------------
    st.subheader("📊 Cross-Architecture Robustness Comparison")

    # Display original and degraded images side by side
    st.markdown("#### 🖼️ Visual Input Comparison")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("##### 📷 Original (Clean) Image")
        render_image(clean_image_pil, caption=f"Size: {clean_image_pil.size[0]}x{clean_image_pil.size[1]}")

    with img_col2:
        st.markdown("##### 🧪 Degraded Image")
        deg_labels = []
        if blur_val > 0.0:
            deg_labels.append(f"Blur σ={blur_val:.1f}")
        if occlusion_val > 0.0:
            deg_labels.append(f"Occlusion {occlusion_val:.2f}")
        if brightness_alpha < 1.0:
            deg_labels.append(f"Brightness α={brightness_alpha:.2f}")
        render_image(
            degraded_image_pil,
            caption="Applied: " + (", ".join(deg_labels) if deg_labels else "None (Clean)"),
        )

    st.markdown("---")
    st.markdown("#### 🤖 Model Inference & Robustness Metrics")

    # Evaluate all three architectures
    results: Dict[str, Dict[str, Any]] = {}
    for name, cfg in MODEL_CONFIGS.items():
        m, dev = get_model(cfg["arch"], cfg["ckpt_path"])
        clean_out = run_inference(m, dev, clean_image_pil, model_name=name)
        deg_out = run_inference(m, dev, degraded_image_pil, model_name=name)
        results[name] = {
            "clean": clean_out,
            "degraded": deg_out,
            "cfg": cfg,
            "device": dev,
        }

    # Render results in 3 columns
    arch_cols = st.columns(3)
    for idx, (arch_name, res) in enumerate(results.items()):
        with arch_cols[idx]:
            cfg = res["cfg"]
            clean_out = res["clean"]
            deg_out = res["degraded"]

            st.markdown(
                f"""
                <div style="background-color: rgba(100, 116, 139, 0.08); border-top: 4px solid {cfg['badge_color']};
                            padding: 10px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="font-weight: 700; font-size: 1.1rem; color: {cfg['badge_color']};">{arch_name}</div>
                    <div style="font-size: 0.82rem; margin-top: 2px;">
                        <b>{cfg['params']}</b> | <b>{cfg['gflops']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top prediction badge on degraded image
            render_prediction_badge(
                top_class=deg_out["top_class"],
                top_prob=deg_out["top_prob"],
                is_degraded=True,
                baseline_class=clean_out["top_class"],
            )

            st.markdown(
                f"""
                <div style="background-color: rgba(100, 116, 139, 0.12); padding: 5px 10px;
                            border-radius: 6px; display: inline-block; margin-bottom: 10px; font-size: 0.9rem;">
                    ⚡ <b>Latency:</b> <code>{deg_out['latency_ms']:.2f} ms</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Class Probabilities")
            render_probability_bars(deg_out["probs"], deg_out["top_idx"])

    # Side-by-side Summary Table
    st.markdown("---")
    st.markdown("#### 📋 Comparative Summary Table")

    summary_rows = []
    for arch_name, res in results.items():
        clean_out = res["clean"]
        deg_out = res["degraded"]
        cfg = res["cfg"]
        is_same = clean_out["top_class"] == deg_out["top_class"]
        status_str = "✅ Retained" if is_same else f"❌ Shifted to {deg_out['top_class']}"
        conf_drop = (clean_out["top_prob"] - deg_out["top_prob"]) * 100.0

        summary_rows.append(
            {
                "Architecture": arch_name,
                "Parameters": cfg["params"],
                "GFLOPs": cfg["gflops"],
                "Clean Top-1": f"{clean_out['top_class']} ({clean_out['top_prob'] * 100:.1f}%)",
                "Degraded Top-1": f"{deg_out['top_class']} ({deg_out['top_prob'] * 100:.1f}%)",
                "Prediction Status": status_str,
                "Confidence Drop": f"{conf_drop:+.1f}%",
                "Latency": f"{deg_out['latency_ms']:.2f} ms",
            }
        )

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Footer: Thesis Chapter 4 Contextual Note
# -----------------------------------------------------------------------------
st.markdown("<hr style='margin: 36px 0 16px 0;'>", unsafe_allow_html=True)

st.info(
    "📌 **Thesis Context (§4.8)**: Models trained on TrashNet only; real-world photos are out-of-distribution "
    "(see thesis §4.8: EdgeNeXt retains 38.99% of its clean accuracy on RW-TS).",
    icon="🔬",
)

st.caption(
    "Models trained on TrashNet only; real-world photos are out-of-distribution "
    "(see thesis §4.8: EdgeNeXt retains 38.99% of its clean accuracy on RW-TS)."
)
