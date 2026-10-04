import math
from typing import Union, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard evaluation preprocessing pipeline
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def apply_gaussian_blur(img_np: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using sigma.
    ksize is computed as 2 * ceil(3 * sigma) + 1.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return img_np.copy()

    ksize = 2 * math.ceil(3 * sigma) + 1
    # Ensure minimum ksize is 3 and odd
    if ksize < 3:
        ksize = 3
    elif ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_rectangle_occlusion(
    img_np: np.ndarray,
    coverage: float,
    prng: Optional[Union[np.random.Generator, np.random.RandomState, int]] = None
) -> np.ndarray:
    """
    Apply random rectangular occlusion with gray fill (value 128).
    Coverage is bounded in [0.0, 0.30], aspect ratio uniform in [0.5, 2.0].
    """
    coverage = float(coverage)
    if coverage <= 0.0:
        return img_np.copy()

    # Bounded between 0.0 and 0.30
    coverage = min(coverage, 0.30)

    # Initialize PRNG
    if prng is None:
        rng = np.random.default_rng()
    elif isinstance(prng, (int, np.integer)):
        rng = np.random.default_rng(prng)
    elif isinstance(prng, np.random.RandomState):
        rng = prng
    else:
        rng = prng

    h, w = img_np.shape[:2]
    total_area = h * w
    target_area = coverage * total_area

    # Aspect ratio uniform in [0.5, 2.0]
    if hasattr(rng, "uniform"):
        aspect_ratio = rng.uniform(0.5, 2.0)
    else:
        aspect_ratio = np.random.uniform(0.5, 2.0)

    # w_occ / h_occ = aspect_ratio  =>  w_occ * h_occ = target_area
    h_occ = int(round(math.sqrt(target_area / aspect_ratio)))
    w_occ = int(round(math.sqrt(target_area * aspect_ratio)))

    # Clamp bounding box dimensions to image dimensions
    h_occ = max(1, min(h, h_occ))
    w_occ = max(1, min(w, w_occ))

    max_y = h - h_occ
    max_x = w - w_occ

    if max_y > 0:
        if hasattr(rng, "integers"):
            y1 = int(rng.integers(0, max_y + 1))
        elif hasattr(rng, "randint"):
            y1 = int(rng.randint(0, max_y + 1))
        else:
            y1 = int(np.random.randint(0, max_y + 1))
    else:
        y1 = 0

    if max_x > 0:
        if hasattr(rng, "integers"):
            x1 = int(rng.integers(0, max_x + 1))
        elif hasattr(rng, "randint"):
            x1 = int(rng.randint(0, max_x + 1))
        else:
            x1 = int(np.random.randint(0, max_x + 1))
    else:
        x1 = 0

    y2 = y1 + h_occ
    x2 = x1 + w_occ

    out = img_np.copy()
    out[y1:y2, x1:x2] = 128
    return out


def apply_brightness_darkening(img_np: np.ndarray, alpha: float) -> np.ndarray:
    """
    Darken image by factor alpha: np.clip(img_np * alpha, 0, 255) with alpha in [0.3, 1.0].
    """
    alpha = max(0.3, min(float(alpha), 1.0))
    darkened = np.clip(img_np.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return darkened


def pad_to_square(
    img: Union[Image.Image, np.ndarray],
    fill_color: Tuple[int, ...] = (255, 255, 255)
) -> Union[Image.Image, np.ndarray]:
    """
    Center non-square image with white padding to achieve a 1:1 aspect ratio.
    Supports both PIL.Image and numpy.ndarray inputs.
    """
    if isinstance(img, Image.Image):
        w, h = img.size
        if w == h:
            return img
        max_dim = max(w, h)
        if img.mode == "RGB":
            bg_color = fill_color[:3]
        elif img.mode == "RGBA":
            bg_color = fill_color + (255,) if len(fill_color) == 3 else fill_color
        elif img.mode == "L":
            bg_color = fill_color[0]
        else:
            bg_color = fill_color

        square_img = Image.new(img.mode, (max_dim, max_dim), bg_color)
        left = (max_dim - w) // 2
        top = (max_dim - h) // 2
        square_img.paste(img, (left, top))
        return square_img

    elif isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        if w == h:
            return img.copy()

        max_dim = max(w, h)
        left = (max_dim - w) // 2
        top = (max_dim - h) // 2

        if img.ndim == 3:
            c = img.shape[2]
            fill = np.array(fill_color[:c], dtype=img.dtype)
            square_img = np.full((max_dim, max_dim, c), fill, dtype=img.dtype)
            square_img[top:top + h, left:left + w] = img
        else:
            square_img = np.full((max_dim, max_dim), fill_color[0], dtype=img.dtype)
            square_img[top:top + h, left:left + w] = img
        return square_img

    else:
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(img)}")
